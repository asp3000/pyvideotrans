# -*- coding: utf-8 -*-
"""
智能断句模块（借鉴 VideoCaptioner 的双层方案）
================================================
提供两种断句算法，由高级设置 `resegment_mode` 切换：
  - local : 本地规则层算法（时间间隔分组 + 连接词切分 + 最大间隔二分拆分），零外部依赖
  - llm   : LLM 语义断句（agent loop 自动验证反馈 + difflib 滑动窗口回填字级时间戳）

调度入口 smart_resegment()：
  1. resegment_mode == 'llm' 且已配置可用 LLM 渠道（高级设置-LLM纠错渠道 且已填密钥）→ LLM 断句
  2. LLM 未配置 / 调用失败 / 校验不通过 → 自动降级为本地规则断句
  3. 本地规则也异常 → 返回 None，由调用方回退到原 _resegment 逻辑
所有识别渠道（faster-whisper / openai / qwen / deepgram / elevenlabs / fishaudio / gemini 等）
共用 _resegment 入口，因此均自动获得本能力。
"""
import re
import difflib
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional

from videotrans.configure.config import settings, logger

# 与 _stt_utils 保持一致的无空格语言集合
NO_SPACE_LANGS = {'zh', "zh-cn", "zh-tw", 'ja', 'th', 'yue', 'ko', 'km'}

# ==================== 规则层常量（源自 VideoCaptioner split.py） ====================
MAX_WORD_COUNT_CJK = 25        # CJK 文本单行最大字数（规则层）
MAX_WORD_COUNT_ENGLISH = 18    # 英文单行最大单词数（规则层）
SEGMENT_WORD_THRESHOLD = 500   # 长文本分块阈值（字数）
MAX_GAP = 1500                 # 回填时允许的最大时间间隔(ms)
RULE_SPLIT_GAP = 500           # 规则分割时间间隔阈值(ms)
TIME_GAP_WINDOW_SIZE = 5       # 时间间隔滑动窗口大小
TIME_GAP_MULTIPLIER = 3        # 大间隔判断倍数
MIN_GROUP_SIZE = 5             # 异常大间隔分组的最小组大小
SPLIT_SEARCH_RANGE = 30        # 分块时分割点前后搜索范围
PREFIX_WORD_RATIO = 0.6        # 前缀连接词分割比例
SUFFIX_WORD_RATIO = 0.4        # 后缀词分割比例
RULE_MIN_SEGMENT_SIZE = 4      # 规则分割最小片段数

# ==================== LLM 层常量 ====================
LLM_MAX_STEPS = 2                          # agent loop 最大重试次数
LLM_CONCURRENT = 6                         # 分块并发数
LLM_SIMILARITY_THRESHOLD = 0.96            # 内容一致性阈值
MATCH_SIMILARITY_THRESHOLD = 0.5           # 时间戳回填匹配相似度阈值
MATCH_MAX_SHIFT = 30                       # 回填滑动窗口最大偏移
MATCH_MAX_UNMATCHED = 5                    # 允许的最大未匹配句数
MATCH_LARGE_SHIFT = 100                    # 未匹配时的大偏移量

# LLM 单行字数上限（可被 settings 覆盖：llm_split_cjk_len / llm_split_en_len）
LLM_CJK_LEN_DEFAULT = 15
LLM_EN_LEN_DEFAULT = 18

_CJK_RE = re.compile(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]')

# 连接词表（源自 VideoCaptioner）
PREFIX_SPLIT_WORDS = {
    "and", "or", "but", "if", "then", "because", "as", "until", "while", "what",
    "when", "where", "nor", "yet", "so", "for", "however", "moreover",
    "和", "及", "与", "但", "而", "或", "因", "我", "你", "他", "她", "它",
    "咱", "您", "这", "那", "哪",
}
SUFFIX_SPLIT_WORDS = {
    ".", ",", "!", "?", "。", "，", "！", "？",
    "的", "了", "着", "过", "吗", "呢", "吧", "啊", "呀", "嘛", "啦",
    "mine", "yours", "hers", "its", "ours", "theirs", "either", "neither",
}


# ==================== 基础工具 ====================

def _is_cjk_text(text: str) -> bool:
    """判断文本是否以 CJK 为主"""
    cjk = len(_CJK_RE.findall(text))
    total = len(re.sub(r'\s', '', text))
    return total > 0 and cjk / total >= 0.3


def _count_words(text: str, is_cjk: bool) -> int:
    return len(re.sub(r'\s', '', text)) if is_cjk else len(text.split())


def _flatten_words(texts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """把 whisper 等返回的字级时间戳展平为毫秒列表
    word: {'word': str, 'start_ms': int, 'end_ms': int}
    """
    all_words = []
    for item in texts or []:
        for w in item.get('words', []):
            wt = str(w.get('word', ''))
            if not wt.strip():
                continue
            try:
                s = int(round(float(w['start']) * 1000))
                e = int(round(float(w['end']) * 1000))
            except (KeyError, TypeError, ValueError):
                continue
            all_words.append({'word': wt, 'start_ms': s, 'end_ms': max(s, e)})
    return all_words


def _join_words(words: List[Dict[str, Any]], use_space: bool) -> str:
    parts = [w['word'].strip() for w in words]
    if use_space:
        text = " ".join(p for p in parts if p)
        text = re.sub(r'\s+([.,?!:;])', r'\1', text)
        return text
    return "".join(parts)


def _normalize(s: str) -> str:
    """匹配前文本标准化：小写 + 空格规范化"""
    return " ".join(s.lower().split())


# ==================== 本地规则层（VideoCaptioner 规则算法） ====================

def _group_by_time_gaps(words: List[Dict[str, Any]], max_gap: int = RULE_SPLIT_GAP,
                        check_large_gaps: bool = False) -> List[List[int]]:
    """按时间间隔分组，返回索引分组列表"""
    if not words:
        return []
    result = []
    current = [0]
    recent_gaps = []
    for i in range(1, len(words)):
        gap = words[i]['start_ms'] - words[i - 1]['end_ms']
        if check_large_gaps:
            recent_gaps.append(gap)
            if len(recent_gaps) > TIME_GAP_WINDOW_SIZE:
                recent_gaps.pop(0)
            if len(recent_gaps) == TIME_GAP_WINDOW_SIZE:
                avg = sum(recent_gaps) / len(recent_gaps)
                if gap > avg * TIME_GAP_MULTIPLIER and len(current) > MIN_GROUP_SIZE:
                    result.append(current)
                    current = []
                    recent_gaps = []
        if gap > max_gap:
            result.append(current)
            current = []
            recent_gaps = []
        current.append(i)
    if current:
        result.append(current)
    return result


def _split_by_common_words(idx_group: List[int], words: List[Dict[str, Any]],
                           max_cjk: int, max_en: int) -> List[List[int]]:
    """在常见连接词/标点处分割长组，返回索引分组列表"""
    result = []
    current = []
    group_text = _join_words([words[i] for i in idx_group], True)
    is_cjk = _is_cjk_text(group_text)
    limit = max_cjk if is_cjk else max_en
    for pos, i in enumerate(idx_group):
        w_text = words[i]['word'].strip().lower()
        # 前缀连接词：在其前面分割
        if any(w_text.startswith(p) for p in PREFIX_SPLIT_WORDS) \
                and len(current) >= int(limit * PREFIX_WORD_RATIO):
            result.append(current)
            current = []
        # 后缀词：在上一个词后面分割
        elif pos > 0:
            prev_text = words[idx_group[pos - 1]]['word'].strip().lower()
            if any(prev_text.endswith(s) for s in SUFFIX_SPLIT_WORDS) \
                    and len(current) >= int(limit * SUFFIX_WORD_RATIO):
                result.append(current)
                current = []
        current.append(i)
    if current:
        result.append(current)
    return result


def _split_long_group(idx_group: List[int], words: List[Dict[str, Any]],
                      max_cjk: int, max_en: int) -> List[List[int]]:
    """超长组递归二分拆分：在最大时间间隔点下刀，返回索引分组列表"""
    out = []
    queue = [idx_group]
    while queue:
        group = queue.pop(0)
        if not group:
            continue
        text = _join_words([words[i] for i in group], True)
        is_cjk = _is_cjk_text(text)
        limit = max_cjk if is_cjk else max_en
        n = len(group)
        if _count_words(text, is_cjk) <= limit or n < RULE_MIN_SEGMENT_SIZE:
            out.append(group)
            continue
        gaps = [words[group[i + 1]]['start_ms'] - words[group[i]]['end_ms'] for i in range(n - 1)]
        all_equal = all(abs(g - gaps[0]) < 1e-6 for g in gaps)
        if all_equal:
            split_at = n // 2
        else:
            start_i = max(n // 6, 1)
            end_i = min((5 * n) // 6, n - 2)
            split_at = max(range(start_i, end_i),
                           key=lambda i: gaps[i], default=n // 2)
            if split_at <= 0 or split_at >= n - 1:
                split_at = n // 2
        queue.append(group[:split_at + 1])
        queue.append(group[split_at + 1:])
    return out


def rule_split(all_words: List[Dict[str, Any]], use_space: bool) -> List[List[int]]:
    """本地规则断句：输入展平词列表，返回索引分组列表"""
    max_cjk = MAX_WORD_COUNT_CJK
    max_en = MAX_WORD_COUNT_ENGLISH
    # 1. 按时间间隔分组
    groups = _group_by_time_gaps(all_words, RULE_SPLIT_GAP, check_large_gaps=True)
    # 2. 超长组按连接词分割
    groups2 = []
    for g in groups:
        text = _join_words([all_words[i] for i in g], use_space)
        limit = max_cjk if _is_cjk_text(text) else max_en
        if _count_words(text, _is_cjk_text(text)) > limit:
            groups2.extend(_split_by_common_words(g, all_words, max_cjk, max_en))
        else:
            groups2.append(g)
    # 3. 仍然超长的组二分拆分
    out = []
    for g in groups2:
        out.extend(_split_long_group(g, all_words, max_cjk, max_en))
    out.sort(key=lambda g: all_words[g[0]]['start_ms'])
    return out


# ==================== LLM 层（VideoCaptioner 四件套） ====================

def _build_chat_fn(uuid: Optional[str] = None) -> Optional[Callable]:
    """根据高级设置里的 LLM 纠错渠道构建 chat(messages)->str 函数。
    渠道未配置（密钥为空）返回 None。"""
    try:
        from videotrans.translator import LLM_CONCERT_DICT
        from videotrans.translator._runner import get_model_transobj
        ai_type = int(settings.get('llm_ai_type', 1) or 1)
        info = LLM_CONCERT_DICT[ai_type]
        api_key = settings.get(info['api_key_name'])
        if not api_key:
            return None
        ob = get_model_transobj(translate_type=info['id'], uuid=uuid)
        if not hasattr(ob, '_create_completion'):
            return None
    except Exception as e:
        logger.warning(f'构建LLM断句客户端失败: {e}')
        return None

    def chat(messages: List[Dict[str, str]]) -> str:
        kwargs = {
            "model": ob.model_name,
            "temperature": 0.1,
            "timeout": 300,
            "messages": messages,
        }
        if getattr(ob, 'reasoning_effort', None):
            kwargs["reasoning_effort"] = ob.reasoning_effort
        if "api.openai.com" in (ob.api_url or "") or (
                getattr(ob, 'ainame', '') == 'chatgpt'
                and re.match(r'^(gpt|o\d)', ob.model_name or '', flags=re.I)):
            kwargs["max_completion_tokens"] = int(ob.max_tokens)
        else:
            kwargs["max_tokens"] = int(ob.max_tokens)
        resp = ob._create_completion(kwargs)
        return resp.choices[0].message.content or ""

    return chat


def _load_prompt() -> str:
    from videotrans.configure._paths import ROOT_DIR
    from pathlib import Path
    p = Path(f'{ROOT_DIR}/videotrans/prompts/resegment/sentence.txt')
    return p.read_text(encoding='utf-8')


def _strip_think(text: str) -> str:
    return re.sub(r'<think>(.*?)</think>', '', text, flags=re.I | re.S)


def _llm_sentences(txt: str, chat_fn: Callable, max_cjk: int, max_en: int,
                   is_cjk: bool) -> List[str]:
    """LLM 语义断句 + agent loop 验证反馈，返回句子列表"""
    system_prompt = _load_prompt().replace(
        '{max_word_count_cjk}', str(max_cjk)).replace(
        '{max_word_count_english}', str(max_en))
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Please use multiple <br> tags to separate the following sentence:\n{txt}"},
    ]
    last_result = None
    for _step in range(LLM_MAX_STEPS):
        out = _strip_think(chat_fn(messages)).strip()
        cleaned = re.sub(r'\n+', '', out)
        parts = [p.strip() for p in cleaned.split('<br>') if p.strip()]
        last_result = parts
        ok, err = _validate_split(txt, parts, max_cjk, max_en, is_cjk)
        if ok:
            return parts
        logger.warning(f'LLM断句校验未通过，反馈重试: {err[:300]}')
        messages.append({"role": "assistant", "content": out})
        messages.append({"role": "user", "content": (
            f"Error: {err}\nFix the errors above and output the COMPLETE corrected "
            "text with <br> tags (include ALL segments, not just the fixed ones), no explanation.")})
    return last_result or [txt]


def _validate_split(original: str, parts: List[str], max_cjk: int, max_en: int,
                    is_cjk: bool):
    """校验 LLM 断句结果：内容一致性与单行长度。返回 (ok, error_msg)"""
    if not parts:
        return False, "No segments found. Split the text with <br> tags."
    limit = max_cjk if is_cjk else max_en
    # 1. 内容一致性（difflib，允许 4% 差异容忍标点/空格）
    merged = ("" if is_cjk else " ").join(parts)
    a = re.sub(r'\s+', ' ', original).strip()
    b = re.sub(r'\s+', ' ', merged).strip()
    ratio = difflib.SequenceMatcher(None, a, b).ratio()
    if ratio < LLM_SIMILARITY_THRESHOLD:
        return False, f"Content modified (similarity: {ratio:.1%}). Keep original text unchanged, only insert <br> between words."
    # 2. 单行长度
    violations = []
    for i, seg in enumerate(parts, 1):
        n = _count_words(seg, is_cjk)
        if n > limit:
            violations.append(f"Segment {i} '{seg[:40]}...': {n} > {limit} limit")
    if violations:
        return False, "Length violations:\n" + "\n".join(violations) + \
            "\n\nSplit these long segments further with <br>, then output the COMPLETE text with ALL segments."
    return True, ""


def _block_words(all_words: List[Dict[str, Any]], use_space: bool,
                 threshold: int = SEGMENT_WORD_THRESHOLD) -> List[List[int]]:
    """按字数阈值分块，块边界落在附近最大时间间隔处（VC _split_asr_data 算法）"""
    total_words = len(all_words)
    if total_words <= threshold:
        return [list(range(total_words))]
    full_text = _join_words(all_words, use_space)
    is_cjk = _is_cjk_text(full_text)
    total_count = _count_words(full_text, is_cjk)
    num_blocks = max(1, -(-total_count // threshold))
    words_per_block = total_count // num_blocks

    # 累计字数定位初始分割点
    split_points = []
    acc = 0
    for i, w in enumerate(all_words):
        acc += 1 if is_cjk else len(w['word'].strip().split() or [''])
        if acc >= words_per_block and len(split_points) < num_blocks - 1:
            split_points.append(i)
            acc = 0

    adjusted = []
    for sp in split_points:
        start = max(1, sp - SPLIT_SEARCH_RANGE)
        end = min(total_words - 1, sp + SPLIT_SEARCH_RANGE)
        best, best_gap = sp, -1
        for j in range(start, end):
            gap = all_words[j + 1]['start_ms'] - all_words[j]['end_ms']
            if gap > best_gap:
                best_gap, best = gap, j
        adjusted.append(best)
    adjusted = sorted(set(adjusted))

    blocks, prev = [], 0
    for idx in adjusted:
        blocks.append(list(range(prev, idx + 1)))
        prev = idx + 1
    if prev < total_words:
        blocks.append(list(range(prev, total_words)))
    return [b for b in blocks if b]


def _match_back(sentences: List[str], block_words: List[Dict[str, Any]],
                use_space: bool) -> List[List[int]]:
    """把 LLM 句子用滑动窗口 difflib 匹配回字级时间戳（VC 算法）"""
    asr_texts = [w['word'].strip() for w in block_words]
    asr_len = len(asr_texts)
    asr_index = 0
    max_shift = MATCH_MAX_SHIFT
    unmatched = 0
    new_groups = []
    sep = " " if use_space else ""

    for sentence in sentences:
        sent_norm = _normalize(sentence)
        wc = _count_words(sent_norm, not use_space)
        best_ratio, best_pos, best_size = 0.0, None, 0
        max_win = max(1, min(wc * 2 + 2, asr_len - asr_index))
        min_win = max(1, wc // 2)
        window_sizes = sorted(range(min_win, max_win + 1), key=lambda x: abs(x - wc))
        for win in window_sizes:
            max_start = min(asr_index + max_shift + 1, asr_len - win + 1)
            for start in range(max(asr_index, 0), max_start):
                substr = _normalize(sep.join(asr_texts[start:start + win]))
                ratio = difflib.SequenceMatcher(None, sent_norm, substr).ratio()
                if ratio > best_ratio:
                    best_ratio, best_pos, best_size = ratio, start, win
                if ratio == 1.0:
                    break
            if best_ratio == 1.0:
                break

        if best_ratio >= MATCH_SIMILARITY_THRESHOLD and best_pos is not None:
            end_idx = best_pos + best_size - 1
            span = list(range(best_pos, end_idx + 1))
            # 跨度过大时按时间间隔再分组，避免一句横跨长停顿
            span_words = [block_words[i] for i in span]
            for g in _group_by_time_gaps(span_words, MAX_GAP):
                new_groups.append([span[k] for k in g])
            max_shift = MATCH_MAX_SHIFT
            asr_index = end_idx + 1
        else:
            unmatched += 1
            if unmatched > MATCH_MAX_UNMATCHED:
                raise ValueError(f'unmatched sentences exceeded {MATCH_MAX_UNMATCHED}')
            max_shift = MATCH_LARGE_SHIFT
            asr_index = min(asr_index + 1, asr_len - 1)

    # 收尾：剩余未匹配的词并入结果
    if asr_index < asr_len:
        new_groups.append(list(range(asr_index, asr_len)))
    return new_groups


def llm_split(all_words: List[Dict[str, Any]], use_space: bool, chat_fn: Callable,
              max_cjk: int, max_en: int) -> List[List[int]]:
    """LLM 语义断句：分块并发调 LLM + 回填时间戳，返回索引分组列表（相对 all_words）"""
    blocks = _block_words(all_words, use_space)
    full_is_cjk = _is_cjk_text(_join_words(all_words, use_space))

    def _process(block: List[int]) -> List[List[int]]:
        block_words = [all_words[i] for i in block]
        try:
            txt = _join_words(block_words, use_space)
            sentences = _llm_sentences(txt, chat_fn, max_cjk, max_en, full_is_cjk)
            sub_groups = _match_back(sentences, block_words, use_space)
        except Exception as e:
            # 块级降级：该块改用本地规则断句，不影响其他块
            logger.warning(f'该块LLM断句失败，此块降级为本地规则断句: {e}')
            return rule_split(block_words, use_space)
        return [[block[k] for k in g] for g in sub_groups]

    results: List[List[List[int]]] = [None] * len(blocks)
    with ThreadPoolExecutor(max_workers=LLM_CONCURRENT) as pool:
        futures = {pool.submit(_process, b): i for i, b in enumerate(blocks)}
        for fut in as_completed(futures):
            results[futures[fut]] = fut.result()
    groups = []
    for r in results:
        if r:
            groups.extend(r)
    groups.sort(key=lambda g: all_words[g[0]]['start_ms'])
    return groups


# ==================== 调度入口 ====================

def _groups_to_srt(groups: List[List[int]], all_words: List[Dict[str, Any]],
                   use_space: bool) -> List[Any]:
    """把索引分组转成 SrtItem 列表"""
    from videotrans.process._stt_utils import format_srt_time
    from videotrans.task.taskcfg import SrtItem
    srt_output = []
    for chunk in groups:
        chunk_words = [all_words[i] for i in chunk]
        text = _join_words(chunk_words, use_space)
        if not text.strip():
            continue
        start_ms = int(chunk_words[0]['start_ms'])
        end_ms = int(chunk_words[-1]['end_ms'])
        srt_output.append(SrtItem(**{
            "line": len(srt_output) + 1,
            "text": text,
            "start_time": start_ms,
            "end_time": end_ms,
            "startraw": format_srt_time(start_ms),
            "endraw": format_srt_time(end_ms),
            "time": f"{format_srt_time(start_ms)} --> {format_srt_time(end_ms)}",
        }))
    return srt_output


def smart_resegment(texts: List[Dict[str, Any]], language: str,
                    logs_file=None, uuid: Optional[str] = None) -> Optional[List[Any]]:
    """智能断句调度入口。返回 SrtItem 列表；返回 None 表示回退原 _resegment 逻辑。
    - resegment_mode == 'llm' 且已配置 LLM 渠道：LLM 语义断句，失败自动降级规则层
    - 其他情况：本地规则层算法
    """
    from videotrans.process._stt_utils import _write_log
    import json as _json
    try:
        use_space = language.lower() not in NO_SPACE_LANGS
        all_words = _flatten_words(texts)
        if not all_words:
            return None

        mode = str(settings.get('resegment_mode', 'local') or 'local').lower()
        chat_fn = None
        if mode == 'llm':
            chat_fn = _build_chat_fn(uuid=uuid)
            if chat_fn is None:
                logger.warning('resegment_mode=llm 但高级设置中的LLM渠道未配置密钥，自动降级为本地规则断句')

        max_cjk = int(float(settings.get('llm_split_cjk_len', LLM_CJK_LEN_DEFAULT) or LLM_CJK_LEN_DEFAULT))
        max_en = int(float(settings.get('llm_split_en_len', LLM_EN_LEN_DEFAULT) or LLM_EN_LEN_DEFAULT))

        groups = None
        if chat_fn is not None:
            try:
                _write_log(logs_file, _json.dumps({"type": "logs", "text": 'SmartResegment:llm start'}))
                groups = llm_split(all_words, use_space, chat_fn, max_cjk, max_en)
                _write_log(logs_file, _json.dumps({"type": "logs", "text": f'SmartResegment:llm ok {len(groups)} lines'}))
                logger.debug(f'LLM智能断句完成，共 {len(groups)} 行')
            except Exception as e:
                logger.warning(f'LLM断句失败，自动降级为本地规则断句: {e}')
                groups = None
        if groups is None:
            _write_log(logs_file, _json.dumps({"type": "logs", "text": 'SmartResegment:rule start'}))
            groups = rule_split(all_words, use_space)
            _write_log(logs_file, _json.dumps({"type": "logs", "text": f'SmartResegment:rule ok {len(groups)} lines'}))
        return _groups_to_srt(groups, all_words, use_space)
    except Exception as e:
        logger.warning(f'智能断句异常，回退原有断句逻辑: {e}')
        return None
