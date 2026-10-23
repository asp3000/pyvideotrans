import time
import traceback
import numpy as np
import scipy.io.wavfile as Wavfile
from videotrans.configure.config import logger


def _split_oversize_segments(segments, max_ms):
    """将超长片段按 max_ms 均匀硬切，防止识别引擎(全注意力encoder)内存爆炸。
    硬切点位于连续语音中间，字幕会在切点断开，但时间戳精度不受影响。"""
    if not max_ms or max_ms <= 0:
        return segments
    result = []
    for s, e in segments:
        duration = e - s
        if duration <= max_ms * 1.5:
            result.append([s, e])
            continue
        nums = int(np.ceil(duration / max_ms))
        step = duration / nums
        logger.warning(
            f"[VAD] 超长片段 {duration / 1000:.1f}s 强制切分为 {nums} 段: [{s}ms - {e}ms]")
        for j in range(nums):
            _s = int(s + j * step)
            _e = int(s + (j + 1) * step) if j < nums - 1 else e
            result.append([_s, _e])
    return result




def get_speech_timestamp_silero(input_wav,
                                threshold=0.45,
                                min_speech_duration_ms=3000,
                                max_speech_duration_ms=5000,
                                min_silent_duration_ms=600,
                                speech_pad_ms=0,
                                max_merge_gap_ms=800,  #两次说话间隔<800ms且总长不超限时，自动粘合
                                **kw):
    vad_p = {
        "threshold": threshold,
        "min_speech_duration_ms": 100,# 超过该值直接丢弃，不可过大，否则会吞字
        "max_speech_duration_s": float(max_speech_duration_ms / 1000.0),
        "min_silence_duration_ms": int(max(min_silent_duration_ms, 140)),#静音分割区间
        "speech_pad_ms": speech_pad_ms  # 仅 faster-whisper时在此处进行边缘补白，因无需cut_audio
    }
    logger.debug(f'[silero-VAD]:最终断句参数：{vad_p=}')
    _rc=max(int(max_speech_duration_ms*0.2),1500)

    sampling_rate = 16000
    min_isolated_duration_ms=140  # 前后孤立，并且片段时长小于此，则丢弃

    from faster_whisper.audio import decode_audio
    from faster_whisper.vad import (
        VadOptions,
        get_speech_timestamps
    )


    audio_data = decode_audio(input_wav, sampling_rate=sampling_rate)
    total_audio_duration_ms = int(len(audio_data) / sampling_rate * 1000)

    # 低音量自动增益: 远场/低音量录音(如RMS<-30dB)会导致VAD完全检测不到语音
    # 按RMS归一化(峰值限幅),只影响检测阶段，不改变时间戳
    _rms = float(np.sqrt(np.mean(audio_data ** 2))) if len(audio_data) else 0.0
    if 0 < _rms < 0.03:
        _gain = min(0.06 / _rms, 20.0)
        audio_data = np.clip(audio_data * _gain, -1.0, 1.0)
        logger.warning(f'[silero-VAD] 低音量音频自动增益 x{_gain:.1f} (RMS {_rms:.4f}->{min(_rms * _gain, 0.06):.4f})')

    raw_chunks = get_speech_timestamps(
        audio_data, vad_options=VadOptions(**vad_p)
    )
    if not raw_chunks:
        # 降阈值重试
        for _th in (threshold * 0.7, 0.2):
            _th = max(0.1, min(threshold, _th))
            if _th >= threshold:
                continue
            vad_p['threshold'] = _th
            logger.warning(f'[silero-VAD] 未检测到语音, 降低阈值重试 threshold={_th}')
            raw_chunks = get_speech_timestamps(
                audio_data, vad_options=VadOptions(**vad_p)
            )
            if raw_chunks:
                break
    if not raw_chunks:
        # 仍无语音但音频并非数字静音 → 固定窗口切分兜底,交由识别引擎自行判断
        _rms = float(np.sqrt(np.mean(audio_data ** 2))) if len(audio_data) else 0.0
        if _rms > 1e-4 and total_audio_duration_ms <= 60 * 60 * 1000:
            logger.warning(
                f'[silero-VAD] 音频非静音(RMS={_rms:.5f})但未检测到语音, 按25s固定窗口切分兜底')
            _win = 25000
            return [[s, min(s + _win, total_audio_duration_ms)]
                    for s in range(0, total_audio_duration_ms, _win)
                    if min(s + _win, total_audio_duration_ms) > s]
        return []

    raw_segments = [
        [
            int(round(chunk["start"] / sampling_rate * 1000)),
            int(round(chunk["end"] / sampling_rate * 1000)),
        ]
        for chunk in raw_chunks
    ]

    # 贪心粘合 (吸收被噪音打碎的弱音片段)
    merged_segments = []

    for cur_start, cur_end in raw_segments:
        if not merged_segments:
            merged_segments.append([cur_start, cur_end])
            continue

        prev_start, prev_end = merged_segments[-1]
        gap = cur_start - prev_end
        combined_duration = cur_end - prev_start

        # 只要与前一片段距离近，且合并后不超长，全部粘合在一起
        if gap <= max_merge_gap_ms and combined_duration <= (max_speech_duration_ms+_rc):
            merged_segments[-1][1] = cur_end  # 扩大上一个片段的右边界
        else:
            merged_segments.append([cur_start, cur_end])

    # 仅剔除完全孤立的极短爆音
    final_segments = []
    for s, e in merged_segments:
        duration = e - s

        # 如果一个片段合并后依然极其短（如 <140ms），且前后间隔都很远，说明是孤立的杂音/爆音
        if duration < min_isolated_duration_ms:
            logger.warning(f"丢弃前后孤立的极短杂音片段: [{s}ms - {e}ms] ({duration}ms)")
            continue

        # 边界越界保护
        s_clamped = max(0, s)
        e_clamped = min(total_audio_duration_ms, e)
        if e_clamped > s_clamped:
            final_segments.append([s_clamped, e_clamped])

    # 再次合并过短的
    _thrid_segs=[]
    for i,it in enumerate(final_segments):
        _duration=it[1]-it[0]
        if not _thrid_segs or _duration>=min_speech_duration_ms:
            _thrid_segs.append(it)
            continue
        _last_duration=_thrid_segs[-1][1]-_thrid_segs[-1][0]

        if _last_duration>=max_speech_duration_ms and  i< len(final_segments) - 1:
            _thrid_segs.append(it)
            continue

        _thrid_segs[-1][1]=it[1]

    logger.debug(
        f"[silero-VAD]: 原始片段数 {len(raw_segments)} -> {len(final_segments)} -> {len(_thrid_segs)} 句子"
    )

    # 安全网: 超长段(VAD在连续背景音乐下可能产出分钟级片段)硬切,防止识别引擎内存爆炸
    _thrid_segs = _split_oversize_segments(_thrid_segs, max_speech_duration_ms)

    # VAD失效兜底: 远场/混响人声可能只检出零星几段(占比过低),固定窗口切分交由识别引擎判断
    _cov = sum(e - s for s, e in _thrid_segs) / max(total_audio_duration_ms, 1)
    if len(_thrid_segs) < 5 and _cov < 0.05:
        _rms_chk = float(np.sqrt(np.mean(audio_data ** 2))) if len(audio_data) else 0.0
        if _rms_chk > 1e-4 and total_audio_duration_ms <= 60 * 60 * 1000:
            logger.warning(
                f'[silero-VAD] 语音占比过低({_cov * 100:.1f}%,{len(_thrid_segs)}段), VAD可能对该音频失效, 按25s固定窗口切分兜底')
            _win = 25000
            return [[s, min(s + _win, total_audio_duration_ms)]
                    for s in range(0, total_audio_duration_ms, _win)
                    if min(s + _win, total_audio_duration_ms) > s]

    return _thrid_segs






def get_speech_timestamp(
    input_wav=None,
    threshold=0.45,
    max_speech_duration_ms=5000,  # 目标最大片段长度 (建议8~12s)
    min_speech_duration_ms=3000,  # 目标最大片段长度 (建议8~12s)
    min_silent_duration_ms=600,  # VAD停顿判定阈值 (300ms)
    speech_pad_ms=0,  # 不在此处补白，避免时间戳错乱
    max_merge_gap_ms=800,  # 核心：停顿<=800ms一律视为同一句，直接合并
    min_isolated_duration_ms=140,  # 剔除孤立无援的超短噪点(<150ms)
    **kw,
):

    try:
        sr, data = Wavfile.read(input_wav)
    except Exception as e:
        logger.exception(f"读取音频失败: {e}", exc_info=True)
        return None

    logger.debug(f'[ten-vad]最终参数:{threshold=},{max_speech_duration_ms=},{min_speech_duration_ms},{min_silent_duration_ms=},{speech_pad_ms=},{max_merge_gap_ms=},{min_isolated_duration_ms=}')
    _rc=max(int(max_speech_duration_ms*0.2),1500)
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    if np.issubdtype(data.dtype, np.floating):
        data = (np.clip(data, -1.0, 1.0) * 32767).astype(np.int16)
    elif data.dtype != np.int16:
        data = data.astype(np.int16)

    # 低音量自动增益: 与 silero 路径一致,峰值归一化到 int16 满幅的 90%
    _peak = int(np.abs(data).max()) if len(data) else 0
    if 0 < _peak < 16000:
        _gain = 29490 / _peak
        data = np.clip(data.astype(np.int64) * _gain, -32768, 32767).astype(np.int16)
        logger.warning(f'[Ten-VAD] 低音量音频自动增益 x{_gain:.1f} (峰值{_peak})')

    total_audio_duration_ms = int(len(data) / sr * 1000)

    # 2. 帧参数计算
    hop_size = 256
    frame_duration_ms = (hop_size / sr) * 1000.0

    min_sil_frames = max(1, int(min_silent_duration_ms / frame_duration_ms))
    max_speech_frames = (
        int(max_speech_duration_ms / frame_duration_ms)
        if max_speech_duration_ms
        else None
    )

    # 3. 稳健的能量自适应阈值调整 (基于 int16 幅值)
    audio_energy = np.mean(np.abs(data)) if len(data) > 0 else 0
    adjusted_threshold = threshold
    if audio_energy > 12000:
        adjusted_threshold = min(0.85, threshold * 1.15)
    elif audio_energy < 800:
        adjusted_threshold = max(0.2, threshold * 0.8)

    # 执行底层 Ten-VAD 检测
    raw_frame_segments = _detect_raw_segments(
        data,
        adjusted_threshold,
        min_sil_frames,
        max_speech_frames=max_speech_frames,
    )

    if not raw_frame_segments:
        # 仍无语音但音频并非数字静音 → 固定窗口切分兜底,交由识别引擎自行判断
        _rms = float(np.sqrt(np.mean(data.astype(np.float64) ** 2))) if len(data) else 0.0
        if _rms > 3.0 and total_audio_duration_ms <= 60 * 60 * 1000:
            logger.warning(
                f'[Ten-VAD] 音频非静音(RMS={_rms:.1f})但未检测到语音, 按25s固定窗口切分兜底')
            _win = 25000
            return [[s, min(s + _win, total_audio_duration_ms)]
                    for s in range(0, total_audio_duration_ms, _win)
                    if min(s + _win, total_audio_duration_ms) > s]
        return None

    # 5. 转换为毫秒，并加入 speech_pad_ms 边缘补白
    raw_ms_segments = []
    for s_frame, e_frame in raw_frame_segments:
        s_ms = max(0, int(s_frame * frame_duration_ms - speech_pad_ms))
        e_ms = min(
            total_audio_duration_ms,
            int(e_frame * frame_duration_ms + speech_pad_ms),
        )
        if e_ms > s_ms:
            raw_ms_segments.append([s_ms, e_ms])

    if not raw_ms_segments:
        return None

    # 贪心粘合 (吸收被噪音打碎的弱音片段)
    merged_segments = []
    for cur_start, cur_end in raw_ms_segments:
        if not merged_segments:
            merged_segments.append([cur_start, cur_end])
            continue

        prev_start, prev_end = merged_segments[-1]
        gap = cur_start - prev_end
        combined_duration = cur_end - prev_start

        # 满足以下任一条件即合并：
        # (1) 前后有重叠 (gap <= 0)
        # (2) 间隔停顿小于设定阈值 (gap <= max_merge_gap_ms) 且 合并后总长不超标
        if (gap <= max_merge_gap_ms) and (
            combined_duration <= (max_speech_duration_ms+_rc)
        ):
            # 扩展上一个片段的右边界
            merged_segments[-1][1] = max(prev_end, cur_end)
        else:
            merged_segments.append([cur_start, cur_end])

    # 剔除孤立噪点与边界安全规整
    final_segments = []
    for s, e in merged_segments:
        duration = e - s
        # 只有在完全孤立且时长 < min_isolated_duration_ms 时才丢弃
        if duration < min_isolated_duration_ms:
            logger.debug(
                f"[Ten-VAD] 丢弃孤立短噪点: [{s}ms - {e}ms], 时长: {duration}ms"
            )
            continue
        final_segments.append([s, e])

    # 再次合并过短的
    _thrid_segs=[]
    for i,it in enumerate(final_segments):
        _duration=it[1]-it[0]
        if not _thrid_segs or _duration>=min_speech_duration_ms:
            _thrid_segs.append(it)
            continue

        _last_duration=_thrid_segs[-1][1]-_thrid_segs[-1][0]

        if _last_duration>=max_speech_duration_ms and  i< len(final_segments) - 1:
            _thrid_segs.append(it)
            continue

        _thrid_segs[-1][1]=it[1]


    logger.debug(
        f"[Ten-VAD] {len(merged_segments)} -> {len(final_segments)} -> {len(_thrid_segs)} 优化"
    )

    # 安全网: 超长段硬切,防止识别引擎内存爆炸
    _thrid_segs = _split_oversize_segments(_thrid_segs, max_speech_duration_ms)

    # VAD失效兜底: 与 silero 路径一致
    _cov = sum(e - s for s, e in _thrid_segs) / max(total_audio_duration_ms, 1)
    if len(_thrid_segs) < 5 and _cov < 0.05:
        _rms_chk = float(np.sqrt(np.mean(data.astype(np.float64) ** 2))) if len(data) else 0.0
        if _rms_chk > 3.0 and total_audio_duration_ms <= 60 * 60 * 1000:
            logger.warning(
                f'[Ten-VAD] 语音占比过低({_cov * 100:.1f}%,{len(_thrid_segs)}段), 按25s固定窗口切分兜底')
            _win = 25000
            return [[s, min(s + _win, total_audio_duration_ms)]
                    for s in range(0, total_audio_duration_ms, _win)
                    if min(s + _win, total_audio_duration_ms) > s]

    return _thrid_segs


def _detect_raw_segments(
    data, threshold, min_silent_frames, max_speech_frames=None
):
    """底层 TenVad 逐帧扫描检测"""
    # 惰性导入：ten_vad 缺失时不影响 silero-VAD 路径
    from ten_vad import TenVad
    hop_size = 256
    # 实例化底层 TenVad
    ten_vad_instance = TenVad(hop_size, threshold)

    num_frames = (data.shape[0] - hop_size) // hop_size + 1
    segments = []
    triggered = False
    speech_start_frame = 0
    silence_frame_count = 0

    for i in range(num_frames):
        audio_frame = data[i * hop_size : (i + 1) * hop_size]
        if len(audio_frame) != hop_size:
            continue

        _, is_speech = ten_vad_instance.process(audio_frame)

        if triggered:
            if is_speech == 1:
                silence_frame_count = 0
            else:
                silence_frame_count += 1

            is_silence_timeout = silence_frame_count >= min_silent_frames
            is_max_timeout = (
                max_speech_frames is not None
                and (i - speech_start_frame) >= max_speech_frames
            )

            if is_silence_timeout or is_max_timeout:
                end_frame = i if is_max_timeout else (i - silence_frame_count)
                if end_frame > speech_start_frame:
                    segments.append([speech_start_frame, end_frame])
                triggered = False
                silence_frame_count = 0
        else:
            if is_speech == 1:
                triggered = True
                speech_start_frame = i
                silence_frame_count = 0

    # 音频结束时如果仍处于说话状态
    if triggered:
        end_frame = num_frames - silence_frame_count
        if end_frame > speech_start_frame:
            segments.append([speech_start_frame, end_frame])

    return segments

