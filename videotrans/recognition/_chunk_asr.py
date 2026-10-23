# 共享工具：长音频分块 + 云端ASR结果合并（移植自 VideoCaptioner ChunkedASR 思路）
import os
from difflib import SequenceMatcher
from pathlib import Path

# 分块时长(秒)与重叠时长(毫秒)
CHUNK_SEC = 600
OVERLAP_MS = 10000


def split_wav_chunks(audio_file: str, cache_folder: str, chunk_sec: int = CHUNK_SEC,
                     overlap_sec: int = 10) -> list:
    """将 16k wav 按固定时长切块，块间带 overlap，返回 [{file, offset}, ...]"""
    from pydub import AudioSegment
    tmp = Path(cache_folder) / "asr_chunks"
    tmp.mkdir(parents=True, exist_ok=True)
    audio = AudioSegment.from_wav(audio_file)
    total = len(audio)
    chunks = []
    start = 0
    idx = 0
    while start < total:
        end = min(start + chunk_sec * 1000, total)
        p = tmp / f"chunk_{idx:03d}.wav"
        audio[start:end].export(p.as_posix(), format="wav")
        chunks.append({"file": p.as_posix(), "offset": start})
        if end >= total:
            break
        start += (chunk_sec - overlap_sec) * 1000
        idx += 1
    return chunks


def _norm(t: str) -> str:
    return "".join(t.split())


def merge_results(chunk_segs: list, chunks: list, overlap_ms: int = OVERLAP_MS) -> list:
    """合并各分块识别结果 [(text, abs_start_ms, abs_end_ms), ...]，重叠区去重

    chunk_segs: 与 chunks 对齐的列表，每项为 [(text, 相对start_ms, 相对end_ms), ...]
    """
    merged = []
    for ci, segs in enumerate(chunk_segs):
        if not segs:
            continue
        off = chunks[ci]["offset"]
        tail = [s for s in merged if s[1] >= off - overlap_ms]
        for text, rs, re_ in segs:
            abs_s, abs_e = rs + off, re_ + off
            if ci > 0 and rs < overlap_ms:
                dup = False
                for t, s, e in tail:
                    if _norm(t) and SequenceMatcher(None, _norm(text), _norm(t)).ratio() >= 0.8:
                        dup = True
                        break
                if dup:
                    continue
            merged.append((text, abs_s, abs_e))
    merged.sort(key=lambda x: x[1])
    return merged
