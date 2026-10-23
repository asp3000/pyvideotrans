# Bilibili 必剪(Bcut) 免费云端语音识别渠道
# 移植自 VideoCaptioner core/asr/bcut.py (B接口)
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Union

import requests
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_not_exception_type, before_log, after_log

from videotrans.configure.config import tr, settings, logger
from videotrans.configure.excepts import NO_RETRY_EXCEPT, SpeechToTextError
from videotrans.recognition._base import BaseRecogn
from videotrans.task.taskcfg import SrtItem
from videotrans.util.help_srt import ms_to_time_string

API_BASE_URL = "https://member.bilibili.com/x/bcut/rubick-interface"
API_REQ_UPLOAD = API_BASE_URL + "/resource/create"
API_COMMIT_UPLOAD = API_BASE_URL + "/resource/create/complete"
API_CREATE_TASK = API_BASE_URL + "/task"
API_QUERY_RESULT = API_BASE_URL + "/task/result"

HEADERS = {
    "User-Agent": "Bilibili/1.0.0 (https://www.bilibili.com)",
    "Content-Type": "application/json",
}


@dataclass(repr=False)
class BcutRecogn(BaseRecogn):

    def _upload(self, data: bytes) -> str:
        """分片上传音频并获取 download_url"""
        payload = json.dumps({
            "type": 2,
            "name": "audio.mp3",
            "size": len(data),
            "ResourceFileType": "mp3",
            "model_id": "8",
        })
        resp = requests.post(API_REQ_UPLOAD, data=payload, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        d = resp.json()["data"]

        etags = []
        per = d["per_size"]
        for clip, url in enumerate(d["upload_urls"]):
            r = requests.put(url, data=data[clip * per:(clip + 1) * per], headers=HEADERS, timeout=180)
            r.raise_for_status()
            etag = r.headers.get("Etag")
            if etag:
                etags.append(etag)

        resp2 = requests.post(API_COMMIT_UPLOAD, headers=HEADERS, timeout=30, data=json.dumps({
            "InBossKey": d["in_boss_key"],
            "ResourceId": d["resource_id"],
            "Etags": ",".join(etags),
            "UploadId": d["upload_id"],
            "model_id": "8",
        }))
        resp2.raise_for_status()
        return resp2.json()["data"]["download_url"]

    def _transcribe_chunk(self, wav_file: str) -> List[tuple]:
        """转写单个音频块，返回 [(text, rel_start_ms, rel_end_ms), ...]"""
        with open(wav_file, 'rb') as f:
            data = f.read()

        download_url = self._upload(data)

        resp = requests.post(API_CREATE_TASK, json={"resource": download_url, "model_id": "8"},
                             headers=HEADERS, timeout=30)
        resp.raise_for_status()
        task_id = resp.json()["data"]["task_id"]

        d = None
        for _ in range(300):
            if self._exit():
                return []
            rr = requests.get(API_QUERY_RESULT, params={"model_id": 7, "task_id": task_id},
                              headers=HEADERS, timeout=30)
            rr.raise_for_status()
            d = rr.json()["data"]
            if d.get("state") == 4:
                break
            time.sleep(1)
        if d is None or d.get("state") != 4:
            raise RuntimeError(f"Bcut ASR task timeout, state={d and d.get('state')}")

        result = json.loads(d["result"])
        segs = []
        for u in result.get("utterances", []):
            txt = (u.get("transcript") or "").strip()
            if txt and u.get("end_time", 0) > u.get("start_time", 0):
                segs.append((txt, u["start_time"], u["end_time"]))
        return segs

    @retry(retry=retry_if_not_exception_type(NO_RETRY_EXCEPT),
           stop=(stop_after_attempt(settings.get('retry_nums'))), wait=wait_fixed(2),
           before=before_log(logger, logging.INFO), after=after_log(logger, logging.INFO))
    def _exec(self) -> Union[List[SrtItem], None]:
        if self._exit():
            return
        from videotrans.recognition._chunk_asr import split_wav_chunks, merge_results

        self.signal(text="Bilibili Bcut(必剪) free ASR...")
        chunks = split_wav_chunks(self.audio_file, self.cache_folder)
        logger.debug(f'Bcut ASR: {len(chunks)} chunks')
        results = [None] * len(chunks)
        errs = []

        def _work(i, ch):
            results[i] = self._transcribe_chunk(ch["file"])

        with ThreadPoolExecutor(max_workers=min(3, len(chunks))) as ex:
            futs = {ex.submit(_work, i, ch): i for i, ch in enumerate(chunks)}
            for f in as_completed(futs):
                i = futs[f]
                try:
                    f.result()
                    self.signal(text=f"Bcut ASR chunk {i + 1}/{len(chunks)} ok")
                except Exception as e:
                    errs.append(f"chunk{i + 1}: {e}")
                    logger.error(f'Bcut chunk {i + 1} failed: {e}')

        # 过滤失败的分块
        pairs = [(results[i], chunks[i]) for i in range(len(chunks)) if results[i]]
        if not pairs:
            raise SpeechToTextError(f"Bcut(必剪) ASR failed: {'; '.join(errs) or 'no result'}")
        if errs:
            logger.warning(f'Bcut partial chunks failed: {errs}')

        segs = merge_results([p[0] for p in pairs], [p[1] for p in pairs])

        raws: List[SrtItem] = []
        for text, s, e in segs:
            if e <= s:
                continue
            startraw = ms_to_time_string(ms=s)
            endraw = ms_to_time_string(ms=e)
            item = SrtItem(
                line=len(raws) + 1,
                time=f"{startraw} --> {endraw}",
                text=text,
                start_time=s,
                end_time=e,
                startraw=startraw,
                endraw=endraw,
            )
            raws.append(item)
            self.signal(text=f"{tr('yuyinshibiejindu')} {item['line']}/{len(segs)}")
            self.signal(text=f"{text}\n", type='subtitle')
        return raws
