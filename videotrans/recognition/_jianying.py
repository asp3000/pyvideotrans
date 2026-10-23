# 剪映(CapCut 国内版) 免费云端语音识别渠道
# 移植自 VideoCaptioner core/asr/jianying.py (J接口)
# 注意: 请求签名依赖第三方签名服务 https://asrtools-update.bkfeng.top/sign
import datetime
import hashlib
import hmac
import json
import logging
import time
import uuid
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

import requests
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_not_exception_type, before_log, after_log

from videotrans.configure.config import tr, settings, logger
from videotrans.configure.excepts import NO_RETRY_EXCEPT, SpeechToTextError
from videotrans.recognition._base import BaseRecogn
from videotrans.task.taskcfg import SrtItem
from videotrans.util.help_srt import ms_to_time_string

SIGN_URL = "https://asrtools-update.bkfeng.top/sign"
LV_BASE = "https://lv-pc-api-sinfonlinec.ulikecam.com"
UA_CRONET = "Cronet/TTNetVersion:d4572e53 2024-06-12 QuicVersion:4bf243e0 2023-04-17"


def _sign(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()


def _signature_key(secret_key: str, date_stamp: str, region: str, service: str) -> bytes:
    k_date = _sign(("AWS4" + secret_key).encode("utf-8"), date_stamp)
    k_region = _sign(k_date, region)
    k_service = _sign(k_region, service)
    return _sign(k_service, "aws4_request")


def _aws_signature(secret_key: str, request_parameters: str, headers: Dict[str, str],
                   method: str = "GET", payload: str = "", region: str = "cn", service: str = "vod") -> str:
    canonical_uri = "/"
    canonical_querystring = request_parameters
    canonical_headers = "\n".join([f"{k}:{v}" for k, v in headers.items()]) + "\n"
    signed_headers = ";".join(headers.keys())
    payload_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    canonical_request = f"{method}\n{canonical_uri}\n{canonical_querystring}\n{canonical_headers}\n{signed_headers}\n{payload_hash}"
    amzdate = headers["x-amz-date"]
    datestamp = amzdate.split("T")[0]
    algorithm = "AWS4-HMAC-SHA256"
    credential_scope = f"{datestamp}/{region}/{service}/aws4_request"
    string_to_sign = f"{algorithm}\n{amzdate}\n{credential_scope}\n{hashlib.sha256(canonical_request.encode('utf-8')).hexdigest()}"
    signing_key = _signature_key(secret_key, datestamp, region, service)
    return hmac.new(signing_key, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()


class _JianyingAPI:
    """剪映 audio_subtitle 接口封装"""

    def __init__(self):
        self.tdid = self._get_tid()

    @staticmethod
    def _get_tid() -> str:
        i = str(datetime.datetime.now().year)[3]
        fr = 390 + int(i)
        ed = "3278516897751" if int(i) % 2 != 0 else f"{uuid.getnode():013d}"
        return f"{fr}{ed}"

    def _sign(self, url_path: str) -> Tuple[str, str]:
        current_time = str(int(time.time()))
        resp = requests.post(SIGN_URL, json={
            "url": url_path, "current_time": current_time,
            "pf": "4", "appvr": "6.6.0", "tdid": self.tdid,
        }, headers={"User-Agent": "pyvideotrans", "tdid": self.tdid, "t": current_time}, timeout=20)
        resp.raise_for_status()
        sign = resp.json().get("sign")
        if not sign:
            raise RuntimeError("剪映签名服务(asrtools-update.bkfeng.top)未返回sign")
        return sign.lower(), current_time

    def _headers(self, url_path: str) -> Dict[str, str]:
        sign, t = self._sign(url_path)
        return {
            "User-Agent": UA_CRONET,
            "appvr": "6.6.0",
            "device-time": t,
            "pf": "4",
            "sign": sign,
            "sign-ver": "1",
            "tdid": self.tdid,
        }

    def transcribe(self, data: bytes) -> List[dict]:
        """上传音频并转写，返回 utterances 列表"""
        crc32_hex = format(zlib.crc32(data) & 0xFFFFFFFF, "08x")
        store_uri = self._upload(data, crc32_hex)
        query_id = self._submit(store_uri)
        return self._query(query_id)

    def _upload(self, data: bytes, crc32_hex: str) -> str:
        # 1) 获取上传凭证
        h = self._headers("/lv/v1/upload_sign")
        resp = requests.post(f"{LV_BASE}/lv/v1/upload_sign", data=json.dumps({"biz": "pc-recognition"}),
                             headers=h, timeout=30)
        resp.raise_for_status()
        login = resp.json()["data"]
        access_key = login["access_key_id"]
        secret_key = login["secret_access_key"]
        session_token = login["session_token"]

        # 2) 申请上传地址 (AWS V4 签名)
        request_parameters = (f"Action=ApplyUploadInner&FileSize={len(data)}&FileType=object"
                              f"&IsInner=1&SpaceName=lv-mac-recognition&Version=2020-11-19&s=5y0udbjapi")
        t = datetime.datetime.now(datetime.timezone.utc)
        amz_date = t.strftime("%Y%m%dT%H%M%SZ")
        datestamp = t.strftime("%Y%m%d")
        headers = {"x-amz-date": amz_date, "x-amz-security-token": session_token}
        signature = _aws_signature(secret_key, request_parameters, headers, region="cn", service="vod")
        headers["authorization"] = (
            f"AWS4-HMAC-SHA256 Credential={access_key}/{datestamp}/cn/vod/aws4_request, "
            f"SignedHeaders=x-amz-date;x-amz-security-token, Signature={signature}"
        )
        resp = requests.get(f"https://vod.bytedanceapi.com/?{request_parameters}", headers=headers, timeout=30)
        resp.raise_for_status()
        infos = resp.json()["Result"]["UploadAddress"]
        store_uri = infos["StoreInfos"][0]["StoreUri"]
        auth = infos["StoreInfos"][0]["Auth"]
        upload_id = infos["StoreInfos"][0]["UploadID"]
        session_key = infos["SessionKey"]
        upload_host = infos["UploadHosts"][0]
        _ = session_key

        upload_headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.138 Safari/537.36 Thea/1.0.1",
            "Authorization": auth,
            "Content-CRC32": crc32_hex,
        }

        # 3) 上传文件
        resp = requests.put(f"https://{upload_host}/{store_uri}?partNumber=1&uploadID={upload_id}",
                            data=data, headers=upload_headers, timeout=300)
        d = resp.json()
        if d.get("success") != 0:
            raise RuntimeError(f"剪映音频上传失败: {resp.text[:200]}")

        # 4) 校验
        requests.post(f"https://{upload_host}/{store_uri}?uploadID={upload_id}",
                      data=f"1:{crc32_hex}", headers=upload_headers, timeout=30)

        # 5) 提交
        requests.put(f"https://{upload_host}/{store_uri}?uploadID={upload_id}&partNumber=1"
                     f"&x-amz-security-token={session_token}",
                     data=data, headers=upload_headers, timeout=300)
        return store_uri

    def _submit(self, store_uri: str) -> str:
        payload = {
            "adjust_endtime": 200,
            "audio": store_uri,
            "caption_type": 2,
            "client_request_id": str(uuid.uuid4()),
            "max_lines": 1,
            "songs_info": [{"end_time": 6000, "id": "", "start_time": 0}],
            "words_per_line": 16,
        }
        h = self._headers("/lv/v1/audio_subtitle/submit")
        resp = requests.post(f"{LV_BASE}/lv/v1/audio_subtitle/submit", json=payload, headers=h, timeout=30)
        d = resp.json()
        if d.get("ret") != "0":
            raise RuntimeError(f"剪映ASR提交失败: {d.get('errmsg', 'Unknown')} (ret={d.get('ret')})")
        return d["data"]["id"]

    def _query(self, query_id: str) -> List[dict]:
        h = self._headers("/lv/v1/audio_subtitle/query")
        resp = requests.post(f"{LV_BASE}/lv/v1/audio_subtitle/query",
                             json={"id": query_id, "pack_options": {"need_attribute": True}},
                             headers=h, timeout=30)
        d = resp.json()
        if d.get("ret") != "0":
            raise RuntimeError(f"剪映ASR查询失败: {d.get('errmsg', 'Unknown')} (ret={d.get('ret')})")
        return d["data"].get("utterances", [])


@dataclass(repr=False)
class JianyingRecogn(BaseRecogn):

    def _transcribe_chunk(self, wav_file: str) -> List[tuple]:
        with open(wav_file, 'rb') as f:
            data = f.read()
        utterances = _JianyingAPI().transcribe(data)
        segs = []
        for u in utterances:
            txt = (u.get("text") or "").strip()
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

        self.signal(text="Jianying(剪映) free ASR...")
        chunks = split_wav_chunks(self.audio_file, self.cache_folder)
        logger.debug(f'Jianying ASR: {len(chunks)} chunks')
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
                    self.signal(text=f"Jianying ASR chunk {i + 1}/{len(chunks)} ok")
                except Exception as e:
                    errs.append(f"chunk{i + 1}: {e}")
                    logger.error(f'Jianying chunk {i + 1} failed: {e}')

        pairs = [(results[i], chunks[i]) for i in range(len(chunks)) if results[i]]
        if not pairs:
            raise SpeechToTextError(f"剪映ASR failed: {'; '.join(errs) or 'no result'}")
        if errs:
            logger.warning(f'Jianying partial chunks failed: {errs}')

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
