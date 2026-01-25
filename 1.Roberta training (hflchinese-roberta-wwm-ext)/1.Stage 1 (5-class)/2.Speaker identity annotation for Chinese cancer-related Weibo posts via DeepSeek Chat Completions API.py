# -*- coding: utf-8 -*-
"""
Speaker Identity Annotation for Cancer-Related Weibo Posts (DeepSeek API)
High-concurrency, cacheable, research-grade pipeline.

Output labels (closed set, single choice):
1) 患者本人发帖
2) 亲属发帖
3) 朋友和其他非照顾性的社会关系
4) 其他非肿瘤相关（广告、引流、垃圾文本）
5) 无法判断
"""

import os
import json
import re
import asyncio
from pathlib import Path
from typing import Any, Dict, Optional, List

import httpx
import pandas as pd
from tqdm import tqdm


# =========================
# Configuration
# =========================

API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
if not API_KEY:
    API_KEY = "YOUR_API_KEY_HERE"

API_URL = "https://api.deepseek.com/chat/completions"

MODEL_CANDIDATES = [
    "deepseek-chat",
]

INPUT_XLSX = r"D:\your_path\input.xlsx"
OUTPUT_XLSX = r"D:\your_path\output.xlsx"
CACHE_JSONL = r"D:\your_path\speaker_cache.jsonl"

CONCURRENCY = 20
BATCH_SIZE = 100

TIMEOUT_SEC = 30
TEMPERATURE = 0.1
MAX_RETRY_PER_MODEL = 5

ID_COL = "wid"
TEXT_COL = "content"
OUTPUT_COL = "speaker_type"

FAILED_TAG = "__API_FAILED__"


# =========================
# Label set
# =========================

LABELS: List[str] = [
    "患者本人发帖",
    "亲属发帖",
    "朋友和其他非照顾性的社会关系",
    "其他非肿瘤相关（广告、引流、垃圾文本）",
    "无法判断",
]
LABEL_SET = set(LABELS)


# =========================
# Prompt specification
# =========================

SYSTEM_PROMPT = """
Task:
Assign exactly one speaker identity label for a Chinese Weibo post, using only evidence within the post text.

Output format:
Return a single-line JSON object with exactly one key:
{"speaker_type":"<label>"}
No additional keys. No explanation. No extra text.

Allowed labels (closed set):
- 患者本人发帖
- 亲属发帖
- 朋友和其他非照顾性的社会关系
- 其他非肿瘤相关（广告、引流、垃圾文本）
- 无法判断

Decision rules (evidence-based):
A. 患者本人发帖 (patient self-narrative)
Use when the author is the cancer patient describing their own diagnosis, treatment, symptoms, tests, prognosis, or emotions.
Strong cues include first-person statements tied to cancer care, for example:
"我确诊", "我得了XX癌", "我化疗/放疗/靶向/免疫/手术", "我复发/转移",
"我疼/我吐/我发烧/我白细胞低", "我的病理/检查报告".

B. 亲属发帖 (family caregiver or relative)
Use when the author is a family member or close caregiver describing a patient's cancer experience.
Strong cues include explicit kinship terms with cancer context:
"我妈/我爸/我老公/我老婆/我孩子/爷爷/奶奶/婆婆/公公"
and caregiving actions:
"陪他她看病", "照顾/陪护/挂号/住院办理/筹钱", reporting patient's indicators as an observer.

C. 朋友和其他非照顾性的社会关系 (friend or non-family social tie)
Use when the author is a friend, colleague, classmate, neighbor, teacher, or other non-family tie.
Strong cues:
"朋友/同事/同学/闺蜜/老师/领导/邻居" with cancer context,
episodic exposure or relayed information:
"听说/得知/看到朋友圈/她告诉我/去探望".

D. 其他非肿瘤相关（广告、引流、垃圾文本）
Use when the text does not constitute a specific personal cancer narrative and primarily serves promotion, lead generation, generic health education, news reposting, or noisy content.
Cues include:
"加微信/私信/咨询/课程/带货/领取资料/关注账号" or purely generic content without a concrete personal case.

E. 无法判断 (indeterminate)
Use when the post lacks sufficient evidence for A–D, or references are ambiguous or conflicting.
Prefer "无法判断" over guessing.

General principles:
- Label the author (speaker), not the person mentioned in the text.
- Do not use usernames, profile information, or assumptions.
- If multiple roles are mentioned, decide based on the author perspective.
""".strip()


USER_PROMPT_TEMPLATE = """
Classify the speaker identity label for the post below.

Post text:
"{text}"

Return only one-line JSON:
{{"speaker_type":"患者本人发帖"}}
""".strip()


# =========================
# Helpers
# =========================

def _compact_text(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def _strip_code_fences(s: str) -> str:
    if not s:
        return ""
    t = s.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.I).strip()
    t = re.sub(r"\s*```$", "", t, flags=re.I).strip()
    return t


def _safe_json_load(s: str) -> Optional[Dict[str, Any]]:
    if not s:
        return None
    t = _strip_code_fences(s)
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else None
    except Exception:
        start, end = t.find("{"), t.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                obj = json.loads(t[start:end + 1])
                return obj if isinstance(obj, dict) else None
            except Exception:
                return None
    return None


def _normalize_label(raw: Any) -> str:
    s = str(raw or "").strip()
    if s in LABEL_SET:
        return s

    # Conservative normalization for minor variations
    if "无法" in s or "不确定" in s or "未知" in s or "判断" in s:
        return "无法判断"
    if "患者" in s or "自述" in s or "本人" in s:
        return "患者本人发帖"
    if "亲属" in s or "家属" in s or "家人" in s:
        return "亲属发帖"
    if "朋友" in s or "同事" in s or "同学" in s or "邻居" in s:
        return "朋友和其他非照顾性的社会关系"
    if "广告" in s or "引流" in s or "垃圾" in s or "科普" in s or "新闻" in s:
        return "其他非肿瘤相关（广告、引流、垃圾文本）"

    return "无法判断"


def _load_cache(cache_path: Path) -> Dict[str, str]:
    cache: Dict[str, str] = {}
    if not cache_path.exists():
        return cache
    with cache_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                wid = str(obj.get("wid", "")).strip()
                lab = str(obj.get("speaker_type", "")).strip()
                if wid and lab:
                    cache[wid] = lab
            except Exception:
                continue
    return cache


def _append_cache(cache_path: Path, wid: str, label: str) -> None:
    with cache_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"wid": wid, "speaker_type": label}, ensure_ascii=False) + "\n")


# =========================
# API call
# =========================

async def _call_once(client: httpx.AsyncClient, model: str, text: str) -> Optional[str]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(text=_compact_text(text))},
        ],
        "temperature": TEMPERATURE,
    }
    r = await client.post(
        API_URL,
        json=payload,
        headers={"Authorization": f"Bearer {API_KEY}"},
        timeout=TIMEOUT_SEC,
    )
    if r.status_code != 200:
        return None

    data = r.json()
    content = data["choices"][0]["message"]["content"]
    obj = _safe_json_load(content)
    if not obj or "speaker_type" not in obj:
        return None
    return _normalize_label(obj["speaker_type"])


async def annotate_one(client: httpx.AsyncClient, text: str) -> str:
    for model in MODEL_CANDIDATES:
        for attempt in range(MAX_RETRY_PER_MODEL):
            try:
                out = await _call_once(client, model, text)
                if out:
                    return out
            except Exception:
                pass
            await asyncio.sleep(min(10, 2 ** attempt))
    return FAILED_TAG


async def worker(item: Dict[str, str], client: httpx.AsyncClient, sem: asyncio.Semaphore) -> str:
    async with sem:
        return await annotate_one(client, item["text"])


# =========================
# Main
# =========================

async def main() -> None:
    df = pd.read_excel(INPUT_XLSX)

    if ID_COL not in df.columns:
        raise ValueError(f"Missing required column: {ID_COL}")
    if TEXT_COL not in df.columns:
        raise ValueError(f"Missing required column: {TEXT_COL}")

    df[ID_COL] = df[ID_COL].astype(str)

    cache_path = Path(CACHE_JSONL)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = _load_cache(cache_path)

    pending: List[Dict[str, str]] = []
    for _, row in df.iterrows():
        wid = str(row[ID_COL]).strip()
        text = str(row.get(TEXT_COL, "")).strip()
        if not wid:
            continue
        if wid in cache:
            continue
        if not text or text.lower() == "nan":
            continue
        pending.append({"wid": wid, "text": text})

    print(f"Total rows: {len(df)} | Cached: {len(cache)} | Pending: {len(pending)}")

    sem = asyncio.Semaphore(CONCURRENCY)

    async with httpx.AsyncClient() as client:
        with tqdm(total=len(pending), desc="Annotating") as pbar:
            for i in range(0, len(pending), BATCH_SIZE):
                batch = pending[i:i + BATCH_SIZE]
                results = await asyncio.gather(*[worker(it, client, sem) for it in batch])

                for it, lab in zip(batch, results):
                    if lab == FAILED_TAG:
                        continue
                    cache[it["wid"]] = lab
                    _append_cache(cache_path, it["wid"], lab)

                pbar.update(len(batch))

    df[OUTPUT_COL] = df[ID_COL].map(lambda x: cache.get(str(x), FAILED_TAG))

    out_path = Path(OUTPUT_XLSX)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(OUTPUT_XLSX, index=False)
    print("Done. Output written.")


if __name__ == "__main__":
    asyncio.run(main())