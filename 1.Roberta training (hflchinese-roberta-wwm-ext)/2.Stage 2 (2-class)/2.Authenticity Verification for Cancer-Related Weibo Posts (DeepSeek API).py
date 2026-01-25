# -*- coding: utf-8 -*-
"""
Authenticity Verification for Cancer-Related Weibo Posts (DeepSeek API)
High-concurrency, cacheable pipeline.

Output labels (closed set, single choice):
- Authentic
- Non-authentic
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

INPUT_XLSX = r"data/raw/stage2_input.xlsx"
OUTPUT_XLSX = r"data/derived/stage2_output.xlsx"
CACHE_JSONL = r"data/cache/stage2_auth_cache.jsonl"

CONCURRENCY = 20
BATCH_SIZE = 100

TIMEOUT_SEC = 30
TEMPERATURE = 0.1
MAX_RETRY_PER_MODEL = 5

ID_COL = "wid"
TEXT_COL = "content"
OUTPUT_COL = "authenticity"

FAILED_TAG = "__API_FAILED__"


# =========================
# Label set (Stage 2)
# =========================

LABELS: List[str] = [
    "Authentic",
    "Non-authentic",
]
LABEL_SET = set(LABELS)


# =========================
# Prompt specification (Stage 2)
# =========================

SYSTEM_PROMPT = """
Task:
Decide whether a Chinese Weibo post contains a concrete cancer-related experiential narrative.

Output format:
Return a single-line JSON object with exactly one key:
{"authenticity":"<label>"}
No additional keys. No explanation. No extra text.

Allowed labels (closed set):
- Authentic
- Non-authentic

Operational definitions (evidence-based):
A) Authentic
Use when the post describes a concrete, real-world cancer-related event or lived experience involving a specific person.
Examples of experiential content include diagnosis, treatment, symptoms, tests, recovery, relapse/progression, death,
caregiving actions, or direct emotional reactions to such events.
The described person can be the author or someone else (family/friend/colleague/teacher, etc.), as long as the post
is an experiential narrative rather than abstract commentary.

B) Non-authentic
Use when the post is not an experiential narrative, including:
- Metaphors, jokes, or figurative usage
- Hypothetical statements or wishes
- General advice/education without a concrete case
- News-like reporting, policy promotion, reposts
- Broad claims/rumors without a specific event narrative
If cancer-related terms appear but there is no concrete experiential narrative or event reference, label as Non-authentic.

General principles:
- Use only evidence within the post text.
- Label the post content type (experiential vs non-experiential), not who is speaking.
- Prefer Non-authentic when content is generic/abstract and lacks a concrete case.
""".strip()


USER_PROMPT_TEMPLATE = """
Classify the post below as Authentic vs Non-authentic.

Post text:
"{text}"

Return only one-line JSON:
{{"authenticity":"Authentic"}}
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

    s_low = s.lower()
    if "auth" in s_low or "真实" in s or "经历" in s or "体验" in s:
        return "Authentic"
    if "non" in s_low or "inauth" in s_low or "非" in s or "泛" in s:
        return "Non-authentic"

    return "Non-authentic"


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
                lab = str(obj.get(OUTPUT_COL, "")).strip()
                if wid and lab:
                    cache[wid] = lab
            except Exception:
                continue
    return cache


def _append_cache(cache_path: Path, wid: str, label: str) -> None:
    with cache_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"wid": wid, OUTPUT_COL: label}, ensure_ascii=False) + "\n")


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
    if not obj or OUTPUT_COL not in obj:
        return None
    return _normalize_label(obj[OUTPUT_COL])


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