# -*- coding: utf-8 -*-
"""
Structured Annotation of Social Media Narratives
LLM-assisted, research-grade, reproducible implementation.

IMPORTANT:
This script is a reusable execution framework. It is designed to be paired with a specific
prompt/specification (SYSTEM_PROMPT + USER_PROMPT_TEMPLATE + label constraints).
When applying it to a different annotation task, you should replace the prompt section
and update VALID_LABELS / output key expectations accordingly.
"""

import json
import re
import asyncio
from pathlib import Path

import httpx
import pandas as pd
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Configuration (public placeholders)
# - Do NOT hardcode secrets in public code. Prefer environment variables.
# - Paths are intentionally generic in the public version.
# -----------------------------------------------------------------------------
API_KEY = "YOUR_API_KEY_HERE"
MODEL = "your-llm-model"

INPUT_XLSX = "data/input.xlsx"
OUTPUT_XLSX = "outputs/output.xlsx"
CACHE_JSONL = "outputs/cache.jsonl"

TEXT_COL = "text"
ID_COL = "id"
OUTPUT_COL = "annotation"

CONCURRENCY = 10
BATCH_SIZE = 100
MAX_RETRY = 3
TIMEOUT_SEC = 30
TEMPERATURE = 0.1

FAILED_TAG = "__API_FAILED__"

# -----------------------------------------------------------------------------
# Prompt specification (TASK-DEPENDENT)
# Replace these prompts to match your labeling schema.
# The model is expected to return JSON only (no explanations).
# -----------------------------------------------------------------------------
SYSTEM_PROMPT = "SYSTEM PROMPT PLACEHOLDER"

USER_PROMPT_TEMPLATE = """
Task: Annotate the following text according to a predefined labeling schema.

Text:
"{content}"

Output JSON only.
""".strip()

# -----------------------------------------------------------------------------
# Output constraints (TASK-DEPENDENT)
# Update VALID_LABELS and the expected output key ("label" below) to match your task.
# -----------------------------------------------------------------------------
VALID_LABELS = {"Label_A", "Label_B", "Label_C"}


def build_prompt(content: str) -> str:
    # Build the user prompt from the template (task-dependent content injection)
    return USER_PROMPT_TEMPLATE.format(content=(content or "").strip())


def safe_json_parse(text: str):
    # Robust JSON extraction for typical LLM responses (may include code fences)
    if not text:
        return None
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.I).strip()
    try:
        return json.loads(text)
    except Exception:
        s, e = text.find("{"), text.rfind("}")
        if s != -1 and e != -1 and e > s:
            try:
                return json.loads(text[s:e + 1])
            except Exception:
                return None
    return None


def normalize_label(x):
    # Enforce closed-set labels. Adjust this function if your task is multi-label
    # or requires additional normalization/mapping.
    if isinstance(x, str) and x in VALID_LABELS:
        return x
    return None


async def call_api(client: httpx.AsyncClient, content: str):
    # Single-item API call with retries.
    # NOTE: URL and payload format must match your provider.
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(content)},
        ],
        "temperature": TEMPERATURE,
    }

    url = "https://api.example.com/chat/completions"

    for _ in range(MAX_RETRY):
        try:
            r = await client.post(
                url,
                json=payload,
                headers={"Authorization": f"Bearer {API_KEY}"},
                timeout=TIMEOUT_SEC,
            )
            if r.status_code == 200:
                raw = r.json()["choices"][0]["message"]["content"]
                parsed = safe_json_parse(raw)

                # TASK-DEPENDENT: expected JSON key is "label"
                if parsed and "label" in parsed:
                    label = normalize_label(parsed["label"])
                    if label:
                        return label
        except Exception:
            await asyncio.sleep(1)

    return FAILED_TAG


async def process_item(item, client, semaphore):
    # Concurrency guard for rate limiting and stability
    async with semaphore:
        return await call_api(client, item["content"])


async def run():
    # Load input table
    df = pd.read_excel(INPUT_XLSX)
    df[ID_COL] = df[ID_COL].astype(str)

    # Load cache for resumable processing (id -> annotation)
    cache = {}
    cache_path = Path(CACHE_JSONL)
    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    cache[obj[ID_COL]] = obj[OUTPUT_COL]
                except Exception:
                    continue

    # Build pending queue (skip cached and empty rows)
    pending = [
        {"id": str(row[ID_COL]), "content": str(row.get(TEXT_COL, "")).strip()}
        for _, row in df.iterrows()
        if str(row[ID_COL]) not in cache
        and str(row.get(TEXT_COL, "")).strip().lower() != "nan"
    ]

    print(f"[INFO] total_rows={len(df)} pending={len(pending)}")

    semaphore = asyncio.Semaphore(CONCURRENCY)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Run batched async annotation with periodic cache writes
    async with httpx.AsyncClient() as client:
        with tqdm(total=len(pending), desc="Annotation") as pbar:
            for i in range(0, len(pending), BATCH_SIZE):
                batch = pending[i : i + BATCH_SIZE]
                results = await asyncio.gather(
                    *[process_item(item, client, semaphore) for item in batch]
                )

                # Append successful results to cache JSONL (audit-friendly)
                with cache_path.open("a", encoding="utf-8") as f:
                    for item, res in zip(batch, results):
                        if res != FAILED_TAG:
                            f.write(
                                json.dumps(
                                    {ID_COL: item["id"], OUTPUT_COL: res},
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
                            cache[item["id"]] = res

                pbar.update(len(batch))

    # Write final outputs (merge cache back to the full dataframe)
    df[OUTPUT_COL] = df[ID_COL].map(lambda x: cache.get(str(x), FAILED_TAG))
    Path(OUTPUT_XLSX).parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(OUTPUT_XLSX, index=False)
    print("[INFO] done")


if __name__ == "__main__":
    asyncio.run(run())