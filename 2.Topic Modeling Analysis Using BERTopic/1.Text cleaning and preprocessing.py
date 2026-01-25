# -*- coding: utf-8 -*-
"""
clean_stage.py
癌症患者文本清洗

Note (for reviewers):
This preprocessing removes structural noise (URLs, mentions, hashtags, time expressions, locations/institutions, emojis,
excessive repetition) while preserving a whitelist of medically meaningful English terms/abbreviations to reduce sparsity.
"""

import pandas as pd
import re
from opencc import OpenCC  # pip install opencc


# ==============================
# Path configuration (public version)
# Use local relative paths to avoid leaking any runtime environment details.
# ==============================
INPUT_PATH = "data/raw/content_2nd_label_0.csv"
OUTPUT_PATH = "data/derived/clean_stage.csv"
PLACE_PATH = "resources/china_places.txt"
RESERVED_ENG_PATH = "resources/medical_english_reserved_words.txt"
TEXT_COL = "content"


# ==============================
# 读取医学英文保留表
# ==============================
with open(RESERVED_ENG_PATH, "r", encoding="utf-8") as f:
    RESERVED_ENG = sorted({line.strip() for line in f if line.strip()}, key=len, reverse=True)

RESERVED_ENG_PATTERN = r"|".join([re.escape(w) for w in RESERVED_ENG])
print(f"[INFO] reserved medical English terms: {len(RESERVED_ENG)}")


# ==============================
# 医院关键词
# ==============================
HOSPITAL_KEYS = [
    "医院","肿瘤医院","人民医院","中心医院","中医医院","医学院","附属医院","省医院","市医院",
    "县医院","第一医院","第二医院","第三医院","第四医院","第五医院","协和","华西","湘雅","齐鲁","华山",
    "中山","同济","仁济","新华","友谊","华北","北医三院","中南","西京","医科大学","医大","医学院",
    "医大附属","大学附属","医学院附属","卫生院","医务室","保健院","妇幼保健院","妇产医院","儿童医院",
    "口腔医院","眼科医院","骨科医院","胸科医院","肝胆医院","心血管医院","神经医院","康复医院","整形医院",
    "皮肤医院","肾病医院","结核医院","传染病医院","急救中心","临床中心","肿瘤中心","癌症中心","医疗中心",
    "放疗中心","医养中心","疗养院","门诊部","卫生服务中心","保健中心",
    "人民解放军总医院","中国人民解放军总医院","解放军总医院","总医院","武警医院",
    "空军医院","海军医院","陆军医院","铁路医院","煤矿医院","矿务局医院","工人医院",
    "职工医院","矿工医院","社区医院","街道医院","乡镇医院","地区医院","民医院",
    "市中医院","省中医院","区人民医院","县人民医院","南方医科大学珠江医院"
]
HOSP_PATTERN = r"|".join([re.escape(k) for k in HOSPITAL_KEYS])


# ==============================
# 宗教祈祷关键词
# ==============================
PRAYER_KEYWORDS = [
    "南无","阿弥陀佛","观世音菩萨","地藏王菩萨","药师佛","佛祖","菩萨","保佑",
    "祈祷","天主","耶稣","上帝","阿门","神啊","主啊","主耶稣","显灵","赐福"
]
PRAYER_PATTERN = r"|".join([re.escape(k) for k in PRAYER_KEYWORDS])


# ==============================
# 时间模糊表达（扩展版）
# ==============================
TIME_WORDS = [
    "今天","明天","后天","昨天","前天","刚刚","最近","这几天","这些天","那天","那晚","那年",
    "今年","去年","前年","明年","后年","年初","年末","年前","年后","上半年","下半年",
    "早上","早晨","清晨","上午","中午","下午","晚上","凌晨","傍晚","夜里","深夜",
    "小年","大年","除夕","春节","大年三十","大年二十九",
    "大年初一","大年初二","大年初三","大年初四","大年初五","大年初六",
    "周一","周二","周三","周四","周五","周六","周日","周天",
    "星期一","星期二","星期三","星期四","星期五","星期六","星期日","星期天"
]
TIME_PATTERN = r"|".join([re.escape(w) for w in sorted(TIME_WORDS, key=len, reverse=True)])


# ==============================
# 加载地名库
# ==============================
with open(PLACE_PATH, "r", encoding="utf-8") as f:
    CITY_LIST = [x.strip() for x in f if x.strip()]
CITY_PATTERN = r"|".join([re.escape(c) for c in CITY_LIST])

print(f"[INFO] place-name entries loaded: {len(CITY_LIST)}")


# ==============================
# 情绪重复折叠（哈哈哈哈 → 哈哈）
# ==============================
def normalize_repeats(text: str) -> str:
    text = re.sub(r"(happy){2,}", "happy", text, flags=re.IGNORECASE)
    text = re.sub(r"([\u4e00-\u9fa5A-Za-z])\1{2,}", r"\1\1", text)
    return text


# ==============================
# 核心清洗函数（单轮）
# ==============================
cc_trans = OpenCC("t2s")

def clean_once(raw: str) -> str:
    text = str(raw)

    text = cc_trans.convert(text)
    text = re.sub(r"\s+", "", text)

    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"@\S+", "", text)
    text = re.sub(r"#([^#]{1,50})#", "", text)
    text = re.sub(r"^.*?超话", "", text)

    if re.search(PRAYER_PATTERN, text):
        return ""

    text = re.sub(r"\d{4}年\d{1,2}月\d{1,2}日?", "", text)
    text = re.sub(r"\d{4}年\d{1,2}月", "", text)
    text = re.sub(r"\d{4}年", "", text)
    text = re.sub(r"\d{1,2}月\d{1,2}日?", "", text)
    text = re.sub(r"\d{1,2}\.\d{1,2}(\.\d{1,2})*", "", text)
    text = re.sub(r"\d{1,2}点\d{1,2}分?", "", text)
    text = re.sub(r"\d{1,2}点", "", text)
    text = re.sub(r"\d+号(?!楼|床|房)", "", text)

    text = re.sub(TIME_PATTERN, "", text)

    text = re.sub(CITY_PATTERN, "", text)
    text = re.sub(HOSP_PATTERN, "", text)

    text = re.sub(r"（[^）]{1,20}）", "", text)
    text = re.sub(r"\([^)]{1,20}\)", "", text)

    text = normalize_repeats(text)

    # Preserve medically meaningful English abbreviations via placeholder swapping.
    for w in RESERVED_ENG:
        text = re.sub(rf"{re.escape(w)}", f"[[[{w}]]]", text, flags=re.IGNORECASE)

    text = re.sub(r"[A-Za-z0-9]+", "", text)

    for w in RESERVED_ENG:
        text = text.replace(f"[[[{w}]]]", w)

    text = re.sub(r"[^A-Za-z0-9\u4e00-\u9fa5]", "", text)
    text = re.sub(r"\d+", "", text)

    return text.strip()


# ==============================
# 主流程
# ==============================
if __name__ == "__main__":
    df = pd.read_csv(INPUT_PATH)
    assert TEXT_COL in df.columns

    print(f"[INFO] raw rows: {len(df)}")

    df["clean_content"] = df[TEXT_COL].astype(str).apply(clean_once)
    df["clean_content"] = df["clean_content"].astype(str).apply(clean_once)

    df = df[df["clean_content"].str.strip().astype(bool)]
    df = df[df["clean_content"].str.len() > 6]

    if "wid" in df.columns:
        df = df[["wid", "clean_content"]]
    else:
        df = df[["clean_content"]]

    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print(f"[INFO] cleaned rows: {len(df)}")
    print("[INFO] output written.")