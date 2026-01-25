# -*- coding: utf-8 -*-
"""
extract_english_tokens_v1.py
从原始癌症文本中抽取所有英文/英文+数字 token，统计频次，为后续“医学白名单”提供依据
"""

import pandas as pd
import re
from collections import Counter

# ========== 路径配置 ==========
INPUT_PATH = "/kaggle/input/bertopic-predata/content_2nd_label_0.csv"  # 原始文件
OUTPUT_PATH = "/kaggle/working/english_tokens_stats.csv"               # 统计结果输出
TEXT_COL = "content"                                                   # 文本列名


# ========== 主程序 ==========
def main():
    # 1. 读取数据
    df = pd.read_csv(INPUT_PATH)
    assert TEXT_COL in df.columns, f"未找到列：{TEXT_COL}"
    print(f"📂 原始数据量: {len(df)}")

    # 2. 准备计数器
    #  ALPHA: 纯英文 (CT, MRI, EGFR)
    #  ALNUM: 英文+数字 (CA125, FOLFOX6, PD1)
    alpha_counter = Counter()
    alnum_counter = Counter()

    # 可调参数：最短 token 长度，避免 a、i 这种无意义噪音
    MIN_LEN = 2

    # 3. 正则提取 token: 连续的 [A-Za-z0-9]
    pattern = re.compile(r"[A-Za-z0-9]+")

    for i, text in enumerate(df[TEXT_COL].astype(str), start=1):
        tokens = pattern.findall(text)
        for tok in tokens:
            if len(tok) < MIN_LEN:
                continue

            # 统一转大写做归并统计（CA125 / ca125 视为同一 token）
            key = tok.upper()

            has_alpha = any(c.isalpha() for c in key)
            has_digit = any(c.isdigit() for c in key)

            if has_alpha and not has_digit:
                # 纯字母：CT, MRI, ALT, AFP, FOLFOX 等
                alpha_counter[key] += 1
            elif has_alpha and has_digit:
                # 字母 + 数字：CA125, CA199, FOLFOX6, PD1 等
                alnum_counter[key] += 1
            else:
                # 纯数字这里暂时不管，主要看“英文相关”的 token
                pass

        if i % 50000 == 0:
            print(f"  已处理 {i} 条…")

    # 4. 汇总为 DataFrame

    rows = []

    for tok, cnt in alpha_counter.items():
        rows.append({"token": tok, "kind": "ALPHA", "count": cnt})

    for tok, cnt in alnum_counter.items():
        rows.append({"token": tok, "kind": "ALNUM", "count": cnt})

    stats_df = pd.DataFrame(rows)

    # 按频次从高到低排序，方便你筛选
    stats_df = stats_df.sort_values(by="count", ascending=False).reset_index(drop=True)

    # 也可以设置一个最小频次阈值，比如只保留出现 >= 3 次的 token
    MIN_COUNT = 2
    stats_df = stats_df[stats_df["count"] >= MIN_COUNT]

    # 5. 保存结果
    stats_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print(f"✅ 抽取完成，共 {len(stats_df)} 个英文/英文数字 token")
    print(f"📄 结果文件：{OUTPUT_PATH}")
    print("\n🔍 Top 30 预览：")
    print(stats_df.head(30))


if __name__ == "__main__":
    main()