from __future__ import annotations

from collections import Counter

import pandas as pd


POSITIVE_KEYWORDS = ["好吃", "分量足", "便宜", "出餐快", "干净", "味道稳定", "性价比高", "搭配合理"]
NEGATIVE_KEYWORDS = ["太咸", "太油", "分量少", "排队久", "价格贵", "出餐慢", "不够热", "太甜"]

POSITIVE_TEMPLATES = [
    "{dish_name}味道稳定，吃起来很满意。",
    "{dish_name}分量足，而且出餐快。",
    "{dish_name}性价比高，下次还会再点。",
    "{canteen}这道{dish_name}很干净，整体体验不错。",
]

NEGATIVE_TEMPLATES = [
    "{dish_name}今天有点太咸，口味需要调整。",
    "{canteen}这道{dish_name}出餐慢，排队时间有点长。",
    "{dish_name}分量偏少，感觉性价比不高。",
    "{dish_name}有点太油，吃完负担比较重。",
]


def _build_synthetic_feedback(data: pd.DataFrame, max_rows: int = 80) -> pd.DataFrame:
    ranked = (
        data.groupby(["dish_name", "canteen"], as_index=False)
        .agg(avg_score=("rating", "mean"), freq=("order_id", "nunique"))
        .sort_values(["avg_score", "freq"], ascending=[False, False])
    )

    positive = ranked.head(max_rows // 2).copy()
    negative = ranked.tail(max_rows // 2).copy().sort_values(["avg_score", "freq"], ascending=[True, False])
    rows: list[dict] = []

    for idx, row in positive.iterrows():
        template = POSITIVE_TEMPLATES[idx % len(POSITIVE_TEMPLATES)]
        rows.append(
            {
                "dish_name": row["dish_name"],
                "canteen": row["canteen"],
                "comment": template.format(dish_name=row["dish_name"], canteen=row["canteen"]),
                "avg_score": round(float(max(row["avg_score"], 4.2)), 2),
                "sentiment": "positive",
            }
        )

    for idx, row in negative.iterrows():
        template = NEGATIVE_TEMPLATES[idx % len(NEGATIVE_TEMPLATES)]
        rows.append(
            {
                "dish_name": row["dish_name"],
                "canteen": row["canteen"],
                "comment": template.format(dish_name=row["dish_name"], canteen=row["canteen"]),
                "avg_score": round(float(min(row["avg_score"], 3.8)), 2),
                "sentiment": "negative",
            }
        )

    return pd.DataFrame(rows)


def _prepare_comment_frame(feedback: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
    if feedback.empty or "comment" not in feedback.columns or feedback["comment"].fillna("").eq("").all():
        return _build_synthetic_feedback(data)

    fb = feedback.copy()
    score_cols = [col for col in ["taste_score", "portion_score", "price_score", "service_score"] if col in fb.columns]
    if score_cols:
        fb[score_cols] = fb[score_cols].apply(pd.to_numeric, errors="coerce")
        fb["avg_score"] = fb[score_cols].mean(axis=1)
    elif "avg_score" not in fb.columns:
        fb["avg_score"] = 4.0

    fb["comment"] = fb["comment"].fillna("").astype(str)
    fb = fb[fb["comment"] != ""].copy()
    if fb.empty:
        return _build_synthetic_feedback(data)

    fb["sentiment"] = fb["avg_score"].apply(lambda x: "positive" if float(x) >= 4.0 else "negative")
    return fb[["dish_name", "canteen", "comment", "avg_score", "sentiment"]].reset_index(drop=True)


def _count_keywords(comments: pd.Series, keywords: list[str]) -> list[dict]:
    counter: Counter[str] = Counter()
    for comment in comments:
        text = str(comment)
        for word in keywords:
            if word in text:
                counter[word] += 1
    return [{"keyword": key, "count": value} for key, value in counter.most_common(8)]


def build_comment_keyword_analysis(feedback: pd.DataFrame, data: pd.DataFrame) -> dict:
    comment_df = _prepare_comment_frame(feedback, data)
    positive_df = comment_df[comment_df["sentiment"] == "positive"]
    negative_df = comment_df[comment_df["sentiment"] == "negative"]

    positive_keywords = _count_keywords(positive_df["comment"], POSITIVE_KEYWORDS)
    negative_keywords = _count_keywords(negative_df["comment"], NEGATIVE_KEYWORDS)

    theme_rows = [
        {"theme": "口味问题", "count": sum(item["count"] for item in negative_keywords if item["keyword"] in {"太咸", "太甜", "太油"})},
        {"theme": "出餐效率", "count": sum(item["count"] for item in negative_keywords if item["keyword"] in {"排队久", "出餐慢"})},
        {"theme": "价格与分量", "count": sum(item["count"] for item in negative_keywords if item["keyword"] in {"分量少", "价格贵"})},
        {"theme": "正向体验", "count": sum(item["count"] for item in positive_keywords)},
    ]
    theme_rows = [row for row in theme_rows if row["count"] > 0]

    dish_issue = (
        negative_df.groupby(["dish_name", "canteen"], as_index=False)
        .agg(issue_count=("comment", "count"), avg_score=("avg_score", "mean"))
        .sort_values(["issue_count", "avg_score"], ascending=[False, True])
        .head(8)
    )

    return {
        "positiveKeywords": positive_keywords,
        "negativeKeywords": negative_keywords,
        "themeCounts": theme_rows,
        "samplePositive": positive_df[["dish_name", "canteen", "comment"]].head(6).to_dict(orient="records"),
        "sampleNegative": negative_df[["dish_name", "canteen", "comment"]].head(6).to_dict(orient="records"),
        "dishIssues": dish_issue.to_dict(orient="records"),
        "commentCount": int(len(comment_df)),
    }
