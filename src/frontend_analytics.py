from __future__ import annotations

import json

import pandas as pd


def json_ready_records(df: pd.DataFrame, limit: int | None = None) -> list[dict]:
    if df is None or df.empty:
        return []
    out = df.copy()
    for col in out.columns:
        if str(out[col].dtype).startswith("datetime"):
            out[col] = out[col].astype(str)
    if limit is not None:
        out = out.head(limit)
    return json.loads(out.to_json(orient="records", force_ascii=False))


def build_combo_suggestions(rules: pd.DataFrame, dish_sales: pd.DataFrame) -> pd.DataFrame:
    if rules.empty or dish_sales.empty:
        return pd.DataFrame()

    price_map = dish_sales.set_index("dish_name")["avg_price"].to_dict()
    rows: list[dict] = []
    for _, row in rules.head(20).iterrows():
        dishes = [item for item in [row["antecedents_str"], row["consequents_str"]] if item]
        total_price = sum(float(price_map.get(dish, 0)) for dish in dishes)
        rows.append(
            {
                "combo_name": " + ".join(dishes),
                "dishes": dishes,
                "estimated_price": round(total_price, 1),
                "confidence": round(float(row["confidence"]), 4),
                "lift": round(float(row["lift"]), 4),
                "reason": f"提升度 {float(row['lift']):.2f}，适合做套餐联动推荐。",
            }
        )
    return pd.DataFrame(rows)


def build_lifecycle(data: pd.DataFrame) -> pd.DataFrame:
    daily = data.groupby(["dish_name", "date"], as_index=False).agg(
        sales_qty=("quantity", "sum"),
        avg_rating=("rating", "mean"),
    )
    latest = pd.to_datetime(daily["date"]).max()
    recent = daily[pd.to_datetime(daily["date"]) > latest - pd.Timedelta(days=7)]
    previous = daily[
        (pd.to_datetime(daily["date"]) <= latest - pd.Timedelta(days=7))
        & (pd.to_datetime(daily["date"]) > latest - pd.Timedelta(days=14))
    ]

    recent_agg = recent.groupby("dish_name", as_index=False).agg(
        recent_qty=("sales_qty", "sum"),
        avg_rating=("avg_rating", "mean"),
    )
    prev_agg = previous.groupby("dish_name", as_index=False).agg(prev_qty=("sales_qty", "sum"))
    merged = recent_agg.merge(prev_agg, on="dish_name", how="left").fillna({"prev_qty": 0})
    merged["growth"] = merged.apply(
        lambda row: 1.0
        if row["prev_qty"] == 0 and row["recent_qty"] > 0
        else (row["recent_qty"] - row["prev_qty"]) / max(row["prev_qty"], 1),
        axis=1,
    )

    merged["stage"] = "稳定期"
    merged.loc[(merged["prev_qty"] == 0) & (merged["recent_qty"] > 0), "stage"] = "新品期"
    merged.loc[merged["growth"] >= 0.2, "stage"] = "成长期"
    merged.loc[merged["growth"] <= -0.2, "stage"] = "衰退期"
    merged.loc[(merged["growth"] <= -0.35) & (merged["avg_rating"] < 3.9), "stage"] = "淘汰观察期"

    advice_map = {
        "新品期": "建议增加曝光并观察复购表现。",
        "成长期": "建议增加供应，重点保障高峰时段备餐。",
        "稳定期": "建议保持供应，关注评分和口味波动。",
        "衰退期": "建议促销或微调配方，测试学生接受度。",
        "淘汰观察期": "建议考虑替换菜品或下架试运行。",
    }
    merged["advice"] = merged["stage"].map(advice_map)
    return merged.sort_values(["stage", "recent_qty"], ascending=[True, False]).reset_index(drop=True)


def build_feedback_insights(feedback: pd.DataFrame) -> dict:
    if feedback.empty:
        return {"dishScores": [], "canteenScores": [], "lowComments": []}

    fb = feedback.copy()
    score_cols = [col for col in ["taste_score", "portion_score", "price_score", "service_score"] if col in fb.columns]
    if not score_cols:
        return {"dishScores": [], "canteenScores": [], "lowComments": []}

    fb[score_cols] = fb[score_cols].apply(pd.to_numeric, errors="coerce")
    fb["avg_score"] = fb[score_cols].mean(axis=1)

    dish_scores = (
        fb.groupby("dish_name", as_index=False)
        .agg(avg_score=("avg_score", "mean"), feedbacks=("dish_name", "count"))
        .sort_values(["avg_score", "feedbacks"], ascending=[False, False])
    )
    canteen_scores = (
        fb.groupby("canteen", as_index=False)
        .agg(avg_score=("avg_score", "mean"), feedbacks=("canteen", "count"))
        .sort_values(["avg_score", "feedbacks"], ascending=[False, False])
    )
    low_comments = (
        fb[fb["avg_score"] <= 3.5][["dish_name", "canteen", "comment", "avg_score"]]
        .sort_values("avg_score")
        .head(10)
    )

    return {
        "dishScores": json_ready_records(dish_scores),
        "canteenScores": json_ready_records(canteen_scores),
        "lowComments": json_ready_records(low_comments),
    }


def build_resource_utilization(canteen_compare: pd.DataFrame, meal_plan: pd.DataFrame) -> pd.DataFrame:
    if canteen_compare.empty:
        return pd.DataFrame()

    if "canteen" in meal_plan.columns:
        plan = meal_plan.groupby("canteen", as_index=False).agg(
            pred_total=("pred_quantity", "sum"),
            prep_total=("suggested_prep", "sum"),
        )
    else:
        plan = pd.DataFrame(columns=["canteen", "pred_total", "prep_total"])

    merged = canteen_compare.merge(plan, on="canteen", how="left").fillna({"pred_total": 0, "prep_total": 0})
    merged["utilization_index"] = merged.apply(
        lambda row: round((row["orders"] / max(row["students"], 1)) * 10 + (row["sales"] / max(row["orders"], 1)), 2),
        axis=1,
    )
    return merged.sort_values("utilization_index", ascending=False).reset_index(drop=True)
