from __future__ import annotations

import pandas as pd


def build_promo_evaluation(data: pd.DataFrame, promotions: pd.DataFrame) -> pd.DataFrame:
    if data.empty or promotions.empty:
        return pd.DataFrame()

    df = data.copy()
    df["date"] = pd.to_datetime(df["date"])
    latest_date = df["date"].max()
    promo_candidates = promotions[promotions["promo_type"] != "正常"].copy()
    if promo_candidates.empty:
        promo_candidates = promotions.sort_values(["decline_ratio", "recent_rating"]).head(5).copy()

    rows: list[dict] = []
    for idx, promo in promo_candidates.head(6).iterrows():
        dish = promo["dish_name"]
        dish_df = (
            df[df["dish_name"] == dish]
            .groupby("date", as_index=False)
            .agg(quantity=("quantity", "sum"), sales=("amount", "sum"), rating=("rating", "mean"))
            .sort_values("date")
        )
        if dish_df.empty:
            continue

        end_date = min(latest_date, dish_df["date"].max())
        promo_start = end_date - pd.Timedelta(days=2)
        before_start = promo_start - pd.Timedelta(days=7)
        before_end = promo_start - pd.Timedelta(days=1)

        before = dish_df[(dish_df["date"] >= before_start) & (dish_df["date"] <= before_end)]
        after = dish_df[dish_df["date"] >= promo_start]
        if before.empty or after.empty:
            continue

        before_qty = float(before["quantity"].mean())
        after_qty = float(after["quantity"].mean())
        before_sales = float(before["sales"].mean())
        after_sales = float(after["sales"].mean())
        before_rating = float(before["rating"].mean())
        after_rating = float(after["rating"].mean())

        qty_change = (after_qty - before_qty) / max(before_qty, 1)
        sales_change = (after_sales - before_sales) / max(before_sales, 1)
        rating_change = after_rating - before_rating

        outcome = "效果一般"
        if qty_change >= 0.15 and sales_change >= 0.08:
            outcome = "效果明显"
        elif qty_change <= -0.05 or rating_change <= -0.15:
            outcome = "需要复盘"

        rows.append(
            {
                "dish_name": dish,
                "promo_type": promo["promo_type"],
                "before_qty": round(before_qty, 1),
                "after_qty": round(after_qty, 1),
                "qty_change": round(qty_change, 4),
                "before_sales": round(before_sales, 2),
                "after_sales": round(after_sales, 2),
                "sales_change": round(sales_change, 4),
                "before_rating": round(before_rating, 2),
                "after_rating": round(after_rating, 2),
                "rating_change": round(rating_change, 2),
                "outcome": outcome,
                "review": _build_review(outcome, qty_change, sales_change, rating_change),
            }
        )

    return pd.DataFrame(rows).sort_values(["outcome", "qty_change"], ascending=[True, False]).reset_index(drop=True)


def _build_review(outcome: str, qty_change: float, sales_change: float, rating_change: float) -> str:
    if outcome == "效果明显":
        return f"促销后销量提升 {qty_change:.0%}，销售额提升 {sales_change:.0%}，建议保留该策略。"
    if outcome == "需要复盘":
        return f"促销后评分变化 {rating_change:+.2f}，或销量提升不足，建议检查折扣力度与菜品口味。"
    return f"促销后销量变化 {qty_change:.0%}，整体较平稳，可继续小范围测试。"
