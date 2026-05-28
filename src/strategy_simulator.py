from __future__ import annotations

import pandas as pd


QUEUE_MAP = {
    "3-5 分钟": 4,
    "6-10 分钟": 8,
    "12-18 分钟": 15,
    "20+ 分钟": 22,
}


def build_strategy_baseline(
    canteen_compare: pd.DataFrame,
    risk_overview: pd.DataFrame,
    crowding: pd.DataFrame,
    efficiency_ranking: pd.DataFrame,
) -> pd.DataFrame:
    if canteen_compare.empty:
        return pd.DataFrame()

    queue_df = crowding[["canteen", "queue_time", "level"]].copy() if not crowding.empty else pd.DataFrame(columns=["canteen", "queue_time", "level"])
    queue_df["queue_minutes"] = queue_df["queue_time"].map(QUEUE_MAP).fillna(8)

    risk_df = (
        risk_overview[["canteen", "shortage_count", "waste_count", "prep_ratio", "risk_level"]].copy()
        if not risk_overview.empty
        else pd.DataFrame(columns=["canteen", "shortage_count", "waste_count", "prep_ratio", "risk_level"])
    )
    eff_df = (
        efficiency_ranking[["canteen", "efficiency_score", "sales_per_student", "orders_per_student"]].copy()
        if not efficiency_ranking.empty
        else pd.DataFrame(columns=["canteen", "efficiency_score", "sales_per_student", "orders_per_student"])
    )

    merged = canteen_compare.merge(risk_df, on="canteen", how="left").merge(queue_df, on="canteen", how="left").merge(eff_df, on="canteen", how="left")
    merged = merged.fillna(
        {
            "shortage_count": 0,
            "waste_count": 0,
            "prep_ratio": 1.1,
            "risk_level": "中",
            "queue_time": "6-10 分钟",
            "level": "中",
            "queue_minutes": 8,
            "efficiency_score": 60,
            "sales_per_student": 0,
            "orders_per_student": 0,
        }
    )

    merged["avg_ticket"] = (merged["sales"] / merged["orders"].clip(lower=1)).round(2)
    merged["baseline_cost"] = (merged["sales"] * 0.62).round(2)
    merged["baseline_profit"] = (merged["sales"] - merged["baseline_cost"]).round(2)
    return merged.sort_values("sales", ascending=False).reset_index(drop=True)
