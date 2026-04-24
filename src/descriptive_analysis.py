from __future__ import annotations

import pandas as pd


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["order_time"] = pd.to_datetime(data["order_time"], errors="coerce")
    data["date"] = data["order_time"].dt.date
    data["hour"] = data["order_time"].dt.hour
    data["weekday"] = data["order_time"].dt.day_name()
    data["weekday_cn"] = data["order_time"].dt.weekday.map(
        {0: "周一", 1: "周二", 2: "周三", 3: "周四", 4: "周五", 5: "周六", 6: "周日"}
    )
    data["period"] = data["hour"].map(period_from_hour)
    return data


def period_from_hour(hour: int) -> str:
    if 6 <= hour < 10:
        return "早餐"
    if 10 <= hour < 14:
        return "午餐"
    if 16 <= hour < 20:
        return "晚餐"
    return "夜宵"


def compute_kpis(df: pd.DataFrame) -> dict[str, float]:
    data = add_time_features(df)
    order_df = data.groupby("order_id", as_index=False).agg(order_amount=("amount", "sum"), date=("date", "first"))
    days = max(order_df["date"].nunique(), 1)
    return {
        "total_sales": float(data["amount"].sum()),
        "total_orders": float(order_df["order_id"].nunique()),
        "total_students": float(data["student_id"].nunique()),
        "avg_ticket": float(order_df["order_amount"].mean()),
        "daily_orders": float(order_df["order_id"].nunique() / days),
    }


def top_n(df: pd.DataFrame, group_col: str, n: int = 10, metric: str = "quantity") -> pd.DataFrame:
    value_col = "quantity" if metric == "quantity" else "amount"
    out = (
        df.groupby(group_col, as_index=False)
        .agg(value=(value_col, "sum"), order_count=("order_id", "nunique"))
        .sort_values("value", ascending=False)
        .head(n)
        .reset_index(drop=True)
    )
    return out
