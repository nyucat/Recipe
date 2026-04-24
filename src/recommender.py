from __future__ import annotations

import pandas as pd


def recommend_hot_dishes(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    stat = (
        df.groupby("dish_name", as_index=False)
        .agg(
            sales_qty=("quantity", "sum"),
            avg_rating=("rating", "mean"),
            order_count=("order_id", "nunique"),
        )
        .copy()
    )
    stat["score"] = (
        stat["sales_qty"] / stat["sales_qty"].max()
        + stat["avg_rating"] / 5
        + stat["order_count"] / stat["order_count"].max()
    ) / 3
    out = stat.sort_values("score", ascending=False).head(top_n).reset_index(drop=True)
    out["reason"] = "热门 + 评分较高 + 复购稳定"
    return out


def recommend_for_student(
    df: pd.DataFrame,
    student_cluster_df: pd.DataFrame,
    student_id: str,
    top_n: int = 6,
) -> pd.DataFrame:
    if student_id not in set(student_cluster_df["student_id"]):
        return pd.DataFrame()

    cluster = student_cluster_df.loc[student_cluster_df["student_id"] == student_id, "cluster"].iloc[0]
    seen = set(df.loc[df["student_id"] == student_id, "dish_name"])
    peer_students = set(student_cluster_df.loc[student_cluster_df["cluster"] == cluster, "student_id"])
    peer_df = df[df["student_id"].isin(peer_students)]
    cand = (
        peer_df.groupby("dish_name", as_index=False)
        .agg(sales_qty=("quantity", "sum"), avg_rating=("rating", "mean"), order_count=("order_id", "nunique"))
        .sort_values(["sales_qty", "avg_rating"], ascending=False)
        .copy()
    )
    cand = cand[~cand["dish_name"].isin(seen)].head(top_n).reset_index(drop=True)
    cand["reason"] = "与你同消费群体同学偏好相似"
    return cand
