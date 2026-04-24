from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.descriptive_analysis import add_time_features


def build_student_features(df: pd.DataFrame) -> pd.DataFrame:
    data = add_time_features(df)
    base = (
        data.groupby("student_id", as_index=False)
        .agg(
            total_amount=("amount", "sum"),
            order_count=("order_id", "nunique"),
            avg_ticket=("amount", "mean"),
            avg_rating=("rating", "mean"),
            avg_price=("price", "mean"),
            high_price_ratio=("price", lambda s: float((s >= 15).mean())),
            category_diversity=("category", "nunique"),
        )
        .copy()
    )

    period_ratio = (
        data.pivot_table(index="student_id", columns="period", values="order_id", aggfunc="count", fill_value=0)
        .reset_index()
        .copy()
    )
    period_cols = [c for c in period_ratio.columns if c != "student_id"]
    total = period_ratio[period_cols].sum(axis=1).replace(0, 1)
    for c in period_cols:
        period_ratio[f"{c}_ratio"] = period_ratio[c] / total
    keep_cols = ["student_id"] + [c for c in period_ratio.columns if c.endswith("_ratio")]
    period_ratio = period_ratio[keep_cols]

    feat = base.merge(period_ratio, on="student_id", how="left").fillna(0)
    return feat


def _label_cluster(row: pd.Series) -> str:
    if row.get("夜宵_ratio", 0) > 0.25:
        return "夜宵偏好型"
    if row.get("早餐_ratio", 0) > 0.35:
        return "早餐依赖型"
    if row.get("high_price_ratio", 0) > 0.45 and row.get("avg_ticket", 0) > 16:
        return "低频高消费型"
    if row.get("order_count", 0) > row.get("order_count_median", 0):
        return "高频刚需型"
    return "价格敏感型"


def train_user_clusters(df: pd.DataFrame, n_clusters: int = 4) -> tuple[pd.DataFrame, pd.DataFrame, KMeans]:
    feat = build_student_features(df)
    X = feat.drop(columns=["student_id"])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    feat["cluster"] = model.fit_predict(X_scaled)

    profile = feat.groupby("cluster", as_index=False).mean(numeric_only=True)
    profile["size"] = feat.groupby("cluster")["student_id"].size().values
    profile["size_ratio"] = profile["size"] / profile["size"].sum()
    profile["order_count_median"] = profile["order_count"].median()
    profile["cluster_name"] = profile.apply(_label_cluster, axis=1)
    profile = profile.drop(columns=["order_count_median"])

    name_map = profile.set_index("cluster")["cluster_name"].to_dict()
    feat["cluster_name"] = feat["cluster"].map(name_map)
    return feat, profile, model
