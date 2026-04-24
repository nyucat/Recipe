from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.descriptive_analysis import add_time_features


@dataclass(frozen=True)
class DishProfile:
    dish_name: str
    taste: str
    goal_tags: tuple[str, ...]
    heat_level: str
    protein_level: str
    fat_level: str
    healthy_tag: str


DISH_PROFILES = [
    DishProfile("包子", "咸", ("省钱", "吃饱"), "中", "低", "低", "清淡"),
    DishProfile("豆浆", "甜", ("减脂", "省钱"), "低", "中", "低", "清淡"),
    DishProfile("鸡蛋", "咸", ("高蛋白", "省钱"), "低", "中", "低", "均衡"),
    DishProfile("热干面", "咸", ("吃饱",), "中", "中", "中", "均衡"),
    DishProfile("牛肉面", "咸", ("高蛋白", "吃饱"), "中", "高", "中", "均衡"),
    DishProfile("黄焖鸡米饭", "咸", ("吃饱", "高蛋白"), "高", "高", "中", "均衡"),
    DishProfile("麻辣香锅", "辣", ("尝鲜", "吃饱"), "高", "中", "高", "重口"),
    DishProfile("番茄炒蛋", "咸", ("均衡饮食",), "中", "中", "中", "均衡"),
    DishProfile("红烧肉", "咸", ("吃饱",), "高", "中", "高", "重口"),
    DishProfile("青椒土豆丝", "辣", ("省钱", "减脂"), "低", "低", "低", "清淡"),
    DishProfile("紫菜蛋花汤", "清淡", ("减脂",), "低", "低", "低", "清淡"),
    DishProfile("绿豆汤", "甜", ("减脂", "低糖"), "低", "低", "低", "清淡"),
    DishProfile("奶茶", "甜", ("尝鲜",), "高", "低", "中", "高糖"),
    DishProfile("可乐", "甜", ("尝鲜",), "高", "低", "低", "高糖"),
    DishProfile("炸鸡排", "咸", ("吃饱", "高蛋白"), "高", "中", "高", "油炸"),
    DishProfile("炒饭", "咸", ("吃饱",), "中", "中", "中", "均衡"),
    DishProfile("炒面", "咸", ("吃饱",), "中", "中", "中", "均衡"),
    DishProfile("水果沙拉", "清淡", ("减脂", "低糖"), "低", "低", "低", "健康"),
]


def get_dish_profile_df() -> pd.DataFrame:
    rows = []
    for p in DISH_PROFILES:
        rows.append(
            {
                "dish_name": p.dish_name,
                "taste": p.taste,
                "goal_tags": list(p.goal_tags),
                "heat_level": p.heat_level,
                "protein_level": p.protein_level,
                "fat_level": p.fat_level,
                "healthy_tag": p.healthy_tag,
            }
        )
    return pd.DataFrame(rows)


def budget_to_range(budget: str) -> tuple[float, float]:
    mapping = {
        "10元以内": (0, 10),
        "10-15元": (10, 15),
        "15-20元": (15, 20),
        "不限": (0, 999),
    }
    return mapping[budget]


def build_today_recommendation(
    df: pd.DataFrame,
    student_id: str,
    budget: str,
    taste: str,
    goal: str,
    period: str,
    canteen_pref: str,
    disliked_dishes: set[str] | None = None,
    top_n: int = 5,
) -> pd.DataFrame:
    disliked_dishes = disliked_dishes or set()
    data = add_time_features(df)
    dish_base = (
        data.groupby(["dish_name", "canteen", "window", "category"], as_index=False)
        .agg(
            avg_price=("price", "mean"),
            avg_rating=("rating", "mean"),
            sales_qty=("quantity", "sum"),
            period_ratio=("period", lambda s: float((s == period).mean())),
        )
        .copy()
    )
    profile = get_dish_profile_df()
    dish_base = dish_base.merge(profile, on="dish_name", how="left")

    pmin, pmax = budget_to_range(budget)
    cand = dish_base[(dish_base["avg_price"] >= pmin) & (dish_base["avg_price"] <= pmax)].copy()
    if taste != "不限":
        cand = cand[cand["taste"] == taste]
    if canteen_pref != "不限":
        cand = cand[cand["canteen"] == canteen_pref]

    cand = cand[~cand["dish_name"].isin(disliked_dishes)]

    student_hist = data[data["student_id"] == student_id]
    fav_category = None
    if not student_hist.empty:
        fav_category = student_hist["category"].value_counts().index[0]

    max_sales = max(float(cand["sales_qty"].max()), 1.0) if not cand.empty else 1.0
    cand["score"] = 0.0
    cand["score"] += (cand["avg_rating"] / 5.0) * 0.25
    cand["score"] += (cand["sales_qty"] / max_sales) * 0.2
    cand["score"] += cand["period_ratio"] * 0.15
    cand["score"] += cand["goal_tags"].apply(lambda tags: 1.0 if isinstance(tags, list) and goal in tags else 0.0) * 0.25
    if fav_category is not None:
        cand["score"] += (cand["category"] == fav_category).astype(float) * 0.15

    cand = cand.sort_values("score", ascending=False).head(top_n).reset_index(drop=True)
    if cand.empty:
        return cand

    def reason(row: pd.Series) -> str:
        parts = [
            f"评分 {row['avg_rating']:.1f}",
            f"均价 {row['avg_price']:.1f} 元",
            f"{period}时段适配度较高",
        ]
        if fav_category and row["category"] == fav_category:
            parts.append("符合你的历史偏好")
        if isinstance(row["goal_tags"], list) and goal in row["goal_tags"]:
            parts.append(f"符合“{goal}”目标")
        return "；".join(parts)

    cand["reason"] = cand.apply(reason, axis=1)
    return cand


def crowding_prediction(df: pd.DataFrame, now: datetime | None = None) -> pd.DataFrame:
    now = now or datetime.now()
    data = df.copy()
    data["order_time"] = pd.to_datetime(data["order_time"], errors="coerce")
    data = data.dropna(subset=["order_time"])
    data["weekday"] = data["order_time"].dt.weekday
    data["slot"] = data["order_time"].dt.floor("15min")
    data["slot_time"] = data["slot"].dt.time
    data["date"] = data["order_time"].dt.date

    slot_stat = (
        data.groupby(["canteen", "weekday", "slot_time", "date"], as_index=False)
        .agg(order_cnt=("order_id", "nunique"))
        .copy()
    )
    hist = (
        slot_stat.groupby(["canteen", "weekday", "slot_time"], as_index=False)
        .agg(hist_mean=("order_cnt", "mean"), hist_std=("order_cnt", "std"))
        .fillna(0)
    )

    cur_slot = pd.Timestamp(now).floor("15min").time()
    wd = pd.Timestamp(now).weekday()
    current = hist[(hist["weekday"] == wd) & (hist["slot_time"] == cur_slot)].copy()
    if current.empty:
        current = hist.groupby("canteen", as_index=False).agg(hist_mean=("hist_mean", "mean"), hist_std=("hist_std", "mean"))
    current["pred_order"] = (current["hist_mean"] + 0.4 * current["hist_std"]).round().astype(int)
    current["ratio"] = current["pred_order"] / current["hist_mean"].replace(0, 1)

    current["level"] = "低"
    current.loc[current["ratio"] >= 1.0, "level"] = "中"
    current.loc[current["ratio"] >= 1.3, "level"] = "高"
    current.loc[current["ratio"] >= 1.6, "level"] = "爆满"

    queue_map = {"低": "3-5 分钟", "中": "6-10 分钟", "高": "12-18 分钟", "爆满": "20+ 分钟"}
    current["queue_time"] = current["level"].map(queue_map)
    return current.sort_values("pred_order", ascending=False).reset_index(drop=True)


def student_budget_report(df: pd.DataFrame, student_id: str, month_budget: float) -> dict:
    data = df.copy()
    data["order_time"] = pd.to_datetime(data["order_time"], errors="coerce")
    data = data.dropna(subset=["order_time"])
    cur_month = pd.Timestamp.now().to_period("M")
    month_df = data[data["order_time"].dt.to_period("M") == cur_month].copy()
    stu = month_df[month_df["student_id"] == student_id]
    if stu.empty:
        return {
            "spent": 0.0,
            "remaining": month_budget,
            "days_passed": int(pd.Timestamp.now().day),
            "days_total": int(pd.Timestamp.now().days_in_month),
            "suggest_daily": month_budget / max(int(pd.Timestamp.now().days_in_month), 1),
            "forecast_month_end": 0.0,
            "overspend": 0.0,
        }

    daily = stu.groupby(stu["order_time"].dt.date, as_index=False).agg(day_amount=("amount", "sum"))
    spent = float(stu["amount"].sum())
    days_passed = int(pd.Timestamp.now().day)
    days_total = int(pd.Timestamp.now().days_in_month)
    forecast = spent / max(days_passed, 1) * days_total
    return {
        "spent": spent,
        "remaining": month_budget - spent,
        "days_passed": days_passed,
        "days_total": days_total,
        "suggest_daily": max(month_budget - spent, 0) / max(days_total - days_passed, 1),
        "forecast_month_end": forecast,
        "overspend": max(forecast - month_budget, 0),
        "daily_df": daily,
    }


def nutrition_structure(df: pd.DataFrame, student_id: str, recent_days: int = 7) -> pd.DataFrame:
    data = df.copy()
    data["order_time"] = pd.to_datetime(data["order_time"], errors="coerce")
    data = data.dropna(subset=["order_time"])
    end_date = data["order_time"].max()
    start_date = end_date - pd.Timedelta(days=recent_days)
    part = data[(data["student_id"] == student_id) & (data["order_time"] >= start_date)].copy()
    if part.empty:
        return pd.DataFrame()
    profile = get_dish_profile_df()
    part = part.merge(profile[["dish_name", "healthy_tag"]], on="dish_name", how="left")
    out = part.groupby("healthy_tag", as_index=False).agg(quantity=("quantity", "sum"))
    out["ratio"] = out["quantity"] / out["quantity"].sum()
    return out.sort_values("ratio", ascending=False)


def suggest_promotions(df: pd.DataFrame) -> pd.DataFrame:
    data = add_time_features(df)
    by_dish_day = (
        data.groupby(["dish_name", "date"], as_index=False)
        .agg(sales_qty=("quantity", "sum"), avg_rating=("rating", "mean"))
        .copy()
    )
    recent_date = by_dish_day["date"].max()
    recent = by_dish_day[by_dish_day["date"] > (recent_date - pd.Timedelta(days=7))]
    previous = by_dish_day[(by_dish_day["date"] <= recent_date - pd.Timedelta(days=7)) & (by_dish_day["date"] > (recent_date - pd.Timedelta(days=14)))]
    r = recent.groupby("dish_name", as_index=False).agg(recent_qty=("sales_qty", "sum"), recent_rating=("avg_rating", "mean"))
    p = previous.groupby("dish_name", as_index=False).agg(prev_qty=("sales_qty", "sum"))
    merged = r.merge(p, on="dish_name", how="left").fillna({"prev_qty": 0})
    merged["decline_ratio"] = np.where(merged["prev_qty"] > 0, (merged["recent_qty"] - merged["prev_qty"]) / merged["prev_qty"], 0)
    merged["promo_type"] = "正常"
    merged.loc[merged["decline_ratio"] <= -0.2, "promo_type"] = "建议促销"
    merged.loc[(merged["promo_type"] == "正常") & (merged["recent_rating"] >= 4.4), "promo_type"] = "建议加曝光"
    merged["advice"] = merged["promo_type"].map(
        {
            "建议促销": "销量下降明显，可做限时折扣或套餐。",
            "建议加曝光": "评分高但潜在曝光不足，可置顶推荐。",
            "正常": "维持当前策略。",
        }
    )
    return merged.sort_values(["promo_type", "decline_ratio"], ascending=[False, True]).reset_index(drop=True)


def quality_score_report(df: pd.DataFrame) -> dict:
    check = df.copy()
    n = len(check)
    if n == 0:
        return {"score": 0, "items": ["数据为空"]}
    missing_ratio = float(check.isna().sum().sum() / max(check.shape[0] * check.shape[1], 1))
    duplicate_ratio = float(check.duplicated().mean())
    price = pd.to_numeric(check.get("price", pd.Series(dtype=float)), errors="coerce")
    qty = pd.to_numeric(check.get("quantity", pd.Series(dtype=float)), errors="coerce")
    invalid_price_ratio = float(((price <= 0) | (price > 80) | price.isna()).mean())
    invalid_qty_ratio = float(((qty <= 0) | (qty > 10) | qty.isna()).mean())
    dish_norm = check.get("dish_name", pd.Series(dtype=str)).astype(str).str.replace(r"\s+", "", regex=True).str.lower()
    inconsistency_ratio = float((dish_norm != check.get("dish_name", pd.Series(dtype=str)).astype(str).str.lower()).mean())
    time_recent_ratio = float(
        (pd.to_datetime(check.get("order_time", pd.Series(dtype=str)), errors="coerce") >= (pd.Timestamp.now() - pd.Timedelta(days=180))).mean()
    )

    score = 100
    score -= int(missing_ratio * 100 * 0.35)
    score -= int(duplicate_ratio * 100 * 0.2)
    score -= int(invalid_price_ratio * 100 * 0.2)
    score -= int(invalid_qty_ratio * 100 * 0.15)
    score -= int(inconsistency_ratio * 100 * 0.05)
    score += int(time_recent_ratio * 10)
    score = max(0, min(score, 100))

    items = []
    items.append(f"缺失值比例：{missing_ratio:.2%}")
    items.append(f"重复记录比例：{duplicate_ratio:.2%}")
    items.append(f"价格异常比例：{invalid_price_ratio:.2%}")
    items.append(f"数量异常比例：{invalid_qty_ratio:.2%}")
    items.append(f"近180天数据占比：{time_recent_ratio:.2%}")
    return {"score": score, "items": items}


def load_table(path: Path, columns: list[str]) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame(columns=columns)


def save_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
