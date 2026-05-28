from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.association_rules import mine_association_rules
from src.data_preprocess import clean_orders, generate_sample_orders
from src.descriptive_analysis import add_time_features, compute_kpis
from src.recommender import recommend_hot_dishes
from src.sales_predict import build_meal_plan, forecast_next_days, train_sales_model
from src.smart_services import (
    build_today_recommendation,
    crowding_prediction,
    nutrition_structure,
    quality_score_report,
    student_budget_report,
    suggest_promotions,
)
from src.user_cluster import train_user_clusters


PROCESSED_PATH = ROOT / "data" / "processed" / "cleaned_orders.csv"
FEEDBACK_PATH = ROOT / "data" / "processed" / "dish_feedback.csv"
PREFERENCES_PATH = ROOT / "data" / "processed" / "student_preferences.csv"
VOTES_PATH = ROOT / "data" / "processed" / "dish_votes.csv"
ANNOUNCEMENTS_PATH = ROOT / "data" / "processed" / "announcements.csv"
FRONTEND_DATA_PATH = ROOT / "frontend" / "public" / "dashboard-data.json"


def _read_or_bootstrap_orders() -> pd.DataFrame:
    if PROCESSED_PATH.exists():
        return pd.read_csv(PROCESSED_PATH)
    sample = clean_orders(generate_sample_orders())
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(PROCESSED_PATH, index=False, encoding="utf-8-sig")
    return sample


def _load_optional_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def _json_ready_records(df: pd.DataFrame, limit: int | None = None) -> list[dict]:
    if df is None or df.empty:
        return []
    out = df.copy()
    for col in out.columns:
        if str(out[col].dtype).startswith("datetime"):
            out[col] = out[col].astype(str)
    if limit is not None:
        out = out.head(limit)
    return json.loads(out.to_json(orient="records", force_ascii=False))


def build_dashboard_payload() -> dict:
    orders = _read_or_bootstrap_orders()
    orders["order_time"] = pd.to_datetime(orders["order_time"], errors="coerce")
    data = add_time_features(orders)

    student_ids = sorted(data["student_id"].astype(str).unique().tolist())
    demo_student = student_ids[0] if student_ids else ""

    kpis = compute_kpis(orders)
    quality = quality_score_report(orders)
    promotions = suggest_promotions(orders)
    hot_dishes = recommend_hot_dishes(orders, top_n=10)
    rules = mine_association_rules(orders, min_support=0.008, min_confidence=0.2, min_lift=1.05)
    crowding = crowding_prediction(orders)
    student_features, cluster_profile, _ = train_user_clusters(orders, n_clusters=4)
    model_pack = train_sales_model(orders)
    future = forecast_next_days(model_pack, days=7)
    meal_plan = build_meal_plan(future)

    daily_sales = (
        data.groupby("date", as_index=False)
        .agg(sales=("amount", "sum"), orders=("order_id", "nunique"))
        .sort_values("date")
    )
    canteen_compare = (
        data.groupby("canteen", as_index=False)
        .agg(sales=("amount", "sum"), orders=("order_id", "nunique"), students=("student_id", "nunique"), rating=("rating", "mean"))
        .sort_values("sales", ascending=False)
    )
    dish_sales = (
        data.groupby("dish_name", as_index=False)
        .agg(sales_qty=("quantity", "sum"), sales_amount=("amount", "sum"), avg_rating=("rating", "mean"), avg_price=("price", "mean"))
        .sort_values("sales_qty", ascending=False)
    )
    period_stat = (
        data.groupby("period", as_index=False)
        .agg(orders=("order_id", "nunique"), sales=("amount", "sum"))
    )
    heat = (
        data.groupby(["weekday_cn", "hour"], as_index=False)
        .agg(orders=("order_id", "nunique"))
        .sort_values(["weekday_cn", "hour"])
    )
    half_hour = (
        data.assign(half_hour=pd.to_datetime(data["order_time"]).dt.floor("30min").dt.strftime("%H:%M"))
        .groupby("half_hour", as_index=False)
        .agg(orders=("order_id", "nunique"))
        .sort_values("half_hour")
    )
    student_report = student_budget_report(orders, demo_student, 600.0)
    nutrition = nutrition_structure(orders, demo_student, recent_days=7)
    today_rec = build_today_recommendation(orders, demo_student, "10-15元", "不限", "吃饱", "午餐", "不限", top_n=6)

    feedback = _load_optional_csv(FEEDBACK_PATH)
    preferences = _load_optional_csv(PREFERENCES_PATH)
    votes = _load_optional_csv(VOTES_PATH)
    announcements = _load_optional_csv(ANNOUNCEMENTS_PATH)

    return {
        "meta": {
            "generated_at": pd.Timestamp.now().isoformat(),
            "student_ids": student_ids[:200],
            "demo_student": demo_student,
            "canteens": sorted(data["canteen"].dropna().unique().tolist()),
            "periods": ["早餐", "午餐", "晚餐", "夜宵"],
        },
        "summary": {
            "kpis": kpis,
            "quality": quality,
        },
        "student": {
            "todayRecommendation": _json_ready_records(today_rec),
            "budgetReport": {
                k: (v if not isinstance(v, pd.DataFrame) else _json_ready_records(v))
                for k, v in student_report.items()
            },
            "nutrition": _json_ready_records(nutrition),
            "crowding": _json_ready_records(crowding),
            "preferences": _json_ready_records(preferences),
            "feedback": _json_ready_records(feedback),
            "votes": _json_ready_records(votes),
            "announcements": _json_ready_records(announcements),
        },
        "management": {
            "dailySales": _json_ready_records(daily_sales),
            "canteenCompare": _json_ready_records(canteen_compare),
            "dishSales": _json_ready_records(dish_sales, limit=30),
            "hotDishes": _json_ready_records(hot_dishes),
            "promotions": _json_ready_records(promotions, limit=20),
            "periodStat": _json_ready_records(period_stat),
            "heatmap": _json_ready_records(heat),
            "halfHourOrders": _json_ready_records(half_hour),
            "rules": _json_ready_records(rules, limit=60),
            "predictionMetrics": model_pack["metrics"],
            "predictionEval": _json_ready_records(model_pack["eval_df"], limit=120),
            "futurePrediction": _json_ready_records(future, limit=80),
            "mealPlan": _json_ready_records(meal_plan, limit=80),
        },
        "logistics": {
            "canteenCompare": _json_ready_records(canteen_compare),
            "clusterProfile": _json_ready_records(cluster_profile),
            "studentFeatures": _json_ready_records(student_features),
            "mealPlan": _json_ready_records(meal_plan, limit=80),
        },
        "raw": {
            "orders": _json_ready_records(data),
        },
    }


def export_dashboard_data() -> Path:
    FRONTEND_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = build_dashboard_payload()
    FRONTEND_DATA_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return FRONTEND_DATA_PATH


if __name__ == "__main__":
    path = export_dashboard_data()
    print(path)
