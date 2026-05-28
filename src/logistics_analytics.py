from __future__ import annotations

import pandas as pd


def build_satisfaction_trend(data: pd.DataFrame, feedback: pd.DataFrame) -> pd.DataFrame:
    order_scores = (
        data.groupby(["date", "canteen"], as_index=False)
        .agg(order_rating=("rating", "mean"), order_count=("order_id", "nunique"))
        .copy()
    )
    order_scores["date"] = pd.to_datetime(order_scores["date"]).astype(str)

    if feedback.empty or "create_time" not in feedback.columns:
        order_scores["feedback_score"] = order_scores["order_rating"]
        order_scores["composite_score"] = order_scores["order_rating"].round(2)
        return order_scores.sort_values(["date", "canteen"])

    fb = feedback.copy()
    score_cols = [col for col in ["taste_score", "portion_score", "price_score", "service_score"] if col in fb.columns]
    if not score_cols:
        order_scores["feedback_score"] = order_scores["order_rating"]
        order_scores["composite_score"] = order_scores["order_rating"].round(2)
        return order_scores.sort_values(["date", "canteen"])

    fb[score_cols] = fb[score_cols].apply(pd.to_numeric, errors="coerce")
    fb["feedback_score"] = fb[score_cols].mean(axis=1)
    fb["date"] = pd.to_datetime(fb["create_time"], errors="coerce").dt.date.astype(str)
    feedback_scores = (
        fb.groupby(["date", "canteen"], as_index=False)
        .agg(feedback_score=("feedback_score", "mean"), feedback_count=("feedback_score", "count"))
    )

    merged = order_scores.merge(feedback_scores, on=["date", "canteen"], how="left")
    merged["feedback_score"] = merged["feedback_score"].fillna(merged["order_rating"])
    merged["composite_score"] = (merged["order_rating"] * 0.6 + merged["feedback_score"] * 0.4).round(2)
    merged["feedback_count"] = merged["feedback_count"].fillna(0).astype(int)
    return merged.sort_values(["date", "canteen"]).reset_index(drop=True)


def build_canteen_efficiency(canteen_compare: pd.DataFrame, resource_utilization: pd.DataFrame) -> pd.DataFrame:
    merged = canteen_compare.merge(
        resource_utilization[["canteen", "utilization_index", "pred_total", "prep_total"]],
        on="canteen",
        how="left",
    ).fillna({"utilization_index": 0, "pred_total": 0, "prep_total": 0})
    merged["sales_per_student"] = (merged["sales"] / merged["students"].clip(lower=1)).round(2)
    merged["orders_per_student"] = (merged["orders"] / merged["students"].clip(lower=1)).round(2)
    merged["prep_gap"] = (merged["prep_total"] - merged["pred_total"]).round(1)
    sales_norm = merged["sales_per_student"] / max(float(merged["sales_per_student"].max()), 1.0)
    rating_norm = merged["rating"] / 5.0
    util_norm = merged["utilization_index"] / max(float(merged["utilization_index"].max()), 1.0)
    merged["efficiency_score"] = (sales_norm * 40 + rating_norm * 30 + util_norm * 30).round(1)
    return merged.sort_values("efficiency_score", ascending=False).reset_index(drop=True)


def build_risk_overview(meal_plan: pd.DataFrame) -> pd.DataFrame:
    if meal_plan.empty:
        return pd.DataFrame()
    out = meal_plan.groupby("canteen", as_index=False).agg(
        shortage_count=("risk", lambda s: int((s == "缺货风险较高").sum())),
        waste_count=("risk", lambda s: int((s == "浪费风险较高").sum())),
        pred_total=("pred_quantity", "sum"),
        prep_total=("suggested_prep", "sum"),
    )
    out["prep_ratio"] = (out["prep_total"] / out["pred_total"].clip(lower=1)).round(2)
    out["risk_level"] = "中"
    out.loc[(out["shortage_count"] >= 4) | (out["waste_count"] >= 4), "risk_level"] = "高"
    out.loc[(out["shortage_count"] <= 1) & (out["waste_count"] <= 1), "risk_level"] = "低"
    return out.sort_values(["risk_level", "shortage_count", "waste_count"], ascending=[True, False, False]).reset_index(drop=True)


def build_category_structure(data: pd.DataFrame) -> pd.DataFrame:
    category_sales = (
        data.groupby(["canteen", "category"], as_index=False)
        .agg(sales=("amount", "sum"), quantity=("quantity", "sum"))
        .sort_values(["canteen", "sales"], ascending=[True, False])
    )
    category_sales["canteen_total"] = category_sales.groupby("canteen")["sales"].transform("sum")
    category_sales["sales_ratio"] = (category_sales["sales"] / category_sales["canteen_total"].clip(lower=1)).round(4)
    return category_sales.reset_index(drop=True)


def build_decision_actions(
    efficiency: pd.DataFrame,
    risk_overview: pd.DataFrame,
    feedback_insights: dict,
    category_structure: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict] = []

    risk_map = risk_overview.set_index("canteen").to_dict("index") if not risk_overview.empty else {}
    score_map = {}
    for item in feedback_insights.get("canteenScores", []):
        score_map[item["canteen"]] = item

    for _, row in efficiency.iterrows():
        canteen = row["canteen"]
        risk = risk_map.get(canteen, {})
        score = score_map.get(canteen, {})
        top_category = (
            category_structure[category_structure["canteen"] == canteen]
            .sort_values("sales_ratio", ascending=False)
            .head(1)
        )
        top_category_name = top_category["category"].iloc[0] if not top_category.empty else "主食"

        if risk.get("shortage_count", 0) >= 4:
            rows.append(
                {
                    "canteen": canteen,
                    "priority": "高",
                    "theme": "缺货风险",
                    "advice": f"{canteen} 未来一周缺货风险较高，建议优先增加 {top_category_name} 类备餐和高峰时段人手。",
                }
            )
        if risk.get("waste_count", 0) >= 4:
            rows.append(
                {
                    "canteen": canteen,
                    "priority": "中",
                    "theme": "浪费风险",
                    "advice": f"{canteen} 备餐冗余偏高，建议减少低需求菜品预制量，并搭配限时促销。",
                }
            )
        if score and score.get("avg_score", 5) < 4.0:
            rows.append(
                {
                    "canteen": canteen,
                    "priority": "高",
                    "theme": "满意度改进",
                    "advice": f"{canteen} 满意度低于 4.0，建议排查口味稳定性、排队时长和服务体验。",
                }
            )
        if row["efficiency_score"] >= efficiency["efficiency_score"].quantile(0.75):
            rows.append(
                {
                    "canteen": canteen,
                    "priority": "低",
                    "theme": "示范推广",
                    "advice": f"{canteen} 综合效率较高，可复用其排班、备餐和热销菜品策略。",
                }
            )

    if not rows:
        rows.append(
            {
                "canteen": "全部食堂",
                "priority": "低",
                "theme": "常规维护",
                "advice": "当前整体经营平稳，建议继续跟踪高峰排队、学生评分和菜品结构变化。",
            }
        )
    priority_order = {"高": 0, "中": 1, "低": 2}
    return (
        pd.DataFrame(rows)
        .assign(priority_order=lambda df: df["priority"].map(priority_order))
        .sort_values(["priority_order", "canteen"])
        .drop(columns="priority_order")
        .reset_index(drop=True)
    )


def build_report_sections(daily_sales: pd.DataFrame, efficiency: pd.DataFrame, risk_overview: pd.DataFrame) -> dict:
    latest_date = pd.to_datetime(daily_sales["date"]).max() if not daily_sales.empty else None
    latest_day = (
        daily_sales[pd.to_datetime(daily_sales["date"]) == latest_date].iloc[0].to_dict()
        if latest_date is not None and not daily_sales.empty
        else {}
    )

    weekly = daily_sales.copy()
    if not weekly.empty:
        weekly["date"] = pd.to_datetime(weekly["date"])
        weekly["week"] = weekly["date"].dt.strftime("%Y-W%U")
        weekly_summary = (
            weekly.groupby("week", as_index=False)
            .agg(sales=("sales", "sum"), orders=("orders", "sum"))
            .sort_values("week")
            .tail(4)
        )
    else:
        weekly_summary = pd.DataFrame(columns=["week", "sales", "orders"])

    top_eff = efficiency.iloc[0].to_dict() if not efficiency.empty else {}
    high_risk = risk_overview.sort_values(["shortage_count", "waste_count"], ascending=False).head(1)
    high_risk_item = high_risk.iloc[0].to_dict() if not high_risk.empty else {}

    return {
        "daily": {
            "date": str(latest_date.date()) if latest_date is not None else "-",
            "sales": round(float(latest_day.get("sales", 0)), 2),
            "orders": int(latest_day.get("orders", 0)),
        },
        "weekly": weekly_summary.to_dict(orient="records"),
        "monthly": {
            "best_canteen": top_eff.get("canteen", "-"),
            "best_efficiency_score": top_eff.get("efficiency_score", 0),
            "highest_risk_canteen": high_risk_item.get("canteen", "-"),
            "highest_risk_level": high_risk_item.get("risk_level", "-"),
        },
    }
