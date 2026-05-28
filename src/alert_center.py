from __future__ import annotations

import pandas as pd


def _severity_rank(level: str) -> int:
    return {"高": 0, "中": 1, "低": 2}.get(level, 9)


def _add_alert(rows: list[dict], severity: str, category: str, entity: str, signal: str, change: str, advice: str) -> None:
    rows.append(
        {
            "severity": severity,
            "category": category,
            "entity": entity,
            "signal": signal,
            "change": change,
            "advice": advice,
        }
    )


def build_alert_center(data: pd.DataFrame, meal_plan: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    if data.empty:
        empty = pd.DataFrame(columns=["severity", "category", "entity", "signal", "change", "advice"])
        return empty, {"high": 0, "medium": 0, "low": 0, "total": 0}

    df = data.copy()
    df["date"] = pd.to_datetime(df["date"])
    latest_date = df["date"].max()
    recent_3 = df[df["date"] > latest_date - pd.Timedelta(days=3)]
    prev_7 = df[(df["date"] <= latest_date - pd.Timedelta(days=3)) & (df["date"] > latest_date - pd.Timedelta(days=10))]
    latest_day = df[df["date"] == latest_date]
    prev_7_day = df[(df["date"] < latest_date) & (df["date"] >= latest_date - pd.Timedelta(days=7))]

    alerts: list[dict] = []

    recent_rating = recent_3.groupby("canteen", as_index=False).agg(recent_rating=("rating", "mean"))
    prev_rating = prev_7.groupby("canteen", as_index=False).agg(prev_rating=("rating", "mean"))
    rating_compare = recent_rating.merge(prev_rating, on="canteen", how="inner")
    for _, row in rating_compare.iterrows():
      delta = float(row["recent_rating"] - row["prev_rating"])
      if delta <= -0.35:
          _add_alert(
              alerts,
              "高",
              "满意度下降",
              row["canteen"],
              "最近 3 天评分明显下滑",
              f"较此前 7 天下降 {abs(delta):.2f} 分",
              "建议立即排查热门菜品口味稳定性、窗口服务质量和排队体验。",
          )
      elif delta <= -0.2:
          _add_alert(
              alerts,
              "中",
              "满意度下降",
              row["canteen"],
              "最近 3 天评分出现波动",
              f"较此前 7 天下降 {abs(delta):.2f} 分",
              "建议观察差评来源，优先核查投诉较多的窗口和菜品。",
          )

    latest_window = latest_day.groupby(["canteen", "window", "period"], as_index=False).agg(latest_orders=("order_id", "nunique"))
    prev_window = (
        prev_7_day.groupby(["canteen", "window", "period", "date"], as_index=False)
        .agg(day_orders=("order_id", "nunique"))
        .groupby(["canteen", "window", "period"], as_index=False)
        .agg(base_orders=("day_orders", "mean"))
    )
    window_compare = latest_window.merge(prev_window, on=["canteen", "window", "period"], how="inner")
    for _, row in window_compare.iterrows():
        ratio = float(row["latest_orders"] / max(row["base_orders"], 1))
        entity = f"{row['canteen']} - {row['window']}（{row['period']}）"
        if ratio <= 0.55:
            _add_alert(
                alerts,
                "高",
                "窗口订单异常",
                entity,
                "订单量明显低于常态",
                f"仅为历史均值的 {ratio:.0%}",
                "建议检查窗口营业状态、菜品供应是否异常，必要时调整窗口导流。",
            )
        elif ratio >= 1.8:
            _add_alert(
                alerts,
                "中",
                "窗口订单异常",
                entity,
                "订单量异常偏高",
                f"达到历史均值的 {ratio:.0%}",
                "建议增加高峰人手和备餐，防止排队时间继续拉长。",
            )

    recent_dish = (
        recent_3.groupby(["dish_name", "canteen"], as_index=False)
        .agg(recent_qty=("quantity", "sum"))
        .assign(recent_daily=lambda x: x["recent_qty"] / 3)
    )
    prev_dish = (
        prev_7.groupby(["dish_name", "canteen"], as_index=False)
        .agg(prev_qty=("quantity", "sum"))
        .assign(prev_daily=lambda x: x["prev_qty"] / 7)
    )
    dish_compare = recent_dish.merge(prev_dish, on=["dish_name", "canteen"], how="inner")
    for _, row in dish_compare.iterrows():
        ratio = float(row["recent_daily"] / max(row["prev_daily"], 0.1))
        entity = f"{row['canteen']} - {row['dish_name']}"
        if ratio <= 0.55:
            _add_alert(
                alerts,
                "中",
                "菜品销量异常",
                entity,
                "销量快速下降",
                f"日均销量仅为历史的 {ratio:.0%}",
                "建议结合评分与促销活动复盘原因，判断是口味问题还是需求变化。",
            )
        elif ratio >= 1.9:
            _add_alert(
                alerts,
                "中",
                "菜品销量异常",
                entity,
                "销量快速上升",
                f"日均销量达到历史的 {ratio:.0%}",
                "建议适度追加备餐，避免热销菜品在高峰时段断供。",
            )

    if not meal_plan.empty:
        risk_counts = meal_plan.groupby("canteen", as_index=False).agg(
            shortage_count=("risk", lambda s: int((s == "缺货风险较高").sum())),
            waste_count=("risk", lambda s: int((s == "浪费风险较高").sum())),
        )
        for _, row in risk_counts.iterrows():
            if row["shortage_count"] >= 4:
                _add_alert(
                    alerts,
                    "高",
                    "备餐风险异常",
                    row["canteen"],
                    "缺货风险菜品过多",
                    f"共有 {int(row['shortage_count'])} 个菜品处于高缺货风险",
                    "建议立即提高高峰热门菜品安全备餐系数，并优先调配备餐人手。",
                )
            if row["waste_count"] >= 4:
                _add_alert(
                    alerts,
                    "中",
                    "备餐风险异常",
                    row["canteen"],
                    "浪费风险菜品过多",
                    f"共有 {int(row['waste_count'])} 个菜品处于高浪费风险",
                    "建议减少低需求菜品预制量，并测试限时优惠和套餐出清策略。",
                )

    latest_peak = latest_day.groupby(["canteen", "period"], as_index=False).agg(latest_orders=("order_id", "nunique"))
    hist_peak = (
        prev_7_day.groupby(["canteen", "period", "date"], as_index=False)
        .agg(day_orders=("order_id", "nunique"))
        .groupby(["canteen", "period"], as_index=False)
        .agg(base_orders=("day_orders", "mean"))
    )
    peak_compare = latest_peak.merge(hist_peak, on=["canteen", "period"], how="inner")
    for _, row in peak_compare.iterrows():
        ratio = float(row["latest_orders"] / max(row["base_orders"], 1))
        if ratio >= 1.45:
            _add_alert(
                alerts,
                "高" if ratio >= 1.7 else "中",
                "拥挤度异常",
                f"{row['canteen']}（{row['period']}）",
                "当前高峰压力超出常态",
                f"订单量达到历史均值的 {ratio:.0%}",
                "建议启动错峰引导、开放备用窗口或临时增加服务人员。",
            )

    alert_df = pd.DataFrame(alerts)
    if alert_df.empty:
        alert_df = pd.DataFrame(
            [
                {
                    "severity": "低",
                    "category": "运行平稳",
                    "entity": "全部食堂",
                    "signal": "暂未发现显著异常",
                    "change": "整体处于正常波动范围内",
                    "advice": "继续跟踪评分、销量和备餐风险，保持日常巡检。",
                }
            ]
        )

    alert_df = (
        alert_df.assign(sort_key=lambda x: x["severity"].map(_severity_rank))
        .sort_values(["sort_key", "category", "entity"])
        .drop(columns="sort_key")
        .reset_index(drop=True)
    )

    summary = {
        "high": int((alert_df["severity"] == "高").sum()),
        "medium": int((alert_df["severity"] == "中").sum()),
        "low": int((alert_df["severity"] == "低").sum()),
        "total": int(len(alert_df)),
    }
    return alert_df, summary
