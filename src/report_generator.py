from __future__ import annotations

from datetime import datetime

import pandas as pd


def generate_markdown_report(
    kpi: dict,
    top_dishes: pd.DataFrame,
    peak_period: str,
    rule_brief: str,
    cluster_brief: str,
    predict_brief: str,
) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    dish_lines = []
    for _, row in top_dishes.head(5).iterrows():
        dish_lines.append(f"- {row['dish_name']}：销量 {row['sales_qty']:.0f} 份，评分 {row['avg_rating']:.2f}")
    dish_text = "\n".join(dish_lines) if dish_lines else "- 暂无数据"

    return f"""# 食堂经营决策报告

生成时间：{now}

## 1. 总体经营情况
- 总销售额：{kpi['total_sales']:.2f} 元
- 总订单数：{kpi['total_orders']:.0f} 单
- 匿名学生数：{kpi['total_students']:.0f} 人
- 平均客单价：{kpi['avg_ticket']:.2f} 元

## 2. 热门菜品分析
{dish_text}

## 3. 消费高峰
- 当前数据中的主要高峰时段：**{peak_period}**

## 4. 关联规则洞察
- {rule_brief}

## 5. 学生群体洞察
- {cluster_brief}

## 6. 销量预测与备餐建议
- {predict_brief}

## 7. 经营建议
- 在高峰时段提高出餐能力，减少排队时间。
- 对强关联菜品设置组合套餐，提升客单价。
- 对浪费风险较高菜品执行小批量滚动备餐。
- 针对不同消费群体提供差异化菜单与促销活动。
"""
