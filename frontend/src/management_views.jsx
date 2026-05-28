import React from "react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { HeatGrid, MetricCard, MiniTable, Panel } from "./components";
import { PIE_COLORS } from "./constants";
import { formatNumber } from "./utils";

function KeywordList({ title, items, tone = "positive" }) {
  return (
    <div className="keyword-box">
      <h4>{title}</h4>
      <div className="keyword-list">
        {items.length ? (
          items.map((item) => (
            <span className={`keyword-chip ${tone}`} key={item.keyword}>
              {item.keyword} · {item.count}
            </span>
          ))
        ) : (
          <span className="keyword-chip neutral">暂无数据</span>
        )}
      </div>
    </div>
  );
}

export function ManagementDashboardView({ data }) {
  const commentAnalysis = data.management.commentAnalysis || {};

  return (
    <>
      <div className="metric-grid">
        <MetricCard label="总销售额" value={`${formatNumber(data.summary.kpis.total_sales)} 元`} tone="warm" />
        <MetricCard label="订单总数" value={`${formatNumber(data.summary.kpis.total_orders, 0)} 单`} tone="teal" />
        <MetricCard label="消费学生数" value={`${formatNumber(data.summary.kpis.total_students, 0)} 人`} tone="blue" />
        <MetricCard label="数据质量评分" value={`${data.summary.quality.score}/100`} tone="berry" />
      </div>

      <div className="two-col">
        <Panel title="销售趋势">
          <ResponsiveContainer width="100%" height={280}>
            <AreaChart data={data.management.dailySales}>
              <CartesianGrid strokeDasharray="3 3" stroke="#ffffff18" />
              <XAxis dataKey="date" stroke="#d6d0f4" />
              <YAxis stroke="#d6d0f4" />
              <Tooltip />
              <Legend />
              <Area type="monotone" dataKey="sales" fill="#14b8a655" stroke="#14b8a6" />
              <Area type="monotone" dataKey="orders" fill="#0ea5e955" stroke="#0ea5e9" />
            </AreaChart>
          </ResponsiveContainer>
        </Panel>
        <Panel title="多食堂经营对比">
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={data.management.canteenCompare}>
              <CartesianGrid strokeDasharray="3 3" stroke="#ffffff18" />
              <XAxis dataKey="canteen" stroke="#d6d0f4" />
              <YAxis stroke="#d6d0f4" />
              <Tooltip />
              <Legend />
              <Bar dataKey="sales" fill="#f97316" radius={[10, 10, 0, 0]} />
              <Bar dataKey="orders" fill="#8b5cf6" radius={[10, 10, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Panel>
      </div>

      <Panel title="促销建议">
        <MiniTable
          columns={[
            { key: "dish_name", label: "菜品" },
            { key: "recent_qty", label: "近 7 天销量" },
            { key: "decline_ratio", label: "变化率", render: (value) => `${formatNumber(value * 100)}%` },
            { key: "promo_type", label: "建议类型" },
            { key: "advice", label: "建议" },
          ]}
          rows={data.management.promotions}
        />
      </Panel>

      <Panel title="促销效果评估" subtitle="对建议促销菜品进行前后销量、销售额和评分变化复盘">
        <MiniTable
          columns={[
            { key: "dish_name", label: "菜品" },
            { key: "promo_type", label: "促销类型" },
            { key: "qty_change", label: "销量变化", render: (value) => `${formatNumber(value * 100)}%` },
            { key: "sales_change", label: "销售额变化", render: (value) => `${formatNumber(value * 100)}%` },
            { key: "rating_change", label: "评分变化", render: (value) => `${value >= 0 ? "+" : ""}${formatNumber(value, 2)}` },
            { key: "outcome", label: "评估结果" },
            { key: "review", label: "复盘结论" },
          ]}
          rows={data.management.promoEvaluation || []}
        />
      </Panel>

      <div className="two-col">
        <Panel title="数据质量问题摘要">
          <div className="bullet-stack">
            {data.summary.quality.items.map((item, index) => (
              <div className="bullet-row" key={`${item}-${index}`}>
                {item}
              </div>
            ))}
          </div>
        </Panel>
        <Panel title="服务反馈概览">
          <MiniTable
            columns={[
              { key: "dish_name", label: "菜品" },
              { key: "avg_score", label: "平均分", render: (value) => formatNumber(value, 1) },
              { key: "feedbacks", label: "反馈数" },
            ]}
            rows={data.management.feedbackInsights?.dishScores?.slice(0, 6) || []}
          />
        </Panel>
      </div>

      <Panel title="评论关键词分析" subtitle={`当前纳入分析的评论样本数：${commentAnalysis.commentCount || 0}`}>
        <div className="two-col">
          <KeywordList title="高频好评词" items={commentAnalysis.positiveKeywords || []} tone="positive" />
          <KeywordList title="高频差评词" items={commentAnalysis.negativeKeywords || []} tone="negative" />
        </div>
      </Panel>

      <div className="two-col">
        <Panel title="问题主题归因">
          <MiniTable
            columns={[
              { key: "theme", label: "主题" },
              { key: "count", label: "出现次数" },
            ]}
            rows={commentAnalysis.themeCounts || []}
          />
        </Panel>
        <Panel title="重点问题菜品">
          <MiniTable
            columns={[
              { key: "dish_name", label: "菜品" },
              { key: "canteen", label: "食堂" },
              { key: "issue_count", label: "负面反馈数" },
              { key: "avg_score", label: "平均分", render: (value) => formatNumber(value, 2) },
            ]}
            rows={commentAnalysis.dishIssues || []}
          />
        </Panel>
      </div>

      <div className="two-col">
        <Panel title="好评样本">
          <MiniTable
            columns={[
              { key: "dish_name", label: "菜品" },
              { key: "canteen", label: "食堂" },
              { key: "comment", label: "评论内容" },
            ]}
            rows={commentAnalysis.samplePositive || []}
          />
        </Panel>
        <Panel title="差评样本">
          <MiniTable
            columns={[
              { key: "dish_name", label: "菜品" },
              { key: "canteen", label: "食堂" },
              { key: "comment", label: "评论内容" },
            ]}
            rows={commentAnalysis.sampleNegative || []}
          />
        </Panel>
      </div>
    </>
  );
}

export function ManagementSalesView({ data }) {
  return (
    <>
      <div className="two-col">
        <Panel title="热门菜品 Top 10">
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={data.management.hotDishes}>
              <CartesianGrid strokeDasharray="3 3" stroke="#ffffff18" />
              <XAxis dataKey="dish_name" stroke="#d6d0f4" angle={-18} textAnchor="end" height={80} />
              <YAxis stroke="#d6d0f4" />
              <Tooltip />
              <Bar dataKey="sales_qty" fill="#fb923c" radius={[8, 8, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Panel>
        <Panel title="价格与销量关系">
          <ResponsiveContainer width="100%" height={320}>
            <ScatterChart>
              <CartesianGrid strokeDasharray="3 3" stroke="#ffffff18" />
              <XAxis type="number" dataKey="avg_price" name="均价" stroke="#d6d0f4" />
              <YAxis type="number" dataKey="sales_qty" name="销量" stroke="#d6d0f4" />
              <Tooltip cursor={{ strokeDasharray: "3 3" }} />
              <Scatter data={data.management.dishSales} fill="#14b8a6" />
            </ScatterChart>
          </ResponsiveContainer>
        </Panel>
      </div>

      <Panel title="菜品销售明细">
        <MiniTable
          columns={[
            { key: "dish_name", label: "菜品" },
            { key: "sales_qty", label: "销量" },
            { key: "sales_amount", label: "销售额", render: (value) => `${formatNumber(value)} 元` },
            { key: "avg_rating", label: "评分", render: (value) => formatNumber(value, 1) },
            { key: "avg_price", label: "均价", render: (value) => `${formatNumber(value)} 元` },
          ]}
          rows={data.management.dishSales}
        />
      </Panel>

      <Panel title="菜品生命周期分析">
        <MiniTable
          columns={[
            { key: "dish_name", label: "菜品" },
            { key: "stage", label: "阶段" },
            { key: "recent_qty", label: "近 7 天销量" },
            { key: "growth", label: "增长率", render: (value) => `${formatNumber(value * 100)}%` },
            { key: "advice", label: "建议" },
          ]}
          rows={data.management.lifecycle || []}
        />
      </Panel>
    </>
  );
}

export function ManagementPeakView({ data }) {
  return (
    <>
      <div className="two-col">
        <Panel title="时段订单分布">
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={data.management.periodStat}>
              <CartesianGrid strokeDasharray="3 3" stroke="#ffffff18" />
              <XAxis dataKey="period" stroke="#d6d0f4" />
              <YAxis stroke="#d6d0f4" />
              <Tooltip />
              <Legend />
              <Bar dataKey="orders" fill="#0ea5e9" radius={[8, 8, 0, 0]} />
              <Bar dataKey="sales" fill="#f97316" radius={[8, 8, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Panel>
        <Panel title="30 分钟粒度高峰">
          <ResponsiveContainer width="100%" height={280}>
            <LineChart data={data.management.halfHourOrders}>
              <CartesianGrid strokeDasharray="3 3" stroke="#ffffff18" />
              <XAxis dataKey="half_hour" stroke="#d6d0f4" angle={-25} textAnchor="end" height={70} />
              <YAxis stroke="#d6d0f4" />
              <Tooltip />
              <Line type="monotone" dataKey="orders" stroke="#8b5cf6" strokeWidth={3} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </Panel>
      </div>
      <Panel title="星期 - 小时热力图">
        <HeatGrid data={data.management.heatmap} />
      </Panel>
      <Panel title="窗口压力排行">
        <MiniTable
          columns={[
            { key: "window", label: "窗口" },
            { key: "period", label: "高峰时段" },
            { key: "orders", label: "订单量" },
            { key: "sales", label: "销售额", render: (value) => `${formatNumber(value)} 元` },
          ]}
          rows={data.management.windowPressure || []}
        />
      </Panel>
    </>
  );
}

export function ManagementRulesView({ data }) {
  return (
    <>
      <Panel title="强关联搭配组合" subtitle="适合用于套餐设计、联动销售和窗口推荐">
        <div className="rule-grid">
          {data.management.rules.slice(0, 8).map((item, index) => (
            <article className="rule-card" key={`${item.antecedents_str}-${index}`}>
              <span>{item.antecedents_str}</span>
              <strong>{item.consequents_str}</strong>
              <div className="rule-metrics">
                <em>支持度 {formatNumber(item.support * 100)}%</em>
                <em>置信度 {formatNumber(item.confidence * 100)}%</em>
                <em>提升度 {formatNumber(item.lift, 2)}</em>
              </div>
            </article>
          ))}
        </div>
      </Panel>
      <Panel title="关联规则明细">
        <MiniTable
          columns={[
            { key: "antecedents_str", label: "前项" },
            { key: "consequents_str", label: "后项" },
            { key: "support", label: "支持度", render: (value) => `${formatNumber(value * 100)}%` },
            { key: "confidence", label: "置信度", render: (value) => `${formatNumber(value * 100)}%` },
            { key: "lift", label: "提升度", render: (value) => formatNumber(value, 2) },
          ]}
          rows={data.management.rules}
        />
      </Panel>
      <Panel title="可落地套餐建议">
        <MiniTable
          columns={[
            { key: "combo_name", label: "套餐组合" },
            { key: "estimated_price", label: "估算价格", render: (value) => `${formatNumber(value)} 元` },
            { key: "confidence", label: "置信度", render: (value) => `${formatNumber(value * 100)}%` },
            { key: "reason", label: "建议理由" },
          ]}
          rows={data.management.comboSuggestions || []}
        />
      </Panel>
    </>
  );
}

export function ManagementPredictView({ data, predictionMomentum }) {
  const explain = data.management.predictionExplain || {};
  const grouped = explain.groupedImportance || [];
  const topFeatures = explain.topFeatures || [];
  const insights = explain.insights || [];

  return (
    <>
      <div className="metric-grid">
        <MetricCard label="MAE" value={formatNumber(data.management.predictionMetrics.MAE, 3)} tone="warm" />
        <MetricCard label="RMSE" value={formatNumber(data.management.predictionMetrics.RMSE, 3)} tone="teal" />
        <MetricCard label="R²" value={formatNumber(data.management.predictionMetrics.R2, 3)} tone="blue" />
        <MetricCard label="预测周期" value="未来 7 天" tone="berry" />
      </div>

      <div className="two-col">
        <Panel title="测试集真实值 vs 预测值">
          <ResponsiveContainer width="100%" height={320}>
            <LineChart data={data.management.predictionEval}>
              <CartesianGrid strokeDasharray="3 3" stroke="#ffffff18" />
              <XAxis dataKey="date" stroke="#d6d0f4" />
              <YAxis stroke="#d6d0f4" />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="quantity" name="真实值" stroke="#14b8a6" dot={false} />
              <Line type="monotone" dataKey="pred" name="预测值" stroke="#f97316" dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </Panel>
        <Panel title="未来预测清单">
          <MiniTable
            columns={[
              { key: "date", label: "日期" },
              { key: "dish_name", label: "菜品" },
              { key: "pred_quantity", label: "预测销量" },
              { key: "window", label: "窗口" },
            ]}
            rows={data.management.futurePrediction}
          />
        </Panel>
      </div>

      <Panel title="特征重要性解释" subtitle="解释模型为什么会做出当前预测结果">
        <div className="two-col">
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={grouped}>
              <CartesianGrid strokeDasharray="3 3" stroke="#ffffff18" />
              <XAxis dataKey="feature_group" stroke="#d6d0f4" angle={-12} textAnchor="end" height={70} />
              <YAxis stroke="#d6d0f4" />
              <Tooltip formatter={(value) => `${formatNumber(value * 100, 1)}%`} />
              <Bar dataKey="importance_pct" fill="#38bdf8" radius={[8, 8, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
          <div className="bullet-stack">
            {insights.map((item, index) => (
              <div className="bullet-row" key={`${index}-${item}`}>
                {item}
              </div>
            ))}
          </div>
        </div>
      </Panel>

      <div className="two-col">
        <Panel title="最重要的具体特征">
          <MiniTable
            columns={[
              { key: "feature_label", label: "特征" },
              { key: "importance", label: "重要性", render: (value) => `${formatNumber(value * 100, 2)}%` },
            ]}
            rows={topFeatures}
          />
        </Panel>
        <Panel title="特征组贡献度">
          <MiniTable
            columns={[
              { key: "feature_group", label: "特征组" },
              { key: "importance_pct", label: "贡献度", render: (value) => `${formatNumber(value * 100, 1)}%` },
            ]}
            rows={grouped}
          />
        </Panel>
      </div>

      <Panel title="销量变化最快的菜品">
        <MiniTable
          columns={[
            { key: "dish_name", label: "菜品" },
            { key: "first", label: "起始预测" },
            { key: "last", label: "末日预测" },
            { key: "delta", label: "变化量", render: (value) => formatNumber(value, 1) },
          ]}
          rows={predictionMomentum.slice(0, 8)}
        />
      </Panel>
    </>
  );
}

export function ManagementPlanView({ data, windowSummary }) {
  return (
    <>
      <div className="metric-grid">
        <MetricCard label="缺货风险菜品" value={`${(data.management.mealPlan || []).filter((item) => item.risk === "缺货风险较高").length} 个`} tone="warm" />
        <MetricCard label="浪费风险菜品" value={`${(data.management.mealPlan || []).filter((item) => item.risk === "浪费风险较高").length} 个`} tone="berry" />
        <MetricCard
          label="总预测销量"
          value={`${formatNumber((data.management.mealPlan || []).reduce((sum, item) => sum + Number(item.pred_quantity || 0), 0))} 份`}
          tone="blue"
        />
        <MetricCard
          label="总建议备餐"
          value={`${formatNumber((data.management.mealPlan || []).reduce((sum, item) => sum + Number(item.suggested_prep || 0), 0), 0)} 份`}
          tone="teal"
        />
      </div>
      <Panel title="未来 7 天备餐建议">
        <MiniTable
          columns={[
            { key: "date", label: "日期" },
            { key: "dish_name", label: "菜品" },
            { key: "pred_quantity", label: "预测销量" },
            { key: "suggested_prep", label: "建议备餐量" },
            { key: "risk", label: "风险" },
            { key: "advice", label: "建议" },
          ]}
          rows={data.management.mealPlan}
        />
      </Panel>
      <Panel title="窗口级备餐汇总">
        <MiniTable
          columns={[
            { key: "window", label: "窗口" },
            { key: "pred_total", label: "预测总销量", render: (value) => formatNumber(value, 1) },
            { key: "prep_total", label: "建议备餐总量", render: (value) => formatNumber(value, 0) },
            { key: "high_risk", label: "高风险菜品数" },
          ]}
          rows={windowSummary}
        />
      </Panel>
    </>
  );
}
