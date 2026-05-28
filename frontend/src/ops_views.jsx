import React, { useMemo, useState } from "react";
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
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { MetricCard, MiniTable, Panel } from "./components";
import { PIE_COLORS } from "./constants";
import { DEFAULT_STRATEGY, simulateStrategy } from "./strategy_simulator";
import { formatNumber } from "./utils";

function pivotDailySales(rows) {
  const byDate = new Map();
  rows.forEach((item) => {
    if (!byDate.has(item.date)) byDate.set(item.date, { date: item.date });
    byDate.get(item.date)[item.canteen] = Number(item.sales || 0);
  });
  return [...byDate.values()].sort((a, b) => a.date.localeCompare(b.date));
}

function pivotSatisfaction(rows) {
  const byDate = new Map();
  rows.forEach((item) => {
    if (!byDate.has(item.date)) byDate.set(item.date, { date: item.date });
    byDate.get(item.date)[item.canteen] = Number(item.composite_score || 0);
  });
  return [...byDate.values()].sort((a, b) => a.date.localeCompare(b.date));
}

function buildReportViewModel(data) {
  const daily = data.logistics.reportSections?.daily || {};
  const weekly = data.logistics.reportSections?.weekly || [];
  const monthly = data.logistics.reportSections?.monthly || {};
  const highPriority = (data.logistics.decisionActions || []).filter((item) => item.priority === "高");
  const topEfficiency = data.logistics.efficiencyRanking?.[0];
  const topRisk = data.logistics.riskOverview?.[0];

  return {
    daily: {
      title: "每日运行简报",
      subtitle: "聚焦最新一天的经营表现、风险状态和当天动作建议。",
      summary: [
        `最新统计日期为 ${daily.date || "-"}。`,
        `当日销售额 ${formatNumber(daily.sales || 0)} 元，订单量 ${formatNumber(daily.orders || 0, 0)} 单。`,
        `建议优先关注 ${topRisk?.canteen || "当前风险最高食堂"} 的备餐与排队压力。`,
      ],
      metrics: [
        { label: "统计日期", value: daily.date || "-" },
        { label: "当日销售额", value: `${formatNumber(daily.sales || 0)} 元` },
        { label: "当日订单量", value: `${formatNumber(daily.orders || 0, 0)} 单` },
        { label: "高优先建议", value: `${highPriority.length} 条` },
      ],
      bullets: highPriority.slice(0, 4).map((item) => `${item.canteen}：${item.advice}`),
      table: {
        title: "当日重点行动",
        columns: [
          { key: "priority", label: "优先级" },
          { key: "canteen", label: "食堂" },
          { key: "theme", label: "主题" },
          { key: "advice", label: "建议" },
        ],
        rows: highPriority.slice(0, 6),
      },
    },
    weekly: {
      title: "每周经营报告",
      subtitle: "展示最近几周的销售变化，用于观察趋势和资源配置节奏。",
      summary: [
        `近 ${weekly.length} 周经营数据已完成汇总。`,
        `综合效率最高的食堂是 ${topEfficiency?.canteen || "-"}，效率得分 ${formatNumber(topEfficiency?.efficiency_score || 0, 1)}。`,
        "适合用于周例会复盘、排班调整和促销效果评估。",
      ],
      metrics: [
        { label: "统计周数", value: `${weekly.length} 周` },
        { label: "最佳效率食堂", value: topEfficiency?.canteen || "-" },
        { label: "效率得分", value: formatNumber(topEfficiency?.efficiency_score || 0, 1) },
        { label: "最高风险食堂", value: monthly.highest_risk_canteen || "-" },
      ],
      bullets: (data.logistics.decisionActions || []).slice(0, 4).map((item) => `${item.theme}：${item.advice}`),
      table: {
        title: "周度经营数据",
        columns: [
          { key: "week", label: "周次" },
          { key: "sales", label: "销售额", render: (value) => `${formatNumber(value)} 元` },
          { key: "orders", label: "订单量", render: (value) => formatNumber(value, 0) },
        ],
        rows: weekly,
      },
    },
    monthly: {
      title: "月度决策报告",
      subtitle: "更适合课程答辩展示，强调综合效率、风险和长期优化方向。",
      summary: [
        `本月综合效率最优食堂为 ${monthly.best_canteen || "-"}。`,
        `当前最高风险食堂为 ${monthly.highest_risk_canteen || "-"}，风险等级 ${monthly.highest_risk_level || "-" }。`,
        "建议围绕备餐风险、满意度波动与食堂品类结构，持续优化资源投放。",
      ],
      metrics: [
        { label: "最佳效率食堂", value: monthly.best_canteen || "-" },
        { label: "效率得分", value: formatNumber(monthly.best_efficiency_score || 0, 1) },
        { label: "最高风险食堂", value: monthly.highest_risk_canteen || "-" },
        { label: "风险等级", value: monthly.highest_risk_level || "-" },
      ],
      bullets: [
        `优先保障 ${monthly.highest_risk_canteen || "高风险食堂"} 的高峰备餐和窗口调度。`,
        `复用 ${monthly.best_canteen || "最佳食堂"} 的排班与菜品策略。`,
        "结合学生满意度趋势，持续优化口味稳定性和服务体验。",
      ],
      table: {
        title: "月度关键建议",
        columns: [
          { key: "priority", label: "优先级" },
          { key: "canteen", label: "食堂" },
          { key: "theme", label: "主题" },
          { key: "advice", label: "建议" },
        ],
        rows: data.logistics.decisionActions || [],
      },
    },
  };
}

function AlertCard({ item }) {
  return (
    <article className={`alert-card severity-${item.severity}`}>
      <div className="alert-top">
        <span>{item.severity}优先</span>
        <strong>{item.category}</strong>
      </div>
      <h4>{item.entity}</h4>
      <p>{item.signal}</p>
      <div className="alert-meta">
        <em>{item.change}</em>
        <span>{item.advice}</span>
      </div>
    </article>
  );
}

function StrategyOutcome({ simulation, base }) {
  if (!simulation || !base) return null;
  return (
    <>
      <div className="metric-grid">
        <MetricCard label="模拟销售额" value={`${formatNumber(simulation.sales)} 元`} tone="warm" hint={`较当前 ${simulation.deltas.sales >= 0 ? "+" : ""}${formatNumber(simulation.deltas.sales)} 元`} />
        <MetricCard label="模拟订单量" value={`${formatNumber(simulation.orders, 0)} 单`} tone="teal" hint={`较当前 ${simulation.deltas.orders >= 0 ? "+" : ""}${formatNumber(simulation.deltas.orders, 0)} 单`} />
        <MetricCard label="模拟利润" value={`${formatNumber(simulation.profit)} 元`} tone="blue" hint={`较当前 ${simulation.deltas.profit >= 0 ? "+" : ""}${formatNumber(simulation.deltas.profit)} 元`} />
        <MetricCard label="模拟评分" value={formatNumber(simulation.rating, 2)} tone="berry" hint={`较当前 ${simulation.deltas.rating >= 0 ? "+" : ""}${formatNumber(simulation.deltas.rating, 2)}`} />
      </div>
      <div className="two-col">
        <Panel title="风险变化">
          <MiniTable
            columns={[
              { key: "item", label: "指标" },
              { key: "current", label: "当前" },
              { key: "simulated", label: "模拟后" },
            ]}
            rows={[
              { item: "平均排队时间", current: `${base.queue_minutes} 分钟`, simulated: `${formatNumber(simulation.queueMinutes, 1)} 分钟` },
              { item: "缺货风险菜品数", current: base.shortage_count, simulated: simulation.shortageCount },
              { item: "浪费风险菜品数", current: base.waste_count, simulated: simulation.wasteCount },
            ]}
          />
        </Panel>
        <Panel title="策略结论">
          <div className="bullet-stack">
            {simulation.summary.map((item, index) => (
              <div className="bullet-row" key={`${index}-${item}`}>
                {item}
              </div>
            ))}
            {simulation.assumptions.map((item, index) => (
              <div className="bullet-row" key={`assume-${index}`}>
                假设：{item}
              </div>
            ))}
          </div>
        </Panel>
      </div>
    </>
  );
}

export function OpsCompareView({ data }) {
  const salesTrend = useMemo(() => pivotDailySales(data.logistics.canteenDailySales || []), [data]);
  const satisfactionTrend = useMemo(() => pivotSatisfaction(data.logistics.satisfactionTrend || []), [data]);
  const canteens = data.meta.canteens || [];
  const defaultCanteen = canteens[0] || "";
  const alerts = data.logistics.alertCenter || [];
  const alertSummary = data.logistics.alertSummary || { high: 0, medium: 0, low: 0, total: 0 };
  const baselines = data.logistics.strategySimulator || [];

  const [strategy, setStrategy] = useState({ ...DEFAULT_STRATEGY, canteen: defaultCanteen });
  const selectedBaseline = useMemo(
    () => baselines.find((item) => item.canteen === strategy.canteen) || baselines[0] || null,
    [baselines, strategy.canteen],
  );
  const simulation = useMemo(() => simulateStrategy(selectedBaseline, strategy), [selectedBaseline, strategy]);

  return (
    <>
      <div className="metric-grid">
        <MetricCard label="食堂数量" value={`${data.logistics.canteenCompare.length} 个`} tone="warm" />
        <MetricCard label="最高效率食堂" value={data.logistics.efficiencyRanking?.[0]?.canteen || "-"} tone="teal" />
        <MetricCard label="最高风险食堂" value={data.logistics.reportSections?.monthly?.highest_risk_canteen || "-"} tone="berry" />
        <MetricCard label="高优先建议数" value={`${(data.logistics.decisionActions || []).filter((item) => item.priority === "高").length} 条`} tone="blue" />
      </div>

      <Panel title="策略模拟器" subtitle="调整价格、折扣、人手和备餐系数，模拟经营结果变化。">
        <div className="simulation-form">
          <label>
            <span>模拟食堂</span>
            <select value={strategy.canteen} onChange={(e) => setStrategy((prev) => ({ ...prev, canteen: e.target.value }))}>
              {baselines.map((item) => (
                <option key={item.canteen} value={item.canteen}>
                  {item.canteen}
                </option>
              ))}
            </select>
          </label>
          <label>
            <span>价格调整（元）</span>
            <input type="range" min="-2" max="2" step="0.5" value={strategy.priceDelta} onChange={(e) => setStrategy((prev) => ({ ...prev, priceDelta: Number(e.target.value) }))} />
            <strong>{strategy.priceDelta >= 0 ? "+" : ""}{strategy.priceDelta}</strong>
          </label>
          <label>
            <span>折扣系数</span>
            <input type="range" min="0.8" max="1" step="0.02" value={strategy.discountRate} onChange={(e) => setStrategy((prev) => ({ ...prev, discountRate: Number(e.target.value) }))} />
            <strong>{formatNumber(strategy.discountRate, 2)}</strong>
          </label>
          <label>
            <span>增加人手</span>
            <input type="range" min="0" max="3" step="1" value={strategy.staffingDelta} onChange={(e) => setStrategy((prev) => ({ ...prev, staffingDelta: Number(e.target.value) }))} />
            <strong>+{strategy.staffingDelta} 人</strong>
          </label>
          <label>
            <span>备餐系数</span>
            <input type="range" min="0.95" max="1.2" step="0.01" value={strategy.prepFactor} onChange={(e) => setStrategy((prev) => ({ ...prev, prepFactor: Number(e.target.value) }))} />
            <strong>{formatNumber(strategy.prepFactor, 2)}</strong>
          </label>
        </div>
        <StrategyOutcome simulation={simulation} base={selectedBaseline} />
      </Panel>

      <Panel title="异常预警中心" subtitle="自动识别满意度下降、订单异常、销量突变、备餐风险和拥挤度异常。">
        <div className="metric-grid">
          <MetricCard label="异常总数" value={`${alertSummary.total || 0} 条`} tone="warm" />
          <MetricCard label="高优先预警" value={`${alertSummary.high || 0} 条`} tone="berry" />
          <MetricCard label="中优先预警" value={`${alertSummary.medium || 0} 条`} tone="teal" />
          <MetricCard label="低优先提醒" value={`${alertSummary.low || 0} 条`} tone="blue" />
        </div>
        <div className="alert-grid">
          {alerts.slice(0, 6).map((item, index) => (
            <AlertCard key={`${item.category}-${item.entity}-${index}`} item={item} />
          ))}
        </div>
      </Panel>

      <div className="two-col">
        <Panel title="多食堂经营规模">
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={data.logistics.canteenCompare}>
              <CartesianGrid strokeDasharray="3 3" stroke="#ffffff18" />
              <XAxis dataKey="canteen" stroke="#d6d0f4" />
              <YAxis stroke="#d6d0f4" />
              <Tooltip />
              <Legend />
              <Bar dataKey="sales" fill="#f97316" radius={[8, 8, 0, 0]} />
              <Bar dataKey="students" fill="#14b8a6" radius={[8, 8, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Panel>
        <Panel title="综合效率排名">
          <MiniTable
            columns={[
              { key: "canteen", label: "食堂" },
              { key: "efficiency_score", label: "效率分" },
              { key: "sales_per_student", label: "人均销售额", render: (value) => `${formatNumber(value, 2)} 元` },
              { key: "orders_per_student", label: "人均订单数", render: (value) => formatNumber(value, 2) },
            ]}
            rows={data.logistics.efficiencyRanking || []}
          />
        </Panel>
      </div>

      <div className="two-col">
        <Panel title="多食堂销售趋势">
          <ResponsiveContainer width="100%" height={320}>
            <AreaChart data={salesTrend}>
              <CartesianGrid strokeDasharray="3 3" stroke="#ffffff18" />
              <XAxis dataKey="date" stroke="#d6d0f4" />
              <YAxis stroke="#d6d0f4" />
              <Tooltip />
              <Legend />
              {canteens.map((canteen, index) => (
                <Area key={canteen} type="monotone" dataKey={canteen} stackId="sales" stroke={PIE_COLORS[index % PIE_COLORS.length]} fill={PIE_COLORS[index % PIE_COLORS.length]} fillOpacity={0.22} />
              ))}
            </AreaChart>
          </ResponsiveContainer>
        </Panel>
        <Panel title="满意度趋势">
          <ResponsiveContainer width="100%" height={320}>
            <LineChart data={satisfactionTrend}>
              <CartesianGrid strokeDasharray="3 3" stroke="#ffffff18" />
              <XAxis dataKey="date" stroke="#d6d0f4" />
              <YAxis domain={[3.5, 5]} stroke="#d6d0f4" />
              <Tooltip />
              <Legend />
              {canteens.map((canteen, index) => (
                <Line key={canteen} type="monotone" dataKey={canteen} stroke={PIE_COLORS[index % PIE_COLORS.length]} strokeWidth={2.5} dot={false} />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </Panel>
      </div>

      <div className="two-col">
        <Panel title="资源利用指数">
          <MiniTable
            columns={[
              { key: "canteen", label: "食堂" },
              { key: "utilization_index", label: "资源利用指数" },
              { key: "pred_total", label: "预测销量", render: (value) => formatNumber(value, 1) },
              { key: "prep_total", label: "建议备餐", render: (value) => formatNumber(value, 0) },
            ]}
            rows={data.logistics.resourceUtilization || []}
          />
        </Panel>
        <Panel title="备餐风险总览">
          <MiniTable
            columns={[
              { key: "canteen", label: "食堂" },
              { key: "shortage_count", label: "缺货风险菜品数" },
              { key: "waste_count", label: "浪费风险菜品数" },
              { key: "prep_ratio", label: "备餐系数" },
              { key: "risk_level", label: "风险等级" },
            ]}
            rows={data.logistics.riskOverview || []}
          />
        </Panel>
      </div>

      <div className="two-col">
        <Panel title="学生群体结构">
          <div className="cluster-list">
            {data.logistics.clusterProfile.map((item) => (
              <div className="cluster-card" key={item.cluster}>
                <strong>{item.cluster_name}</strong>
                <span>{formatNumber(item.size_ratio * 100)}%</span>
                <em>平均客单价 {formatNumber(item.avg_ticket)} 元</em>
              </div>
            ))}
          </div>
        </Panel>
        <Panel title={`食堂品类结构${defaultCanteen ? ` · ${defaultCanteen}` : ""}`}>
          <ResponsiveContainer width="100%" height={320}>
            <PieChart>
              <Pie data={(data.logistics.categoryStructure || []).filter((item) => item.canteen === defaultCanteen)} dataKey="sales" nameKey="category" outerRadius={104}>
                {(data.logistics.categoryStructure || [])
                  .filter((item) => item.canteen === defaultCanteen)
                  .map((entry, index) => (
                    <Cell key={`${entry.category}-${index}`} fill={PIE_COLORS[index % PIE_COLORS.length]} />
                  ))}
              </Pie>
              <Tooltip />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </Panel>
      </div>

      <Panel title="异常预警明细">
        <MiniTable
          columns={[
            { key: "severity", label: "严重程度" },
            { key: "category", label: "异常类型" },
            { key: "entity", label: "对象" },
            { key: "signal", label: "异常信号" },
            { key: "change", label: "变化说明" },
            { key: "advice", label: "处理建议" },
          ]}
          rows={alerts}
        />
      </Panel>

      <Panel title="后勤决策建议">
        <MiniTable
          columns={[
            { key: "priority", label: "优先级" },
            { key: "canteen", label: "食堂" },
            { key: "theme", label: "主题" },
            { key: "advice", label: "建议" },
          ]}
          rows={data.logistics.decisionActions || []}
        />
      </Panel>
    </>
  );
}

export function OpsReportView({ reportText, data }) {
  const [mode, setMode] = useState("daily");
  const reportModes = buildReportViewModel(data);
  const current = reportModes[mode];
  const alerts = data.logistics.alertCenter || [];
  const highAlerts = alerts.filter((item) => item.severity === "高");

  const exportReport = () => {
    const sections = [
      current.title,
      current.subtitle,
      "",
      "一、摘要",
      ...current.summary.map((item, index) => `${index + 1}. ${item}`),
      "",
      "二、关键指标",
      ...current.metrics.map((item) => `${item.label}：${item.value}`),
      "",
      "三、高优先异常",
      ...(highAlerts.length
        ? highAlerts.slice(0, 6).map((item, index) => `${index + 1}. [${item.category}] ${item.entity} - ${item.signal}；${item.advice}`)
        : ["1. 当前无高优先异常。"]),
      "",
      "四、建议",
      ...current.bullets.map((item, index) => `${index + 1}. ${item}`),
      "",
      "五、系统自动生成报告",
      reportText,
    ].join("\n");

    const blob = new Blob([sections], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `canteen-${mode}-report.txt`;
    link.click();
    URL.revokeObjectURL(url);
  };

  return (
    <>
      <div className="report-switcher">
        {[
          ["daily", "每日报告"],
          ["weekly", "每周报告"],
          ["monthly", "月度报告"],
        ].map(([key, label]) => (
          <button key={key} className={`report-tab ${mode === key ? "active" : ""}`} onClick={() => setMode(key)}>
            {label}
          </button>
        ))}
      </div>

      <section className="formal-report">
        <div className="formal-report-head">
          <div>
            <span className="report-kicker">后勤决策报告</span>
            <h2>{current.title}</h2>
            <p>{current.subtitle}</p>
          </div>
          <button className="ghost-button" onClick={exportReport}>
            导出当前报告
          </button>
        </div>

        <div className="formal-metrics">
          {current.metrics.map((item) => (
            <article className="formal-metric-card" key={item.label}>
              <span>{item.label}</span>
              <strong>{item.value}</strong>
            </article>
          ))}
        </div>

        <div className="formal-grid">
          <section className="formal-section">
            <h3>摘要概览</h3>
            <div className="formal-list">
              {current.summary.map((item, index) => (
                <div className="formal-list-item" key={`${index}-${item}`}>
                  <span>{index + 1}</span>
                  <p>{item}</p>
                </div>
              ))}
            </div>
          </section>

          <section className="formal-section">
            <h3>高优先异常</h3>
            <div className="formal-list">
              {(highAlerts.length ? highAlerts.slice(0, 4) : [{ entity: "全部食堂", signal: "当前无高优先异常" }]).map((item, index) => (
                <div className="formal-list-item" key={`${index}-${item.entity}`}>
                  <span>{index + 1}</span>
                  <p>{`${item.entity}：${item.signal}${item.advice ? `，${item.advice}` : ""}`}</p>
                </div>
              ))}
            </div>
          </section>
        </div>

        <section className="formal-section">
          <h3>{current.table.title}</h3>
          <MiniTable columns={current.table.columns} rows={current.table.rows} />
        </section>

        <section className="formal-section">
          <h3>系统自动生成报告正文</h3>
          <pre className="report-block">{reportText}</pre>
        </section>
      </section>
    </>
  );
}
