import React from "react";
import { Area, AreaChart, CartesianGrid, Cell, Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { BUDGET_OPTIONS, GOAL_OPTIONS, PIE_COLORS, TASTE_OPTIONS } from "./constants";
import { MetricCard, MiniTable, NutritionPie, Panel, RecommendationCard, StudentFeedbackView } from "./components";
import { formatNumber } from "./utils";

export function StudentTodayView({ data, studentId, setStudentId, filters, setFilters, recommendation, comboSuggestions }) {
  return (
    <>
      <section className="hero">
        <div>
          <span className="eyebrow">学生智能服务</span>
          <h1>今天吃什么，不再靠随缘</h1>
          <p>预算、口味、目标、时段与食堂位置一起参与推荐，让选餐更快，也让系统更像真正可用的校园服务平台。</p>
        </div>
        <div className="hero-blob">
          <span>实时推荐</span>
          <strong>{recommendation[0]?.dish_name || "午餐组合"}</strong>
          <em>{recommendation[0]?.reason || "根据历史偏好、评分与热度综合生成"}</em>
        </div>
      </section>

      <div className="control-strip">
        <select value={studentId} onChange={(e) => setStudentId(e.target.value)}>
          {data.meta.student_ids.map((id) => (
            <option key={id} value={id}>
              {id}
            </option>
          ))}
        </select>
        <select value={filters.budget} onChange={(e) => setFilters((prev) => ({ ...prev, budget: e.target.value }))}>
          {BUDGET_OPTIONS.map((item) => (
            <option key={item}>{item}</option>
          ))}
        </select>
        <select value={filters.period} onChange={(e) => setFilters((prev) => ({ ...prev, period: e.target.value }))}>
          {data.meta.periods.map((item) => (
            <option key={item}>{item}</option>
          ))}
        </select>
        <select value={filters.taste} onChange={(e) => setFilters((prev) => ({ ...prev, taste: e.target.value }))}>
          {TASTE_OPTIONS.map((item) => (
            <option key={item}>{item}</option>
          ))}
        </select>
        <select value={filters.goal} onChange={(e) => setFilters((prev) => ({ ...prev, goal: e.target.value }))}>
          {GOAL_OPTIONS.map((item) => (
            <option key={item}>{item}</option>
          ))}
        </select>
        <select value={filters.canteen} onChange={(e) => setFilters((prev) => ({ ...prev, canteen: e.target.value }))}>
          {["不限", ...data.meta.canteens].map((item) => (
            <option key={item}>{item}</option>
          ))}
        </select>
      </div>

      <Panel title="推荐菜品" subtitle="综合评分、销量热度、历史偏好和目标匹配的混合推荐">
        <div className="recommend-grid">
          {recommendation.map((item) => (
            <RecommendationCard key={`${item.dish_name}-${item.canteen}`} item={item} />
          ))}
        </div>
      </Panel>

      <div className="two-col">
        <Panel title="个性化套餐生成" subtitle="把关联规则和预算限制组合成更实用的一顿饭">
          <div className="recommend-grid compact-grid">
            {comboSuggestions.map((item) => (
              <article className="recommend-card" key={item.combo_name}>
                <div className="recommend-top">
                  <strong>{item.combo_name}</strong>
                  <span>{formatNumber(item.estimated_price)} 元</span>
                </div>
                <div className="recommend-meta">
                  <span>置信度 {formatNumber(item.confidence * 100)}%</span>
                  <span>提升度 {formatNumber(item.lift, 2)}</span>
                </div>
                <p>{item.reason}</p>
              </article>
            ))}
          </div>
        </Panel>

        <Panel title="公告与今日优惠" subtitle="补足真实校园服务场景，不只是看图表">
          <div className="notice-list">
            {(data.student.announcements || []).slice(0, 4).map((item, index) => (
              <article className="notice-card" key={`${item.title}-${index}`}>
                <span>{item.type || "通知"}</span>
                <strong>{item.title}</strong>
                <p>{item.content}</p>
              </article>
            ))}
            {(data.management.promotions || []).slice(0, 2).map((item) => (
              <article className="notice-card promo" key={item.dish_name}>
                <span>优惠建议</span>
                <strong>{item.dish_name}</strong>
                <p>{item.advice}</p>
              </article>
            ))}
          </div>
        </Panel>
      </div>

      <Panel title="当前拥挤度" subtitle="按历史 15 分钟粒度估计当前排队压力">
        <div className="crowding-grid">
          {data.student.crowding.map((item) => (
            <div className={`crowd-card level-${item.level}`} key={item.canteen}>
              <span>{item.canteen}</span>
              <strong>{item.level}</strong>
              <em>{item.queue_time}</em>
            </div>
          ))}
        </div>
      </Panel>
    </>
  );
}

export function StudentReportView({ currentStudent, studentFeature, peerRecommendation }) {
  return (
    <>
      <div className="metric-grid">
        <MetricCard label="累计消费" value={`${formatNumber(currentStudent.totalAmount)} 元`} tone="warm" />
        <MetricCard label="消费单数" value={`${formatNumber(currentStudent.groupedOrders.length, 0)} 单`} tone="teal" />
        <MetricCard label="平均客单价" value={`${formatNumber(currentStudent.avgTicket)} 元`} tone="blue" />
        <MetricCard label="常去食堂" value={currentStudent.favoriteCanteen} tone="berry" />
      </div>

      <div className="two-col">
        <Panel title="每日消费趋势">
          <ResponsiveContainer width="100%" height={280}>
            <AreaChart data={currentStudent.daily}>
              <CartesianGrid strokeDasharray="3 3" stroke="#ffffff18" />
              <XAxis dataKey="date" stroke="#d6d0f4" />
              <YAxis stroke="#d6d0f4" />
              <Tooltip />
              <Area type="monotone" dataKey="amount" stroke="#fb923c" fill="url(#amountFill)" />
              <defs>
                <linearGradient id="amountFill" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#fb923c" stopOpacity="0.75" />
                  <stop offset="100%" stopColor="#fb923c" stopOpacity="0.05" />
                </linearGradient>
              </defs>
            </AreaChart>
          </ResponsiveContainer>
        </Panel>
        <Panel title="时段偏好">
          <ResponsiveContainer width="100%" height={280}>
            <PieChart>
              <Pie data={currentStudent.periods} dataKey="orders" nameKey="period" outerRadius={100} innerRadius={56}>
                {currentStudent.periods.map((entry, index) => (
                  <Cell key={entry.period} fill={PIE_COLORS[index % PIE_COLORS.length]} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </Panel>
      </div>

      <Panel title="常吃菜品">
        <MiniTable
          columns={[
            { key: "dish_name", label: "菜品" },
            { key: "freq", label: "出现次数" },
            { key: "amount", label: "消费金额", render: (value) => `${formatNumber(value)} 元` },
          ]}
          rows={currentStudent.topDishes}
        />
      </Panel>

      <div className="two-col">
        <Panel title="我的消费群体标签">
          <div className="cluster-card full-height">
            <strong>{studentFeature?.cluster_name || "未识别"}</strong>
            <span>高价菜占比 {formatNumber((studentFeature?.high_price_ratio || 0) * 100)}%</span>
            <em>
              早餐 {formatNumber((studentFeature?.早餐_ratio || 0) * 100)}% · 夜宵 {formatNumber((studentFeature?.夜宵_ratio || 0) * 100)}%
            </em>
          </div>
        </Panel>
        <Panel title="同群体同学还爱吃什么">
          <MiniTable
            columns={[
              { key: "dish_name", label: "菜品" },
              { key: "sales_qty", label: "销量" },
              { key: "avg_rating", label: "评分", render: (value) => formatNumber(value, 1) },
              { key: "reason", label: "理由" },
            ]}
            rows={peerRecommendation || []}
          />
        </Panel>
      </div>
    </>
  );
}

export function StudentBudgetView({ monthBudget, setMonthBudget, currentStudent, budgetAdvice, nutrition }) {
  return (
    <>
      <div className="control-strip">
        <label className="budget-input">
          <span>月预算</span>
          <input type="number" value={monthBudget} onChange={(e) => setMonthBudget(Number(e.target.value || 0))} />
        </label>
      </div>
      <div className="metric-grid">
        <MetricCard label="已消费" value={`${formatNumber(currentStudent.totalAmount)} 元`} tone="warm" />
        <MetricCard label="剩余预算" value={`${formatNumber(monthBudget - currentStudent.totalAmount)} 元`} tone="teal" />
        <MetricCard label="预算完成率" value={`${formatNumber((currentStudent.totalAmount / Math.max(monthBudget, 1)) * 100)} %`} tone="blue" />
        <MetricCard label="状态" value={currentStudent.totalAmount > monthBudget ? "超支预警" : "预算正常"} tone={currentStudent.totalAmount > monthBudget ? "berry" : "teal"} />
      </div>

      <div className="two-col">
        <Panel title="预算进度">
          <div className="progress-shell">
            <div
              className={`progress-bar ${currentStudent.totalAmount > monthBudget ? "danger" : ""}`}
              style={{ width: `${Math.min((currentStudent.totalAmount / Math.max(monthBudget, 1)) * 100, 100)}%` }}
            />
          </div>
          <p className="progress-copy">按当前节奏，建议后续优先选择 {currentStudent.avgTicket > 15 ? "10-15 元以内" : "当前正常区间"} 的套餐。</p>
        </Panel>
        <Panel title="预算节奏建议">
          <div className="advice-stack">
            <div className="advice-chip">
              <strong>剩余预算</strong>
              <span>{formatNumber(budgetAdvice.remaining)} 元</span>
            </div>
            <div className="advice-chip">
              <strong>月底预测</strong>
              <span>{formatNumber(budgetAdvice.projected)} 元</span>
            </div>
            <div className="advice-chip">
              <strong>建议日均</strong>
              <span>{formatNumber(budgetAdvice.dailyTarget)} 元 / 天</span>
            </div>
          </div>
        </Panel>
      </div>

      <Panel title="营养结构分析">
        <div className="two-col">
          <NutritionPie nutrition={nutrition} />
          <div className="advice-stack">
            {nutrition.map((item) => (
              <div className="advice-chip" key={item.healthy_tag}>
                <strong>{item.healthy_tag}</strong>
                <span>{formatNumber(item.ratio * 100)}%</span>
              </div>
            ))}
            <div className="health-note">
              {["油炸", "高糖", "重口"].includes(nutrition[0]?.healthy_tag)
                ? "近期油炸、高糖或重口占比较高，建议增加清淡菜、汤品和蔬菜。"
                : "当前饮食结构相对均衡，可以继续保持。"}
            </div>
          </div>
        </div>
      </Panel>

      <Panel title="高消费菜品提醒">
        <MiniTable
          columns={[
            { key: "dish_name", label: "菜品" },
            { key: "amount", label: "累计消费", render: (value) => `${formatNumber(value)} 元` },
            { key: "freq", label: "出现次数" },
          ]}
          rows={[...currentStudent.topDishes].sort((a, b) => b.amount - a.amount).slice(0, 6)}
        />
      </Panel>
    </>
  );
}

export { StudentFeedbackView };
