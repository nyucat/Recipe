import React, { startTransition, useEffect, useMemo, useState, useDeferredValue } from "react";
import {
  Activity,
  BarChart3,
  BadgeDollarSign,
  BookOpenText,
  ChefHat,
  CircleGauge,
  ClipboardList,
  HeartPulse,
  LineChart as LineChartIcon,
  MessageSquareText,
  Salad,
  Sparkles,
  Store,
  Users,
} from "lucide-react";
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

const NAV_GROUPS = [
  {
    title: "学生端",
    items: [
      { key: "student-today", label: "今日吃什么", icon: ChefHat },
      { key: "student-report", label: "我的消费报告", icon: BookOpenText },
      { key: "student-budget", label: "预算与健康建议", icon: HeartPulse },
      { key: "student-feedback", label: "菜品评分反馈", icon: MessageSquareText },
    ],
  },
  {
    title: "管理端",
    items: [
      { key: "mgmt-dashboard", label: "经营数据看板", icon: CircleGauge },
      { key: "mgmt-sales", label: "菜品销售分析", icon: BarChart3 },
      { key: "mgmt-peak", label: "消费高峰分析", icon: Activity },
      { key: "mgmt-rules", label: "关联规则分析", icon: Sparkles },
      { key: "mgmt-predict", label: "销量预测", icon: LineChartIcon },
      { key: "mgmt-plan", label: "备餐建议", icon: ClipboardList },
    ],
  },
  {
    title: "后勤端",
    items: [
      { key: "ops-compare", label: "多食堂对比", icon: Store },
      { key: "ops-report", label: "自动报告生成", icon: Users },
    ],
  },
];

const PIE_COLORS = ["#f97316", "#14b8a6", "#0ea5e9", "#8b5cf6", "#f43f5e", "#84cc16"];

const DISH_TASTE = {
  包子: "咸",
  豆浆: "甜",
  鸡蛋: "咸",
  热干面: "咸",
  牛肉面: "咸",
  黄焖鸡米饭: "咸",
  麻辣香锅: "辣",
  番茄炒蛋: "清淡",
  红烧肉: "咸",
  青椒土豆丝: "辣",
  紫菜蛋花汤: "清淡",
  绿豆汤: "甜",
  奶茶: "甜",
  可乐: "甜",
  炸鸡排: "咸",
  炒饭: "咸",
  炒面: "咸",
  水果沙拉: "清淡",
};

const GOAL_MAP = {
  吃饱: new Set(["主食", "热菜"]),
  减脂: new Set(["轻食", "汤品"]),
  高蛋白: new Set(["热菜", "主食"]),
  省钱: new Set(["主食", "汤品"]),
  尝鲜: new Set(["饮品", "小吃", "轻食"]),
};

function formatNumber(value, digits = 1) {
  return Number(value || 0).toLocaleString("zh-CN", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

function useLocalJsonState(key, fallback) {
  const [state, setState] = useState(() => {
    const raw = window.localStorage.getItem(key);
    return raw ? JSON.parse(raw) : fallback;
  });

  useEffect(() => {
    window.localStorage.setItem(key, JSON.stringify(state));
  }, [key, state]);

  return [state, setState];
}

function MetricCard({ label, value, tone = "warm", hint }) {
  return (
    <div className={`metric-card tone-${tone}`}>
      <div className="metric-label">{label}</div>
      <div className="metric-value">{value}</div>
      {hint ? <div className="metric-hint">{hint}</div> : null}
    </div>
  );
}

function Panel({ title, subtitle, right, children }) {
  return (
    <section className="panel">
      <div className="panel-head">
        <div>
          <h3>{title}</h3>
          {subtitle ? <p>{subtitle}</p> : null}
        </div>
        {right}
      </div>
      <div className="panel-body">{children}</div>
    </section>
  );
}

function MiniTable({ columns, rows }) {
  return (
    <div className="table-shell">
      <table className="data-table">
        <thead>
          <tr>
            {columns.map((col) => (
              <th key={col.key}>{col.label}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, idx) => (
            <tr key={row.id || `${idx}-${columns[0].key}`}>
              {columns.map((col) => (
                <td key={col.key}>{col.render ? col.render(row[col.key], row) : row[col.key]}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function RecommendationCard({ item }) {
  return (
    <article className="recommend-card">
      <div className="recommend-top">
        <strong>{item.dish_name}</strong>
        <span>{formatNumber(item.avg_price)} 元</span>
      </div>
      <div className="recommend-meta">
        <span>{item.canteen}</span>
        <span>{item.window}</span>
        <span>评分 {formatNumber(item.avg_rating, 1)}</span>
      </div>
      <p>{item.reason}</p>
    </article>
  );
}

function HeatGrid({ data }) {
  const weekdays = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"];
  const hours = [...new Set(data.map((item) => item.hour))].sort((a, b) => a - b);
  const max = Math.max(...data.map((item) => item.orders || 0), 1);
  return (
    <div className="heat-grid">
      <div className="heat-header" />
      {hours.map((hour) => (
        <div className="heat-hour" key={`h-${hour}`}>
          {hour}
        </div>
      ))}
      {weekdays.map((day) => (
        <React.Fragment key={day}>
          <div className="heat-day">{day}</div>
          {hours.map((hour) => {
            const cell = data.find((item) => item.weekday_cn === day && item.hour === hour);
            const ratio = (cell?.orders || 0) / max;
            return (
              <div
                className="heat-cell"
                key={`${day}-${hour}`}
                style={{ opacity: 0.18 + ratio * 0.82 }}
                title={`${day} ${hour}:00 - ${cell?.orders || 0} 单`}
              >
                {cell?.orders || 0}
              </div>
            );
          })}
        </React.Fragment>
      ))}
    </div>
  );
}

function buildStudentRecommendation(orders, studentId, filters, disliked) {
  const studentOrders = orders.filter((item) => item.student_id === studentId);
  const favoriteCategory =
    Object.entries(
      studentOrders.reduce((acc, item) => {
        acc[item.category] = (acc[item.category] || 0) + 1;
        return acc;
      }, {}),
    ).sort((a, b) => b[1] - a[1])[0]?.[0] || null;

  const grouped = Object.values(
    orders.reduce((acc, item) => {
      const key = `${item.dish_name}-${item.canteen}-${item.window}`;
      if (!acc[key]) {
        acc[key] = {
          dish_name: item.dish_name,
          canteen: item.canteen,
          window: item.window,
          category: item.category,
          avg_price: 0,
          avg_rating: 0,
          sales_qty: 0,
          count: 0,
          periodCount: 0,
        };
      }
      acc[key].avg_price += Number(item.price || 0);
      acc[key].avg_rating += Number(item.rating || 0);
      acc[key].sales_qty += Number(item.quantity || 0);
      acc[key].count += 1;
      if (item.period === filters.period) acc[key].periodCount += 1;
      return acc;
    }, {}),
  ).map((item) => ({
    ...item,
    avg_price: item.avg_price / item.count,
    avg_rating: item.avg_rating / item.count,
    period_ratio: item.periodCount / item.count,
    taste: DISH_TASTE[item.dish_name] || "咸",
  }));

  const budgetRanges = {
    "10元以内": [0, 10],
    "10-15元": [10, 15],
    "15-20元": [15, 20],
    不限: [0, 999],
  };

  const [minBudget, maxBudget] = budgetRanges[filters.budget];
  const maxSales = Math.max(...grouped.map((item) => item.sales_qty || 0), 1);

  return grouped
    .filter((item) => item.avg_price >= minBudget && item.avg_price <= maxBudget)
    .filter((item) => (filters.canteen === "不限" ? true : item.canteen === filters.canteen))
    .filter((item) => (filters.taste === "不限" ? true : item.taste === filters.taste))
    .filter((item) => !disliked.includes(item.dish_name))
    .map((item) => {
      let score = 0;
      score += (item.avg_rating / 5) * 0.25;
      score += (item.sales_qty / maxSales) * 0.2;
      score += item.period_ratio * 0.15;
      score += GOAL_MAP[filters.goal]?.has(item.category) ? 0.25 : 0;
      score += favoriteCategory && item.category === favoriteCategory ? 0.15 : 0;
      const reasons = [
        `评分 ${formatNumber(item.avg_rating, 1)}`,
        `均价 ${formatNumber(item.avg_price, 1)} 元`,
        `${filters.period}时段适配度较高`,
      ];
      if (favoriteCategory && item.category === favoriteCategory) reasons.push("符合你的历史偏好");
      if (GOAL_MAP[filters.goal]?.has(item.category)) reasons.push(`符合“${filters.goal}”目标`);
      return { ...item, score, reason: reasons.join("；") };
    })
    .sort((a, b) => b.score - a.score)
    .slice(0, 6);
}

function buildStudentSnapshot(orders, studentId) {
  const list = orders.filter((item) => item.student_id === studentId);
  const groupedOrders = Object.values(
    list.reduce((acc, item) => {
      if (!acc[item.order_id]) {
        acc[item.order_id] = { amount: 0, order_time: item.order_time };
      }
      acc[item.order_id].amount += Number(item.amount || 0);
      return acc;
    }, {}),
  );

  const daily = Object.values(
    list.reduce((acc, item) => {
      const date = String(item.date);
      acc[date] = acc[date] || { date, amount: 0 };
      acc[date].amount += Number(item.amount || 0);
      return acc;
    }, {}),
  ).sort((a, b) => a.date.localeCompare(b.date));

  const topDishes = Object.values(
    list.reduce((acc, item) => {
      acc[item.dish_name] = acc[item.dish_name] || { dish_name: item.dish_name, freq: 0, amount: 0 };
      acc[item.dish_name].freq += 1;
      acc[item.dish_name].amount += Number(item.amount || 0);
      return acc;
    }, {}),
  )
    .sort((a, b) => b.freq - a.freq)
    .slice(0, 8);

  const periods = Object.values(
    list.reduce((acc, item) => {
      acc[item.period] = acc[item.period] || { period: item.period, orders: 0 };
      acc[item.period].orders += 1;
      return acc;
    }, {}),
  ).sort((a, b) => b.orders - a.orders);

  const totalAmount = list.reduce((sum, item) => sum + Number(item.amount || 0), 0);
  const avgTicket = groupedOrders.length ? totalAmount / groupedOrders.length : 0;
  const favoriteCanteen =
    Object.entries(
      list.reduce((acc, item) => {
        acc[item.canteen] = (acc[item.canteen] || 0) + 1;
        return acc;
      }, {}),
    ).sort((a, b) => b[1] - a[1])[0]?.[0] || "-";

  return { list, groupedOrders, daily, topDishes, periods, totalAmount, avgTicket, favoriteCanteen };
}

function buildNutrition(studentOrders) {
  const healthMap = {
    水果沙拉: "健康",
    紫菜蛋花汤: "清淡",
    绿豆汤: "清淡",
    炸鸡排: "油炸",
    奶茶: "高糖",
    可乐: "高糖",
    麻辣香锅: "重口",
  };

  const groups = Object.values(
    studentOrders.reduce((acc, item) => {
      const key = healthMap[item.dish_name] || "均衡";
      acc[key] = acc[key] || { healthy_tag: key, quantity: 0 };
      acc[key].quantity += Number(item.quantity || 0);
      return acc;
    }, {}),
  );
  const total = groups.reduce((sum, item) => sum + item.quantity, 0) || 1;
  return groups.map((item) => ({ ...item, ratio: item.quantity / total })).sort((a, b) => b.ratio - a.ratio);
}

function buildReportText(data) {
  const kpis = data.summary.kpis;
  const bestRule = data.management.rules[0];
  const bestCluster = data.logistics.clusterProfile[0];
  const topDish = data.management.hotDishes[0];
  return [
    "校园食堂智能服务系统自动报告",
    "",
    `总体销售额：${formatNumber(kpis.total_sales)} 元，订单数 ${formatNumber(kpis.total_orders, 0)} 单，消费学生 ${formatNumber(kpis.total_students, 0)} 人。`,
    `当前最受欢迎菜品为 ${topDish?.dish_name || "无"}，评分 ${formatNumber(topDish?.avg_rating || 0, 1)}。`,
    `套餐搭配上，${bestRule?.antecedents_str || "暂无"} -> ${bestRule?.consequents_str || "暂无"} 的提升度表现最好。`,
    `学生群体中占比最高的是 ${bestCluster?.cluster_name || "未知群体"}。`,
    `建议围绕高峰时段提前备餐，并结合评分与销量变化进行菜单优化。`,
  ].join("\n");
}

export default function App() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [activeView, setActiveView] = useState("student-today");
  const [studentId, setStudentId] = useState("");
  const [filters, setFilters] = useState({
    budget: "10-15元",
    period: "午餐",
    taste: "不限",
    goal: "吃饱",
    canteen: "不限",
  });
  const [monthBudget, setMonthBudget] = useState(600);
  const [localPrefs, setLocalPrefs] = useLocalJsonState("canteen-ui-prefs", []);
  const [localFeedback, setLocalFeedback] = useLocalJsonState("canteen-ui-feedback", []);
  const [localVotes, setLocalVotes] = useLocalJsonState("canteen-ui-votes", []);

  useEffect(() => {
    async function loadData() {
      const response = await fetch("/dashboard-data.json");
      const json = await response.json();
      setData(json);
      setStudentId(json.meta.demo_student || json.meta.student_ids[0] || "");
      setLoading(false);
    }
    loadData();
  }, []);

  const deferredStudentId = useDeferredValue(studentId);

  const orders = data?.raw.orders || [];
  const currentStudent = useMemo(
    () => buildStudentSnapshot(orders, deferredStudentId),
    [orders, deferredStudentId],
  );
  const disliked = useMemo(
    () =>
      localPrefs
        .filter((item) => item.student_id === deferredStudentId && item.preference_type === "dislike")
        .map((item) => item.dish_name),
    [deferredStudentId, localPrefs],
  );
  const recommendation = useMemo(
    () => buildStudentRecommendation(orders, deferredStudentId, filters, disliked),
    [orders, deferredStudentId, filters, disliked],
  );
  const nutrition = useMemo(() => buildNutrition(currentStudent.list || []), [currentStudent]);

  const reportText = useMemo(() => (data ? buildReportText(data) : ""), [data]);

  if (loading) {
    return <div className="loading-shell">正在加载前端数据与分析结果...</div>;
  }

  const mergedFeedback = [...(data.student.feedback || []), ...localFeedback];
  const mergedVotes = [...(data.student.votes || []), ...localVotes];

  function selectView(key) {
    startTransition(() => setActiveView(key));
  }

  function savePreference(dishName, type) {
    setLocalPrefs((prev) => {
      const filtered = prev.filter((item) => !(item.student_id === studentId && item.dish_name === dishName));
      return [...filtered, { student_id: studentId, dish_name: dishName, preference_type: type, create_time: new Date().toISOString() }];
    });
  }

  function submitFeedback(payload) {
    setLocalFeedback((prev) => [...prev, payload]);
  }

  function submitVote(payload) {
    setLocalVotes((prev) => [...prev, payload]);
  }

  const view = {
    "student-today": (
      <>
        <section className="hero">
          <div>
            <span className="eyebrow">学生智能服务</span>
            <h1>今天吃什么，不再靠随机。</h1>
            <p>预算、口味、目标、时段、食堂位置一起参与推荐，让食堂选择更像一个真正会思考的服务系统。</p>
          </div>
          <div className="hero-blob">
            <span>实时推荐</span>
            <strong>{recommendation[0]?.dish_name || "午餐组合"}</strong>
            <em>{recommendation[0]?.reason || "根据历史偏好与热度生成"}</em>
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
            {["10元以内", "10-15元", "15-20元", "不限"].map((item) => (
              <option key={item}>{item}</option>
            ))}
          </select>
          <select value={filters.period} onChange={(e) => setFilters((prev) => ({ ...prev, period: e.target.value }))}>
            {data.meta.periods.map((item) => (
              <option key={item}>{item}</option>
            ))}
          </select>
          <select value={filters.taste} onChange={(e) => setFilters((prev) => ({ ...prev, taste: e.target.value }))}>
            {["不限", "清淡", "辣", "甜", "咸"].map((item) => (
              <option key={item}>{item}</option>
            ))}
          </select>
          <select value={filters.goal} onChange={(e) => setFilters((prev) => ({ ...prev, goal: e.target.value }))}>
            {["吃饱", "减脂", "高蛋白", "省钱", "尝鲜"].map((item) => (
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

        <Panel title="当前拥挤度" subtitle="按历史 15 分钟时隙估算当前排队压力">
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
    ),
    "student-report": (
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
                <Legend />
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
      </>
    ),
    "student-budget": (
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
          <MetricCard
            label="预算完成率"
            value={`${formatNumber((currentStudent.totalAmount / Math.max(monthBudget, 1)) * 100)} %`}
            tone="blue"
          />
          <MetricCard
            label="状态"
            value={currentStudent.totalAmount > monthBudget ? "超支预警" : "预算正常"}
            tone={currentStudent.totalAmount > monthBudget ? "berry" : "teal"}
          />
        </div>

        <Panel title="预算进度">
          <div className="progress-shell">
            <div
              className={`progress-bar ${currentStudent.totalAmount > monthBudget ? "danger" : ""}`}
              style={{ width: `${Math.min((currentStudent.totalAmount / Math.max(monthBudget, 1)) * 100, 100)}%` }}
            />
          </div>
          <p className="progress-copy">
            按当前节奏，建议后续优先选择 {currentStudent.avgTicket > 15 ? "10-15元以内" : "正常区间"} 的套餐。
          </p>
        </Panel>

        <Panel title="营养结构分析">
          <div className="two-col">
            <ResponsiveContainer width="100%" height={280}>
              <PieChart>
                <Pie data={nutrition} dataKey="quantity" nameKey="healthy_tag" outerRadius={100}>
                  {nutrition.map((entry, index) => (
                    <Cell key={entry.healthy_tag} fill={PIE_COLORS[index % PIE_COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
            <div className="advice-stack">
              {nutrition.map((item) => (
                <div className="advice-chip" key={item.healthy_tag}>
                  <strong>{item.healthy_tag}</strong>
                  <span>{formatNumber(item.ratio * 100)}%</span>
                </div>
              ))}
              <div className="health-note">
                {["油炸", "高糖", "重口"].includes(nutrition[0]?.healthy_tag)
                  ? "近期油炸/高糖/重口占比较高，建议增加清淡菜、汤品和蔬菜。"
                  : "当前饮食结构相对均衡，可以继续保持。"}
              </div>
            </div>
          </div>
        </Panel>
      </>
    ),
    "student-feedback": (
      <StudentFeedbackView
        studentId={studentId}
        localPrefs={localPrefs}
        savePreference={savePreference}
        feedback={mergedFeedback}
        submitFeedback={submitFeedback}
        votes={mergedVotes}
        submitVote={submitVote}
        dishes={Array.from(new Set(orders.map((item) => item.dish_name)))}
        canteens={data.meta.canteens}
      />
    ),
    "mgmt-dashboard": (
      <>
        <div className="metric-grid">
          <MetricCard label="总销售额" value={`${formatNumber(data.summary.kpis.total_sales)} 元`} tone="warm" />
          <MetricCard label="订单总数" value={`${formatNumber(data.summary.kpis.total_orders, 0)} 单`} tone="teal" />
          <MetricCard label="匿名学生数" value={`${formatNumber(data.summary.kpis.total_students, 0)} 人`} tone="blue" />
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
              { key: "recent_qty", label: "近7天销量" },
              { key: "decline_ratio", label: "变化率", render: (value) => `${formatNumber(value * 100)}%` },
              { key: "promo_type", label: "建议类型" },
              { key: "advice", label: "建议" },
            ]}
            rows={data.management.promotions}
          />
        </Panel>
      </>
    ),
    "mgmt-sales": (
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
      </>
    ),
    "mgmt-peak": (
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
          <Panel title="30分钟粒度高峰">
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
        <Panel title="星期-小时热力图">
          <HeatGrid data={data.management.heatmap} />
        </Panel>
      </>
    ),
    "mgmt-rules": (
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
      </>
    ),
    "mgmt-predict": (
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
      </>
    ),
    "mgmt-plan": (
      <>
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
      </>
    ),
    "ops-compare": (
      <>
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
        </div>
      </>
    ),
    "ops-report": (
      <>
        <Panel
          title="自动报告生成"
          subtitle="把经营指标、学生群体和备餐建议统一汇总为可直接写进课程报告的内容"
          right={
            <button
              className="ghost-button"
              onClick={() => {
                const blob = new Blob([reportText], { type: "text/plain;charset=utf-8" });
                const url = URL.createObjectURL(blob);
                const link = document.createElement("a");
                link.href = url;
                link.download = "canteen-auto-report.txt";
                link.click();
                URL.revokeObjectURL(url);
              }}
            >
              导出文本
            </button>
          }
        >
          <pre className="report-block">{reportText}</pre>
        </Panel>
      </>
    ),
  }[activeView];

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <Salad size={26} />
          <div>
            <strong>Campus Canteen BI</strong>
            <span>React Exhibition Layer</span>
          </div>
        </div>
        <div className="nav-groups">
          {NAV_GROUPS.map((group) => (
            <div className="nav-group" key={group.title}>
              <div className="nav-group-title">{group.title}</div>
              {group.items.map((item) => {
                const Icon = item.icon;
                return (
                  <button
                    className={`nav-item ${activeView === item.key ? "active" : ""}`}
                    key={item.key}
                    onClick={() => selectView(item.key)}
                  >
                    <Icon size={16} />
                    <span>{item.label}</span>
                  </button>
                );
              })}
            </div>
          ))}
        </div>
        <div className="sidebar-foot">
          <span>数据生成时间</span>
          <strong>{new Date(data.meta.generated_at).toLocaleString("zh-CN")}</strong>
        </div>
      </aside>
      <main className="content">
        {view}
      </main>
    </div>
  );
}

function StudentFeedbackView({
  studentId,
  localPrefs,
  savePreference,
  feedback,
  submitFeedback,
  votes,
  submitVote,
  dishes,
  canteens,
}) {
  const [selectedDish, setSelectedDish] = useState(dishes[0] || "");
  const [feedbackDish, setFeedbackDish] = useState(dishes[0] || "");
  const [feedbackCanteen, setFeedbackCanteen] = useState(canteens[0] || "");
  const [scores, setScores] = useState({ taste: 4, portion: 4, price: 4, service: 4 });
  const [comment, setComment] = useState("");
  const [voteDish, setVoteDish] = useState("重庆小面");
  const [voteReason, setVoteReason] = useState("");

  const myPrefs = localPrefs.filter((item) => item.student_id === studentId);
  const voteStats = Object.values(
    votes.reduce((acc, item) => {
      acc[item.dish_candidate] = acc[item.dish_candidate] || { dish_candidate: item.dish_candidate, count: 0 };
      acc[item.dish_candidate].count += 1;
      return acc;
    }, {}),
  ).sort((a, b) => b.count - a.count);

  return (
    <>
      <div className="three-col">
        <Panel title="偏好设置">
          <div className="stack-form">
            <select value={selectedDish} onChange={(e) => setSelectedDish(e.target.value)}>
              {dishes.map((dish) => (
                <option key={dish}>{dish}</option>
              ))}
            </select>
            <div className="button-row">
              <button className="primary-button" onClick={() => savePreference(selectedDish, "like")}>
                收藏菜品
              </button>
              <button className="ghost-button" onClick={() => savePreference(selectedDish, "dislike")}>
                标记不喜欢
              </button>
            </div>
            <div className="tag-list">
              {myPrefs.map((item, idx) => (
                <span className={`tag ${item.preference_type}`} key={`${item.dish_name}-${idx}`}>
                  {item.dish_name} · {item.preference_type}
                </span>
              ))}
            </div>
          </div>
        </Panel>

        <Panel title="评分反馈">
          <div className="stack-form">
            <select value={feedbackDish} onChange={(e) => setFeedbackDish(e.target.value)}>
              {dishes.map((dish) => (
                <option key={dish}>{dish}</option>
              ))}
            </select>
            <select value={feedbackCanteen} onChange={(e) => setFeedbackCanteen(e.target.value)}>
              {canteens.map((canteen) => (
                <option key={canteen}>{canteen}</option>
              ))}
            </select>
            {Object.entries(scores).map(([key, value]) => (
              <label className="slider-field" key={key}>
                <span>{key}</span>
                <input
                  type="range"
                  min="1"
                  max="5"
                  value={value}
                  onChange={(e) => setScores((prev) => ({ ...prev, [key]: Number(e.target.value) }))}
                />
                <strong>{value}</strong>
              </label>
            ))}
            <textarea value={comment} onChange={(e) => setComment(e.target.value)} placeholder="写下你的评价..." />
            <button
              className="primary-button"
              onClick={() => {
                submitFeedback({
                  student_id: studentId,
                  dish_name: feedbackDish,
                  canteen: feedbackCanteen,
                  taste_score: scores.taste,
                  portion_score: scores.portion,
                  price_score: scores.price,
                  service_score: scores.service,
                  comment,
                  create_time: new Date().toISOString(),
                });
                setComment("");
              }}
            >
              提交反馈
            </button>
          </div>
        </Panel>

        <Panel title="上新投票">
          <div className="stack-form">
            <select value={voteDish} onChange={(e) => setVoteDish(e.target.value)}>
              {["重庆小面", "烤盘饭", "轻食沙拉", "麻辣烫", "粤式烧腊", "韩式拌饭"].map((dish) => (
                <option key={dish}>{dish}</option>
              ))}
            </select>
            <textarea value={voteReason} onChange={(e) => setVoteReason(e.target.value)} placeholder="说说为什么想要这个新菜..." />
            <button
              className="primary-button"
              onClick={() => {
                submitVote({ student_id: studentId, dish_candidate: voteDish, reason: voteReason, vote_time: new Date().toISOString() });
                setVoteReason("");
              }}
            >
              提交投票
            </button>
            <MiniTable
              columns={[
                { key: "dish_candidate", label: "候选菜品" },
                { key: "count", label: "票数" },
              ]}
              rows={voteStats}
            />
          </div>
        </Panel>
      </div>

      <Panel title="最新反馈">
        <MiniTable
          columns={[
            { key: "dish_name", label: "菜品" },
            { key: "canteen", label: "食堂" },
            { key: "comment", label: "评论" },
            { key: "create_time", label: "时间", render: (value) => String(value).slice(0, 16).replace("T", " ") },
          ]}
          rows={feedback.slice(-10).reverse()}
        />
      </Panel>
    </>
  );
}
