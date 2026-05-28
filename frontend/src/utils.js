import { DISH_TASTE, GOAL_MAP } from "./constants";

export function formatNumber(value, digits = 1) {
  return Number(value || 0).toLocaleString("zh-CN", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

export function budgetLabelToRange(label) {
  const budgetRanges = {
    "10元以内": [0, 10],
    "10-15元": [10, 15],
    "15-20元": [15, 20],
    不限: [0, 999],
  };
  return budgetRanges[label] || budgetRanges["不限"];
}

export function buildStudentRecommendation(orders, studentId, filters, disliked) {
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

  const [minBudget, maxBudget] = budgetLabelToRange(filters.budget);
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
        `${filters.period}时段更常被选择`,
      ];
      if (favoriteCategory && item.category === favoriteCategory) reasons.push("符合你的历史偏好");
      if (GOAL_MAP[filters.goal]?.has(item.category)) reasons.push(`匹配“${filters.goal}”目标`);

      return { ...item, score, reason: reasons.join("，") };
    })
    .sort((a, b) => b.score - a.score)
    .slice(0, 6);
}

export function buildStudentSnapshot(orders, studentId) {
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

export function buildNutrition(studentOrders) {
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

export function buildBudgetAdvice(totalAmount, monthBudget) {
  const remaining = monthBudget - totalAmount;
  const now = new Date();
  const day = now.getDate();
  const daysInMonth = new Date(now.getFullYear(), now.getMonth() + 1, 0).getDate();
  const projected = (totalAmount / Math.max(day, 1)) * daysInMonth;
  return {
    remaining,
    projected,
    dailyTarget: Math.max(remaining, 0) / Math.max(daysInMonth - day, 1),
  };
}

export function buildWindowSummary(mealPlan) {
  return Object.values(
    mealPlan.reduce((acc, item) => {
      acc[item.window] = acc[item.window] || { window: item.window, pred_total: 0, prep_total: 0, high_risk: 0 };
      acc[item.window].pred_total += Number(item.pred_quantity || 0);
      acc[item.window].prep_total += Number(item.suggested_prep || 0);
      if (item.risk === "缺货风险较高") acc[item.window].high_risk += 1;
      return acc;
    }, {}),
  ).sort((a, b) => b.pred_total - a.pred_total);
}

export function buildPredictionMomentum(futurePrediction) {
  const grouped = Object.values(
    futurePrediction.reduce((acc, item) => {
      acc[item.dish_name] = acc[item.dish_name] || { dish_name: item.dish_name, first: null, last: null };
      if (acc[item.dish_name].first === null) acc[item.dish_name].first = Number(item.pred_quantity || 0);
      acc[item.dish_name].last = Number(item.pred_quantity || 0);
      return acc;
    }, {}),
  ).map((item) => ({ ...item, delta: item.last - item.first }));
  return grouped.sort((a, b) => b.delta - a.delta);
}

export function buildReportText(data) {
  const kpis = data.summary.kpis;
  const bestRule = data.management.rules[0];
  const bestCluster = data.logistics.clusterProfile[0];
  const topDish = data.management.hotDishes[0];
  return [
    "校园食堂智能服务系统自动报告",
    "",
    `总体销售额：${formatNumber(kpis.total_sales)} 元，订单数：${formatNumber(kpis.total_orders, 0)} 单，消费学生：${formatNumber(kpis.total_students, 0)} 人。`,
    `当前最受欢迎菜品：${topDish?.dish_name || "暂无"}，平均评分 ${formatNumber(topDish?.avg_rating || 0, 1)}。`,
    `套餐关联表现最好的组合是 ${bestRule?.antecedents_str || "暂无"} -> ${bestRule?.consequents_str || "暂无"}。`,
    `当前占比最高的学生群体为 ${bestCluster?.cluster_name || "未知群体"}。`,
    "建议围绕高峰时段提前备餐，并结合评分变化、销量波动和学生偏好持续优化菜品结构。",
  ].join("\n");
}
