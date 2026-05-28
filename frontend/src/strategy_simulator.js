export const DEFAULT_STRATEGY = {
  canteen: "",
  priceDelta: 0,
  discountRate: 1,
  staffingDelta: 0,
  prepFactor: 1.1,
};

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

export function simulateStrategy(base, strategy) {
  if (!base) return null;

  const priceDelta = Number(strategy.priceDelta || 0);
  const discountRate = Number(strategy.discountRate || 1);
  const staffingDelta = Number(strategy.staffingDelta || 0);
  const prepFactor = Number(strategy.prepFactor || 1.1);

  const effectivePriceFactor = (1 + priceDelta / Math.max(base.avg_ticket || 12, 1)) * discountRate;
  const orderChange =
    (-priceDelta / Math.max(base.avg_ticket || 12, 1)) * 0.35 +
    (1 - discountRate) * 0.9 +
    staffingDelta * 0.08 +
    (prepFactor - 1.1) * 0.25;
  const orderFactor = clamp(1 + orderChange, 0.65, 1.45);

  const simulatedOrders = base.orders * orderFactor;
  const simulatedSales = simulatedOrders * (base.avg_ticket + priceDelta) * discountRate;

  const queueMinutes = clamp(base.queue_minutes - staffingDelta * 2.2 - (prepFactor - 1.1) * 8, 2, 28);
  const shortageCount = Math.max(0, Math.round(base.shortage_count - staffingDelta * 1.1 - (prepFactor - 1.1) * 16));
  const wasteCount = Math.max(0, Math.round(base.waste_count + (prepFactor - 1.1) * 15 - staffingDelta * 0.4));

  const ratingShift =
    staffingDelta * 0.06 -
    Math.max(priceDelta, 0) * 0.04 +
    (1 - discountRate) * 0.08 -
    Math.max(queueMinutes - base.queue_minutes, 0) * 0.01;
  const simulatedRating = clamp((base.rating || 4) + ratingShift, 3.2, 4.9);

  const costFactor = 0.62 + staffingDelta * 0.015 + Math.max(1.05 - prepFactor, 0) * 0.03;
  const simulatedCost = simulatedSales * costFactor;
  const simulatedProfit = simulatedSales - simulatedCost;

  const summary = buildStrategySummary({
    base,
    simulatedSales,
    simulatedOrders,
    simulatedProfit,
    simulatedRating,
    queueMinutes,
    shortageCount,
    wasteCount,
    strategy,
  });

  return {
    sales: simulatedSales,
    orders: simulatedOrders,
    profit: simulatedProfit,
    rating: simulatedRating,
    queueMinutes,
    shortageCount,
    wasteCount,
    summary,
    deltas: {
      sales: simulatedSales - base.sales,
      orders: simulatedOrders - base.orders,
      profit: simulatedProfit - base.baseline_profit,
      rating: simulatedRating - base.rating,
      queueMinutes: queueMinutes - base.queue_minutes,
    },
    assumptions: [
      "折扣和降价会提升订单量，但可能压缩利润。",
      "增加人手可缩短排队时间，并缓解缺货风险。",
      "提高备餐系数有助于防缺货，但会增加浪费概率。",
    ],
  };
}

function buildStrategySummary({ base, simulatedSales, simulatedOrders, simulatedProfit, simulatedRating, queueMinutes, shortageCount, wasteCount, strategy }) {
  const lines = [];
  if (simulatedSales > base.sales) lines.push(`预计销售额提升 ${(simulatedSales - base.sales).toFixed(1)} 元。`);
  else lines.push(`预计销售额下降 ${Math.abs(simulatedSales - base.sales).toFixed(1)} 元。`);

  if (queueMinutes < base.queue_minutes) lines.push(`预计平均排队时间缩短约 ${(base.queue_minutes - queueMinutes).toFixed(1)} 分钟。`);
  if (shortageCount < base.shortage_count) lines.push("缺货风险得到缓解。");
  if (wasteCount > base.waste_count) lines.push("但浪费风险会上升，需要关注备餐冗余。");
  if (simulatedRating < base.rating) lines.push("该策略可能对学生满意度产生轻微负面影响。");

  if (Number(strategy.discountRate) < 1) lines.push("适合用于短期拉动订单和测试促销敏感度。");
  if (Number(strategy.staffingDelta) > 0) lines.push("更适合高峰期窗口优化和服务体验改善。");
  if (Number(strategy.prepFactor) > 1.15) lines.push("备餐系数偏高，建议同步搭配套餐促销减少浪费。");
  if (simulatedProfit < base.baseline_profit) lines.push("利润可能下滑，建议控制活动持续时间。");
  else lines.push(`预计利润变动为 ${(simulatedProfit - base.baseline_profit).toFixed(1)} 元。`);

  return lines.slice(0, 6);
}
