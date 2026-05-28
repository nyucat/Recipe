import {
  Activity,
  BarChart3,
  BookOpenText,
  ChefHat,
  CircleGauge,
  ClipboardList,
  HeartPulse,
  LineChart as LineChartIcon,
  MessageSquareText,
  Sparkles,
  Store,
  Users,
} from "lucide-react";

export const NAV_GROUPS = [
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

export const PIE_COLORS = ["#f97316", "#14b8a6", "#0ea5e9", "#8b5cf6", "#f43f5e", "#84cc16"];

export const DISH_TASTE = {
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

export const GOAL_MAP = {
  吃饱: new Set(["主食", "热菜"]),
  减脂: new Set(["轻食", "汤品"]),
  高蛋白: new Set(["热菜", "主食"]),
  省钱: new Set(["主食", "汤品"]),
  尝鲜: new Set(["饮品", "小吃", "轻食"]),
};

export const DEFAULT_FILTERS = {
  budget: "10-15元",
  period: "午餐",
  taste: "不限",
  goal: "吃饱",
  canteen: "不限",
};

export const BUDGET_OPTIONS = ["10元以内", "10-15元", "15-20元", "不限"];
export const TASTE_OPTIONS = ["不限", "清淡", "辣", "甜", "咸"];
export const GOAL_OPTIONS = ["吃饱", "减脂", "高蛋白", "省钱", "尝鲜"];
export const NEW_DISH_OPTIONS = ["重庆小面", "烤盘饭", "轻食沙拉", "麻辣烫", "粤式烧腊", "韩式拌饭"];
