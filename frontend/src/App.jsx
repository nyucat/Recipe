import React, { startTransition, useEffect, useMemo, useState, useDeferredValue } from "react";
import { Salad } from "lucide-react";

import { DEFAULT_FILTERS, NAV_GROUPS } from "./constants";
import { useLocalJsonState } from "./hooks";
import {
  buildBudgetAdvice,
  buildPredictionMomentum,
  buildReportText,
  buildStudentRecommendation,
  buildStudentSnapshot,
  buildNutrition,
  buildWindowSummary,
  budgetLabelToRange,
} from "./utils";
import {
  ManagementDashboardView,
  ManagementPeakView,
  ManagementPlanView,
  ManagementPredictView,
  ManagementRulesView,
  ManagementSalesView,
  OpsCompareView,
  OpsReportView,
  StudentBudgetView,
  StudentFeedbackView,
  StudentReportView,
  StudentTodayView,
} from "./views";

export default function App() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [activeView, setActiveView] = useState("student-today");
  const [studentId, setStudentId] = useState("");
  const [filters, setFilters] = useState(DEFAULT_FILTERS);
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
  const budgetAdvice = useMemo(
    () => buildBudgetAdvice(currentStudent.totalAmount, monthBudget),
    [currentStudent.totalAmount, monthBudget],
  );
  const studentFeature = useMemo(
    () => data?.logistics.studentFeatures.find((item) => item.student_id === deferredStudentId),
    [data, deferredStudentId],
  );
  const comboSuggestions = useMemo(() => {
    const [_, maxBudget] = budgetLabelToRange(filters.budget);
    return (data?.management.comboSuggestions || [])
      .filter((item) => Number(item.estimated_price || 0) <= maxBudget)
      .slice(0, 6);
  }, [data, filters.budget]);
  const windowSummary = useMemo(() => buildWindowSummary(data?.management.mealPlan || []), [data]);
  const predictionMomentum = useMemo(() => buildPredictionMomentum(data?.management.futurePrediction || []), [data]);
  const reportText = useMemo(() => (data ? buildReportText(data) : ""), [data]);

  const mergedFeedback = useMemo(() => [...(data?.student.feedback || []), ...localFeedback], [data, localFeedback]);
  const mergedVotes = useMemo(() => [...(data?.student.votes || []), ...localVotes], [data, localVotes]);

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

  if (loading) {
    return <div className="loading-shell">正在加载前端数据与分析结果...</div>;
  }

  const allDishes = [...new Set(orders.map((item) => item.dish_name))].sort();
  const allCanteens = [...new Set(orders.map((item) => item.canteen))].sort();

  const views = {
    "student-today": (
      <StudentTodayView
        data={data}
        studentId={studentId}
        setStudentId={setStudentId}
        filters={filters}
        setFilters={setFilters}
        recommendation={recommendation}
        comboSuggestions={comboSuggestions}
      />
    ),
    "student-report": (
      <StudentReportView
        currentStudent={currentStudent}
        studentFeature={studentFeature}
        peerRecommendation={data.student.peerRecommendation}
      />
    ),
    "student-budget": (
      <StudentBudgetView
        monthBudget={monthBudget}
        setMonthBudget={setMonthBudget}
        currentStudent={currentStudent}
        budgetAdvice={budgetAdvice}
        nutrition={nutrition}
      />
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
        dishes={allDishes}
        canteens={allCanteens}
      />
    ),
    "mgmt-dashboard": <ManagementDashboardView data={data} />,
    "mgmt-sales": <ManagementSalesView data={data} />,
    "mgmt-peak": <ManagementPeakView data={data} />,
    "mgmt-rules": <ManagementRulesView data={data} />,
    "mgmt-predict": <ManagementPredictView data={data} predictionMomentum={predictionMomentum} />,
    "mgmt-plan": <ManagementPlanView data={data} windowSummary={windowSummary} />,
    "ops-compare": <OpsCompareView data={data} />,
    "ops-report": <OpsReportView reportText={reportText} data={data} />,
  };

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
      <main className="content">{views[activeView]}</main>
    </div>
  );
}
