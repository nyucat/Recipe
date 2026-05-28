import React, { useMemo, useState } from "react";
import { Cell, Legend, Pie, PieChart, ResponsiveContainer, Tooltip } from "recharts";

import { NEW_DISH_OPTIONS, PIE_COLORS } from "./constants";
import { formatNumber } from "./utils";

export function MetricCard({ label, value, tone = "warm", hint }) {
  return (
    <div className={`metric-card tone-${tone}`}>
      <div className="metric-label">{label}</div>
      <div className="metric-value">{value}</div>
      {hint ? <div className="metric-hint">{hint}</div> : null}
    </div>
  );
}

export function Panel({ title, subtitle, right, children }) {
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

export function MiniTable({ columns, rows }) {
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

export function RecommendationCard({ item }) {
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

export function HeatGrid({ data }) {
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

export function NutritionPie({ nutrition }) {
  return (
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
  );
}

export function StudentFeedbackView({
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
  const [voteDish, setVoteDish] = useState(NEW_DISH_OPTIONS[0]);
  const [voteReason, setVoteReason] = useState("");

  const myPrefs = useMemo(
    () => localPrefs.filter((item) => item.student_id === studentId),
    [localPrefs, studentId],
  );
  const voteStats = useMemo(
    () =>
      Object.values(
        votes.reduce((acc, item) => {
          acc[item.dish_candidate] = acc[item.dish_candidate] || { dish_candidate: item.dish_candidate, count: 0 };
          acc[item.dish_candidate].count += 1;
          return acc;
        }, {}),
      ).sort((a, b) => b.count - a.count),
    [votes],
  );

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
                  {item.dish_name} · {item.preference_type === "like" ? "喜欢" : "不喜欢"}
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
              {NEW_DISH_OPTIONS.map((dish) => (
                <option key={dish}>{dish}</option>
              ))}
            </select>
            <textarea value={voteReason} onChange={(e) => setVoteReason(e.target.value)} placeholder="说说你为什么想要这道新菜..." />
            <button
              className="primary-button"
              onClick={() => {
                submitVote({
                  student_id: studentId,
                  dish_candidate: voteDish,
                  reason: voteReason,
                  vote_time: new Date().toISOString(),
                });
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
