import React, { useState, useEffect, useCallback, useRef } from "react";
import {
  AreaChart, Area, LineChart, Line,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from "recharts";

const API = "https://energy-prediction-system.onrender.com";

/* ── Design tokens ─────────────────────────────────────────────────────────── */
const T = {
  bg:      "#060910",
  surface: "#0d1117",
  card:    "#111820",
  border:  "#1e2d3d",
  accent:  "#00d4ff",
  green:   "#00ff9d",
  amber:   "#ffb800",
  red:     "#ff4466",
  purple:  "#b060ff",
  text:    "#e8f4f8",
  muted:   "#4a6070",
  dim:     "#1a2535",
};

/* ── Inject global styles ───────────────────────────────────────────────────── */
const STYLE = `
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  html, body, #root { height: 100%; }
  body { background: ${T.bg}; color: ${T.text}; font-family: 'Syne', sans-serif; overflow-x: hidden; }
  ::-webkit-scrollbar { width: 4px; } ::-webkit-scrollbar-track { background: ${T.bg}; } ::-webkit-scrollbar-thumb { background: ${T.border}; border-radius: 2px; }
  .mono { font-family: 'Space Mono', monospace; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }
  @keyframes scan { 0%{transform:translateY(-100%)} 100%{transform:translateY(100vh)} }
  @keyframes fadeUp { from{opacity:0;transform:translateY(20px)} to{opacity:1;transform:translateY(0)} }
  @keyframes glow { 0%,100%{box-shadow:0 0 8px ${T.accent}44} 50%{box-shadow:0 0 24px ${T.accent}88} }
  .fade-up { animation: fadeUp 0.5s ease forwards; }
  .glow-ring { animation: glow 2s ease-in-out infinite; }
`;

function injectStyle(css) {
  const el = document.createElement("style");
  el.textContent = css;
  document.head.appendChild(el);
}

/* ── Scanline background ────────────────────────────────────────────────────── */
function ScanBg() {
  return (
    <div style={{ position: "fixed", inset: 0, pointerEvents: "none", zIndex: 0, overflow: "hidden" }}>
      <div style={{ position: "absolute", inset: 0, background: `repeating-linear-gradient(0deg, transparent, transparent 2px, ${T.accent}04 2px, ${T.accent}04 4px)` }} />
      <div style={{ position: "absolute", width: "100%", height: 2, background: `linear-gradient(90deg, transparent, ${T.accent}22, transparent)`, animation: "scan 8s linear infinite" }} />
    </div>
  );
}

/* ── Live dot ───────────────────────────────────────────────────────────────── */
function LiveDot({ color = T.green }) {
  return <span style={{ display: "inline-block", width: 8, height: 8, borderRadius: "50%", background: color, animation: "pulse 1.5s ease-in-out infinite", marginRight: 6 }} />;
}

/* ── Stat card ──────────────────────────────────────────────────────────────── */
function StatCard({ label, value, unit, color, delay = 0 }) {
  return (
    <div className="fade-up" style={{
      animationDelay: delay + "ms",
      background: T.card,
      border: "1px solid " + (color + "44"),
      borderRadius: 12,
      padding: "20px 24px",
      position: "relative",
      overflow: "hidden",
    }}>
      <div style={{ position: "absolute", top: 0, left: 0, width: "100%", height: 2, background: `linear-gradient(90deg, transparent, ${color}, transparent)` }} />
      <div style={{ color: T.muted, fontSize: 11, letterSpacing: 2, textTransform: "uppercase", marginBottom: 8 }}>{label}</div>
      <div className="mono" style={{ color: color, fontSize: 28, fontWeight: 700, lineHeight: 1 }}>
        {value}<span style={{ fontSize: 14, color: T.muted, marginLeft: 4 }}>{unit}</span>
      </div>
    </div>
  );
}

/* ── Section label ──────────────────────────────────────────────────────────── */
function SLabel({ children }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 20 }}>
      <div style={{ width: 3, height: 20, background: T.accent, borderRadius: 2 }} />
      <span style={{ fontSize: 13, letterSpacing: 3, textTransform: "uppercase", color: T.muted }}>{children}</span>
    </div>
  );
}

/* ── Custom tooltip ─────────────────────────────────────────────────────────── */
function ChartTip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="mono" style={{ background: T.surface, border: "1px solid " + T.border, borderRadius: 8, padding: "10px 14px", fontSize: 12 }}>
      <div style={{ color: T.muted, marginBottom: 4 }}>{label}</div>
      {payload.map((p, i) => (
        <div key={i} style={{ color: p.color || T.accent }}>{p.name}: {typeof p.value === "number" ? p.value.toFixed(4) : p.value}</div>
      ))}
    </div>
  );
}

/* ── Main App ───────────────────────────────────────────────────────────────── */
export default function App() {
  const [tab, setTab]             = useState("forecast");
  const [health, setHealth]       = useState(null);
  const [availableModels, setAvailableModels] = useState([]);
  const [metrics, setMetrics]     = useState([]);
  const [forecast, setForecast]   = useState([]);
  const [horizon, setHorizon]     = useState(24);
  const [forecastModel, setForecastModel] = useState("SARIMA");
  const [temperature, setTemperature] = useState(0.5);
  const [humidity, setHumidity] = useState(0.5);
  const [windSpeed, setWindSpeed] = useState(0.3);
  const [startTime, setStartTime] = useState("");
  const [anomalies, setAnomalies] = useState({});
  const [suggestions, setSuggestions] = useState([]);
  const [loading, setLoading]     = useState({});
  const [error, setError]         = useState(null);
  const [time, setTime]           = useState(new Date());
  const initialized               = useRef(false);

  useEffect(() => {
    if (!initialized.current) { injectStyle(STYLE); initialized.current = true; }
    const tick = setInterval(() => setTime(new Date()), 1000);
    fetch(API + "/health")
      .then(r => r.json())
      .then(d => {
        setHealth(d);
        setAvailableModels(d.available_models || []);
        if (d.best_model) setForecastModel(d.best_model);
      })
      .catch(() => setHealth({ status: "unreachable" }));
    fetch(API + "/model/info").then(r => r.json()).then(d => setMetrics(d.metrics || [])).catch(() => {});
    return () => clearInterval(tick);
  }, []);

  const fetchForecast = useCallback(async () => {
    setLoading(l => ({ ...l, forecast: true })); setError(null);
    try {
      const payload = { horizon, model_name: forecastModel };

      if (forecastModel === "LSTM") {
        payload.temperature = temperature;
        payload.humidity = humidity;
        payload.wind_speed = windSpeed;
        payload.start_time = startTime ? new Date(startTime).toISOString() : new Date().toISOString();
      }

      const res = await fetch(API + "/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
      const d = await res.json();
      if (!res.ok) {
        throw new Error(d?.detail || d?.message || "Forecast request failed.");
      }
      setForecast((d.predictions || []).map((p, i) => ({
        time: p.timestamp ? new Date(p.timestamp).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }) : "+" + (i + 1) + "h",
        kwh: parseFloat(p.predicted_kwh.toFixed(4))
      })));
      if (d.model_used) setForecastModel(d.model_used);
    } catch (err) {
      setError(err?.message || "Cannot reach API. Make sure api.py is running.");
    }
    setLoading(l => ({ ...l, forecast: false }));
  }, [forecastModel, horizon, humidity, startTime, temperature, windSpeed]);

  const fetchAnomalies = useCallback(async () => {
    setLoading(l => ({ ...l, anomaly: true }));
    const readings = Array.from({ length: 48 }, (_, i) =>
      0.4 + 0.15 * Math.sin(i * Math.PI / 12) + (i === 20 || i === 35 ? 1.8 : 0) + Math.random() * 0.05
    );
    try {
      const res = await fetch(API + "/anomalies", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ readings }) });
      const d = await res.json();
      setAnomalies({
        total: d.total_readings, count: d.anomalies_detected, items: d.anomalies || [],
        chartData: readings.map((v, i) => ({ idx: i, value: parseFloat(v.toFixed(4)), anomaly: (d.anomalies || []).some(a => Math.abs(a.value - v) < 0.01) ? v : null })),
      });
    } catch { setError("Cannot reach API."); }
    setLoading(l => ({ ...l, anomaly: false }));
  }, []);

  const fetchOptimize = useCallback(async () => {
    setLoading(l => ({ ...l, opt: true }));
    const hourlyAvg = Array.from({ length: 24 }, (_, h) => h >= 18 && h <= 21 ? 0.85 + Math.random() * 0.1 : 0.35 + Math.random() * 0.05);
    try {
      const res = await fetch(API + "/optimize", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ hourly_avg: hourlyAvg, anomaly_count: 2, has_weekend_spike: true }) });
      const d = await res.json();
      setSuggestions(d.suggestions || []);
    } catch { setError("Cannot reach API."); }
    setLoading(l => ({ ...l, opt: false }));
  }, []);

  const isLive = health?.status === "ok";
  const tabs = [
    { id: "forecast", label: "FORECAST" },
    { id: "anomaly",  label: "ANOMALIES" },
    { id: "optimize", label: "OPTIMIZE" },
  ];

  return (
    <div style={{ minHeight: "100vh", position: "relative", zIndex: 1 }}>
      <ScanBg />

      {/* ── Top bar ── */}
      <div style={{ position: "sticky", top: 0, zIndex: 100, background: T.bg + "ee", backdropFilter: "blur(12px)", borderBottom: "1px solid " + T.border, padding: "0 32px" }}>
        <div style={{ maxWidth: 1280, margin: "0 auto", height: 64, display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
            <div style={{ width: 32, height: 32, border: "2px solid " + T.accent, borderRadius: 8, display: "grid", placeItems: "center" }}>
              <div style={{ width: 12, height: 12, background: T.accent, borderRadius: 2 }} />
            </div>
            <div>
              <div style={{ fontSize: 16, fontWeight: 800, letterSpacing: 1 }}>ENERGY<span style={{ color: T.accent }}>_</span>OS</div>
              <div className="mono" style={{ fontSize: 10, color: T.muted, letterSpacing: 2 }}>FORECASTING SYSTEM v1.0</div>
            </div>
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: 32 }}>
            <div className="mono" style={{ fontSize: 13, color: T.muted }}>{time.toLocaleTimeString()}</div>
            <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "6px 14px", borderRadius: 20, background: isLive ? T.green + "15" : T.red + "15", border: "1px solid " + (isLive ? T.green + "44" : T.red + "44") }}>
              <LiveDot color={isLive ? T.green : T.red} />
              <span className="mono" style={{ fontSize: 11, color: isLive ? T.green : T.red, letterSpacing: 1 }}>
                {isLive ? health.best_model + " ONLINE" : "API OFFLINE"}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* ── Main content ── */}
      <div style={{ maxWidth: 1280, margin: "0 auto", padding: "40px 32px" }}>

        {/* Error */}
        {error && (
          <div style={{ background: T.red + "15", border: "1px solid " + T.red + "44", borderRadius: 10, padding: "12px 18px", marginBottom: 24, color: T.red, fontSize: 13 }} className="mono">
            ERR / {error}
          </div>
        )}

        {/* ── Stat row ── */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: 16, marginBottom: 40 }}>
          <StatCard label="Active Model"   value={health?.best_model || "—"} unit=""     color={T.accent}  delay={0} />
          <StatCard label="Lookback"       value={health?.lookback   || "—"} unit="hrs"  color={T.purple}  delay={80} />
          {metrics[0] && <StatCard label="SARIMA RMSE" value={typeof metrics[0].RMSE === "number" ? metrics[0].RMSE.toFixed(3) : "—"} unit="" color={T.amber} delay={160} />}
          {metrics[1] && <StatCard label="LSTM RMSE"   value={typeof metrics[1].RMSE === "number" ? metrics[1].RMSE.toFixed(3) : "—"} unit="" color={T.green} delay={240} />}
        </div>

        {/* ── Tabs ── */}
        <div style={{ display: "flex", gap: 4, marginBottom: 32, borderBottom: "1px solid " + T.border, paddingBottom: 0 }}>
          {tabs.map(t => (
            <button key={t.id} onClick={() => setTab(t.id)} style={{
              padding: "12px 24px", background: "none", border: "none", cursor: "pointer",
              fontSize: 12, fontWeight: 700, letterSpacing: 2, fontFamily: "'Space Mono', monospace",
              color: tab === t.id ? T.accent : T.muted,
              borderBottom: "2px solid " + (tab === t.id ? T.accent : "transparent"),
              marginBottom: -1, transition: "all 0.2s",
            }}>{t.label}</button>
          ))}
        </div>

        {/* ── FORECAST TAB ── */}
        {tab === "forecast" && (
          <div className="fade-up">
            <SLabel>Energy Consumption Forecast</SLabel>
            <div style={{ background: T.card, border: "1px solid " + T.border, borderRadius: 16, padding: 28, marginBottom: 24 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 16, marginBottom: 20, flexWrap: "wrap" }}>
                <span className="mono" style={{ fontSize: 12, color: T.muted, letterSpacing: 1 }}>MODEL /</span>
                <select value={forecastModel} onChange={e => setForecastModel(e.target.value)} style={{
                  background: T.surface, color: T.text, border: "1px solid " + T.border,
                  borderRadius: 8, padding: "8px 12px", fontFamily: "'Space Mono', monospace",
                  fontSize: 12, outline: "none"
                }}>
                  <option value="SARIMA" disabled={availableModels.length > 0 && !availableModels.includes("SARIMA")}>SARIMA</option>
                  <option value="LSTM" disabled={availableModels.length > 0 && !availableModels.includes("LSTM")}>LSTM</option>
                </select>
                {availableModels.length > 0 && !availableModels.includes(forecastModel) && (
                  <span className="mono" style={{ color: T.amber, fontSize: 11 }}>Selected model artifact is not loaded on the backend.</span>
                )}
              </div>

              {forecastModel === "LSTM" && (
                <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: 12, marginBottom: 20 }}>
                  <label style={{ display: "grid", gap: 8 }}>
                    <span className="mono" style={{ fontSize: 10, color: T.muted, letterSpacing: 2 }}>START DATE / TIME</span>
                    <input type="datetime-local" value={startTime} onChange={e => setStartTime(e.target.value)} style={{ background: T.surface, color: T.text, border: "1px solid " + T.border, borderRadius: 8, padding: "10px 12px", fontFamily: "'Space Mono', monospace", fontSize: 12 }} />
                  </label>
                  <label style={{ display: "grid", gap: 8 }}>
                    <span className="mono" style={{ fontSize: 10, color: T.muted, letterSpacing: 2 }}>TEMPERATURE</span>
                    <input type="number" step="0.01" min="0" max="1" value={temperature} onChange={e => setTemperature(parseFloat(e.target.value))} style={{ background: T.surface, color: T.text, border: "1px solid " + T.border, borderRadius: 8, padding: "10px 12px", fontFamily: "'Space Mono', monospace", fontSize: 12 }} />
                  </label>
                  <label style={{ display: "grid", gap: 8 }}>
                    <span className="mono" style={{ fontSize: 10, color: T.muted, letterSpacing: 2 }}>HUMIDITY</span>
                    <input type="number" step="0.01" min="0" max="1" value={humidity} onChange={e => setHumidity(parseFloat(e.target.value))} style={{ background: T.surface, color: T.text, border: "1px solid " + T.border, borderRadius: 8, padding: "10px 12px", fontFamily: "'Space Mono', monospace", fontSize: 12 }} />
                  </label>
                  <label style={{ display: "grid", gap: 8 }}>
                    <span className="mono" style={{ fontSize: 10, color: T.muted, letterSpacing: 2 }}>WIND SPEED</span>
                    <input type="number" step="0.01" min="0" max="1" value={windSpeed} onChange={e => setWindSpeed(parseFloat(e.target.value))} style={{ background: T.surface, color: T.text, border: "1px solid " + T.border, borderRadius: 8, padding: "10px 12px", fontFamily: "'Space Mono', monospace", fontSize: 12 }} />
                  </label>
                </div>
              )}

              <div style={{ display: "flex", alignItems: "center", gap: 16, marginBottom: 28, flexWrap: "wrap" }}>
                <span className="mono" style={{ fontSize: 12, color: T.muted, letterSpacing: 1 }}>HORIZON /</span>
                <div style={{ display: "flex", gap: 8 }}>
                  {[1, 6, 12, 24, 48, 72].map(h => (
                    <button key={h} onClick={() => setHorizon(h)} style={{
                      padding: "6px 14px", borderRadius: 6, border: "1px solid " + (horizon === h ? T.accent : T.border),
                      background: horizon === h ? T.accent + "22" : "transparent",
                      color: horizon === h ? T.accent : T.muted,
                      cursor: "pointer", fontSize: 12, fontFamily: "'Space Mono', monospace",
                    }}>{h}h</button>
                  ))}
                </div>
                <button onClick={fetchForecast} disabled={loading.forecast} className="glow-ring" style={{
                  marginLeft: "auto", padding: "10px 28px", borderRadius: 8,
                  background: loading.forecast ? T.dim : T.accent,
                  color: loading.forecast ? T.muted : T.bg,
                  border: "none", cursor: loading.forecast ? "not-allowed" : "pointer",
                  fontWeight: 700, fontSize: 13, fontFamily: "'Space Mono', monospace", letterSpacing: 1,
                }}>
                  {loading.forecast ? "LOADING..." : "RUN FORECAST"}
                </button>
              </div>

              {forecast.length > 0 ? (
                <ResponsiveContainer width="100%" height={320}>
                  <AreaChart data={forecast} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
                    <defs>
                      <linearGradient id="accentGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor={T.accent} stopOpacity={0.3} />
                        <stop offset="100%" stopColor={T.accent} stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="1 4" stroke={T.border} />
                    <XAxis dataKey="time" tick={{ fill: T.muted, fontSize: 11, fontFamily: "Space Mono" }} interval={Math.floor(horizon / 6)} axisLine={false} tickLine={false} />
                    <YAxis tick={{ fill: T.muted, fontSize: 11, fontFamily: "Space Mono" }} axisLine={false} tickLine={false} />
                    <Tooltip content={<ChartTip />} />
                    <Area type="monotone" dataKey="kwh" stroke={T.accent} fill="url(#accentGrad)" strokeWidth={2} dot={false} name="kWh" />
                  </AreaChart>
                </ResponsiveContainer>
              ) : (
                <div style={{ height: 320, display: "grid", placeItems: "center" }}>
                  <div style={{ textAlign: "center" }}>
                    <div style={{ fontSize: 48, marginBottom: 16, opacity: 0.3 }}>⚡</div>
                    <div className="mono" style={{ color: T.muted, fontSize: 12, letterSpacing: 2 }}>SELECT HORIZON AND RUN FORECAST</div>
                  </div>
                </div>
              )}
            </div>

            {forecast.length > 0 && (
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))", gap: 12 }}>
                {[
                  { label: "Peak", value: Math.max(...forecast.map(f => f.kwh)).toFixed(4), color: T.red },
                  { label: "Min", value: Math.min(...forecast.map(f => f.kwh)).toFixed(4), color: T.green },
                  { label: "Average", value: (forecast.reduce((s, f) => s + f.kwh, 0) / forecast.length).toFixed(4), color: T.accent },
                  { label: "Total kWh", value: forecast.reduce((s, f) => s + f.kwh, 0).toFixed(3), color: T.amber },
                ].map((s, i) => (
                  <div key={i} style={{ background: T.card, border: "1px solid " + T.border, borderRadius: 10, padding: "14px 18px" }}>
                    <div style={{ color: T.muted, fontSize: 10, letterSpacing: 2, textTransform: "uppercase", marginBottom: 6 }}>{s.label}</div>
                    <div className="mono" style={{ color: s.color, fontSize: 18, fontWeight: 700 }}>{s.value}</div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* ── ANOMALY TAB ── */}
        {tab === "anomaly" && (
          <div className="fade-up">
            <SLabel>Anomaly Detection — Z-Score Analysis</SLabel>
            <div style={{ background: T.card, border: "1px solid " + T.border, borderRadius: 16, padding: 28, marginBottom: 24 }}>
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 24, flexWrap: "wrap", gap: 12 }}>
                <div className="mono" style={{ fontSize: 12, color: T.muted }}>48-POINT SYNTHETIC READING WITH INJECTED SPIKES</div>
                <button onClick={fetchAnomalies} disabled={loading.anomaly} style={{
                  padding: "10px 28px", borderRadius: 8, background: loading.anomaly ? T.dim : T.red + "22",
                  color: loading.anomaly ? T.muted : T.red, border: "1px solid " + (loading.anomaly ? T.border : T.red + "66"),
                  cursor: loading.anomaly ? "not-allowed" : "pointer", fontWeight: 700, fontSize: 13, fontFamily: "'Space Mono', monospace", letterSpacing: 1,
                }}>
                  {loading.anomaly ? "SCANNING..." : "SCAN FOR ANOMALIES"}
                </button>
              </div>

              {anomalies.chartData ? (
                <>
                  <div style={{ display: "flex", gap: 20, marginBottom: 20 }}>
                    <div style={{ padding: "10px 16px", background: T.dim, borderRadius: 8, fontSize: 12 }} className="mono">
                      TOTAL <span style={{ color: T.accent, marginLeft: 8 }}>{anomalies.total}</span>
                    </div>
                    <div style={{ padding: "10px 16px", background: T.red + "15", border: "1px solid " + T.red + "44", borderRadius: 8, fontSize: 12 }} className="mono">
                      FLAGGED <span style={{ color: T.red, marginLeft: 8 }}>{anomalies.count}</span>
                    </div>
                  </div>
                  <ResponsiveContainer width="100%" height={280}>
                    <LineChart data={anomalies.chartData} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="1 4" stroke={T.border} />
                      <XAxis dataKey="idx" tick={{ fill: T.muted, fontSize: 10, fontFamily: "Space Mono" }} axisLine={false} tickLine={false} />
                      <YAxis tick={{ fill: T.muted, fontSize: 10, fontFamily: "Space Mono" }} axisLine={false} tickLine={false} />
                      <Tooltip content={<ChartTip />} />
                      <Line type="monotone" dataKey="value" stroke={T.accent} strokeWidth={1.5} dot={false} name="Reading" />
                      <Line type="monotone" dataKey="anomaly" stroke={T.red} strokeWidth={0} dot={{ r: 5, fill: T.red, strokeWidth: 2, stroke: T.red + "66" }} name="Anomaly" connectNulls={false} />
                    </LineChart>
                  </ResponsiveContainer>

                  {(anomalies.items || []).length > 0 && (
                    <div style={{ marginTop: 20 }}>
                      <div className="mono" style={{ fontSize: 10, color: T.muted, letterSpacing: 2, marginBottom: 10 }}>FLAGGED READINGS</div>
                      {(anomalies.items || []).map((a, i) => (
                        <div key={i} className="mono" style={{ padding: "10px 14px", background: T.red + "0d", border: "1px solid " + T.red + "33", borderRadius: 8, marginBottom: 6, fontSize: 12, display: "flex", gap: 20 }}>
                          <span style={{ color: T.muted }}>{a.timestamp?.slice(0, 19)}</span>
                          <span style={{ color: T.red }}>VAL {a.value?.toFixed(4)}</span>
                          <span style={{ color: T.amber }}>Z {a.z_score}</span>
                        </div>
                      ))}
                    </div>
                  )}
                </>
              ) : (
                <div style={{ height: 280, display: "grid", placeItems: "center" }}>
                  <div className="mono" style={{ color: T.muted, fontSize: 12, letterSpacing: 2 }}>PRESS SCAN TO ANALYZE READINGS</div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* ── OPTIMIZE TAB ── */}
        {tab === "optimize" && (
          <div className="fade-up">
            <SLabel>Energy Optimization Recommendations</SLabel>
            <div style={{ background: T.card, border: "1px solid " + T.border, borderRadius: 16, padding: 28 }}>
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 24 }}>
                <div className="mono" style={{ fontSize: 12, color: T.muted }}>RULE-BASED OPTIMIZATION ENGINE</div>
                <button onClick={fetchOptimize} disabled={loading.opt} style={{
                  padding: "10px 28px", borderRadius: 8, background: loading.opt ? T.dim : T.green + "22",
                  color: loading.opt ? T.muted : T.green, border: "1px solid " + (loading.opt ? T.border : T.green + "66"),
                  cursor: loading.opt ? "not-allowed" : "pointer", fontWeight: 700, fontSize: 13, fontFamily: "'Space Mono', monospace", letterSpacing: 1,
                }}>
                  {loading.opt ? "ANALYZING..." : "ANALYZE"}
                </button>
              </div>

              {suggestions.length > 0 ? (
                <div>
                  {suggestions.map((s, i) => (
                    <div key={i} style={{
                      padding: "16px 20px", background: T.dim, border: "1px solid " + T.border,
                      borderLeft: "3px solid " + T.green, borderRadius: "0 10px 10px 0",
                      marginBottom: 10, fontSize: 14, lineHeight: 1.6, display: "flex", gap: 12, alignItems: "flex-start",
                      animation: `fadeUp 0.3s ease ${i * 80}ms both`,
                    }}>
                      <span className="mono" style={{ color: T.green, fontSize: 11, marginTop: 3, flexShrink: 0 }}>{String(i + 1).padStart(2, "0")}</span>
                      <span>{s}</span>
                    </div>
                  ))}
                </div>
              ) : (
                <div style={{ height: 200, display: "grid", placeItems: "center" }}>
                  <div className="mono" style={{ color: T.muted, fontSize: 12, letterSpacing: 2 }}>PRESS ANALYZE TO GENERATE SUGGESTIONS</div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* ── Footer ── */}
        <div style={{ marginTop: 60, paddingTop: 20, borderTop: "1px solid " + T.border, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <div className="mono" style={{ fontSize: 10, color: T.muted, letterSpacing: 2 }}>ENERGY_OS / SMART METER ANALYTICS</div>
          <div className="mono" style={{ fontSize: 10, color: T.muted, letterSpacing: 2 }}>SARIMA + LSTM</div>
        </div>
      </div>
    </div>
  );
}
