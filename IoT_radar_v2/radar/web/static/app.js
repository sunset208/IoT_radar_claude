"use strict";
// Tableau de bord radar — dessin canvas maison (aucune dépendance externe).

const $ = (id) => document.getElementById(id);
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const STATE_COLORS = { RESPIRATION: "--ok", MOUVEMENT: "--warn", VIDE: "--idle" };

let hist = [];          // [{t, score, state, snr, bpm, ml}]
let wf = [];            // colonnes spectrogramme (dB)
let wfF = null;
let thresholds = { snr_db: 9 };
let recLabel = 1;
let recording = false;

// ---------------------------------------------------------------- canvas utils
function setup(cv) {
  const r = cv.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const w = Math.max(10, Math.round(r.width * dpr)), h = Math.max(10, Math.round(r.height * dpr));
  if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
  const g = cv.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, r.width, r.height);
  g.font = "11px Segoe UI, system-ui, sans-serif";
  return { g, W: r.width, H: r.height };
}

function frame(g, W, H, x0, x1, y0, y1, opts = {}) {
  const L = 38, R = 8, T = 6, B = 18;
  const pw = W - L - R, ph = H - T - B;
  const sx = (x) => L + (x - x0) / (x1 - x0 || 1) * pw;
  const sy = (y) => T + (1 - (y - y0) / (y1 - y0 || 1)) * ph;
  g.strokeStyle = css("--grid"); g.fillStyle = css("--muted"); g.lineWidth = 1;
  const nx = opts.nx || 6, ny = opts.ny || 4;
  for (let i = 0; i <= ny; i++) {
    const v = y0 + (y1 - y0) * i / ny, y = sy(v);
    g.beginPath(); g.moveTo(L, y); g.lineTo(W - R, y); g.stroke();
    g.textAlign = "right"; g.fillText(fmt(v), L - 4, y + 4);
  }
  for (let i = 0; i <= nx; i++) {
    const v = x0 + (x1 - x0) * i / nx, x = sx(v);
    g.beginPath(); g.moveTo(x, T); g.lineTo(x, H - B); g.stroke();
    g.textAlign = "center"; g.fillText(fmt(v), x, H - 4);
  }
  return { sx, sy, L, R, T, B, pw, ph };
}
const fmt = (v) => Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(0) : v.toFixed(1);

function line(g, xs, ys, sx, sy, color, w = 1.5) {
  g.strokeStyle = color; g.lineWidth = w; g.beginPath();
  for (let i = 0; i < xs.length; i++) {
    if (ys[i] === null || ys[i] === undefined) continue;
    const X = sx(xs[i]), Y = sy(ys[i]);
    i === 0 ? g.moveTo(X, Y) : g.lineTo(X, Y);
  }
  g.stroke();
}

// ---------------------------------------------------------------- plots
function drawWave(d) {
  const { g, W, H } = setup($("c-wave"));
  if (!d) return;
  const ys = d.wave, xs = d.wave_t;
  let m = Math.max(...ys.map(Math.abs), 1e-9) * 1.15;
  const fr = frame(g, W, H, xs[0], 0, -m, m, { nx: 8 });
  line(g, xs, ys, fr.sx, fr.sy, css("--plot1"), 1.8);
  $("wave-unit").textContent = d.wave_unit === "mm" ? "(déplacement, mm — arc-tangente)" : "(unités arbitraires — projection)";
}

function drawSpec(d, dec) {
  const { g, W, H } = setup($("c-spec"));
  if (!d) return;
  const xs = d.spec_f, ys = d.spec_db;
  const top = Math.max(20, Math.max(...ys) + 3);
  const fr = frame(g, W, H, 0, 3, -5, top, { nx: 6 });
  const bb = window.__band || [0.1, 0.8];
  g.fillStyle = css("--band");
  g.fillRect(fr.sx(bb[0]), fr.T, fr.sx(bb[1]) - fr.sx(bb[0]), fr.ph);
  g.setLineDash([4, 4]); g.strokeStyle = css("--warn");
  g.beginPath(); g.moveTo(fr.sx(0), fr.sy(thresholds.snr_db)); g.lineTo(fr.sx(3), fr.sy(thresholds.snr_db)); g.stroke();
  g.setLineDash([]);
  line(g, xs, ys, fr.sx, fr.sy, css("--plot1"), 1.5);
  const f = dec && dec.features && dec.features.breath_hz;
  if (f) {
    g.fillStyle = css("--ok"); g.beginPath();
    g.arc(fr.sx(f), fr.sy(dec.features.snr_db), 4, 0, 2 * Math.PI); g.fill();
  }
}

function drawIQ(d) {
  const { g, W, H } = setup($("c-iq"));
  if (!d) return;
  const S = Math.min(W, H) - 16, cx = W / 2, cy = H / 2, k = S / 2.3;
  g.strokeStyle = css("--grid");
  g.beginPath(); g.moveTo(cx - S / 2, cy); g.lineTo(cx + S / 2, cy); g.moveTo(cx, cy - S / 2); g.lineTo(cx, cy + S / 2); g.stroke();
  g.fillStyle = css("--plot2");
  for (const [x, y] of d.iq) { g.fillRect(cx + x * k - 1.2, cy - y * k - 1.2, 2.4, 2.4); }
  if (d.circle) {
    const [ccx, ccy, r] = d.circle.map((v) => v / (d.iq_scale || 1));
    g.strokeStyle = css("--ok"); g.setLineDash([3, 3]); g.beginPath();
    g.arc(cx + ccx * k, cy - ccy * k, r * k, 0, 2 * Math.PI); g.stroke(); g.setLineDash([]);
  }
  g.fillStyle = css("--muted"); g.textAlign = "left";
  g.fillText(`échelle ${(20 * Math.log10(d.iq_scale + 1e-15)).toFixed(0)} dBFS`, 6, 14);
}

function drawWaterfall() {
  const cv = $("c-wf");
  const { g, W, H } = setup(cv);
  if (!wf.length || !wfF) return;
  const L = 38, B = 18, T = 6, pw = W - L - 8, ph = H - T - B;
  const nb = wf[0].length, nc = wf.length;
  const img = g.createImageData(Math.max(1, Math.round(pw)), Math.max(1, Math.round(ph)));
  const iw = img.width, ih = img.height;
  for (let x = 0; x < iw; x++) {
    const c = wf[Math.min(nc - 1, Math.floor(x / iw * nc))];
    for (let y = 0; y < ih; y++) {
      const b = Math.min(nb - 1, Math.floor((1 - y / ih) * nb));
      const v = Math.max(0, Math.min(1, (c[b] + 3) / 30));
      const [r, gg, bl] = viridis(v);
      const o = (y * iw + x) * 4;
      img.data[o] = r; img.data[o + 1] = gg; img.data[o + 2] = bl; img.data[o + 3] = 255;
    }
  }
  // putImageData ignore la transformée du contexte : passage par un canvas hors écran
  const tmp = document.createElement("canvas"); tmp.width = iw; tmp.height = ih;
  tmp.getContext("2d").putImageData(img, 0, 0);
  g.imageSmoothingEnabled = false;
  g.drawImage(tmp, L, T, pw, ph);
  g.fillStyle = css("--muted"); g.textAlign = "right";
  const fmax = wfF[wfF.length - 1];
  for (let i = 0; i <= 3; i++) { const f = fmax * i / 3; g.fillText(f.toFixed(1) + " Hz", L - 2, T + ph - ph * i / 3 + 4); }
  g.textAlign = "center";
  g.fillText(`← ${(nc * (window.__hop || 0.5)).toFixed(0)} s`, L + 30, H - 4);
  g.fillText("maintenant", L + pw - 30, H - 4);
}

function viridis(t) {
  const c = [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]];
  const x = t * (c.length - 1), i = Math.min(c.length - 2, Math.floor(x)), f = x - i;
  return [0, 1, 2].map((k) => Math.round(c[i][k] + (c[i + 1][k] - c[i][k]) * f));
}

function drawHist() {
  const { g, W, H } = setup($("c-hist"));
  if (!hist.length) return;
  const t1 = hist[hist.length - 1].t, t0 = Math.max(hist[0].t, t1 - 300);
  const fr = frame(g, W, H, t0 - t1, 0, 0, 1, { nx: 5 });
  const xs = hist.map((h) => h.t - t1);
  // bande d'état
  for (let i = 0; i < hist.length - 1; i++) {
    g.fillStyle = css(STATE_COLORS[hist[i].state] || "--idle");
    g.globalAlpha = 0.25;
    g.fillRect(fr.sx(xs[i]), fr.T, Math.max(1, fr.sx(xs[i + 1]) - fr.sx(xs[i])), fr.ph);
  }
  g.globalAlpha = 1;
  line(g, xs, hist.map((h) => Math.min(1, Math.max(0, h.snr / 30))), fr.sx, fr.sy, css("--plot2"), 1);
  line(g, xs, hist.map((h) => h.score), fr.sx, fr.sy, css("--plot1"), 1.8);
  if (hist.some((h) => h.ml !== null && h.ml !== undefined))
    line(g, xs, hist.map((h) => h.ml), fr.sx, fr.sy, css("--warn"), 1.2);
  g.fillStyle = css("--muted"); g.textAlign = "left";
  g.fillText("— confiance   — SNR/30 dB" + (hist.some((h) => h.ml != null) ? "   — IA" : ""), fr.L + 4, fr.T + 12);
}

// ---------------------------------------------------------------- status/health
function setStatus(snap) {
  const card = $("status-card");
  if (snap.status === "error") {
    card.className = "status card s-error"; $("state").textContent = "ERREUR";
    $("state-sub").textContent = snap.error || ""; return;
  }
  if (snap.status === "warmup" || snap.status === "starting") {
    card.className = "status card";
    $("state").textContent = "INITIALISATION";
    $("state-sub").textContent = `remplissage de la fenêtre : ${Math.round(100 * (snap.progress || 0))} %`;
    return;
  }
  const d = snap.decision; if (!d) return;
  card.className = "status card s-" + d.state;
  $("state").textContent = d.state === "RESPIRATION" ? "RESPIRATION DÉTECTÉE" : d.state === "MOUVEMENT" ? "MOUVEMENT / PRÉSENCE" : "AUCUNE DÉTECTION";
  const f = d.features;
  const sub = [];
  if (f.motion >= thresholds.motion) sub.push("bouffées de mouvement");
  if (d.raw_positive && d.state !== "RESPIRATION") sub.push("confirmation en cours…");
  for (const w of d.warnings || []) sub.push("⚠ " + w);
  $("state-sub").textContent = sub.join(" · ");
  $("k-bpm").textContent = d.breath_bpm ? d.breath_bpm.toFixed(1) : "—";
  $("k-score").textContent = (100 * d.score).toFixed(0) + " %";
  $("k-snr").textContent = f.snr_db.toFixed(1);
  $("k-mm").textContent = (d.state === "RESPIRATION" && f.disp_mm_pp) ? f.disp_mm_pp.toFixed(1) : "—";
  $("k-hr").textContent = d.heart_bpm ? d.heart_bpm.toFixed(0) : "—";
  const tr = snap.truth;
  $("truth").textContent = tr ? "Vérité terrain : " + Object.entries(tr).filter(([, v]) => v !== null).map(([k, v]) => `${k}=${typeof v === "number" ? v.toFixed?.(1) ?? v : v}`).join(", ") : "";
}

function setHealth(snap) {
  const st = snap.stats || {}, f = snap.decision ? snap.decision.features : {};
  const rows = [];
  const cls = (v, w, b) => v >= b ? "bad" : v >= w ? "warn" : "good";
  if (st.adc_rms_dbfs !== undefined) {
    rows.push(["ADC RMS / crête", `<span class="${cls(st.adc_peak_dbfs, -3, -1)}">${st.adc_rms_dbfs.toFixed(1)} / ${st.adc_peak_dbfs.toFixed(1)} dBFS</span>`]);
  }
  if (f.static_dbfs !== undefined) rows.push(["Clutter statique", f.static_dbfs.toFixed(1) + " dBFS"]);
  if (f.noise_dbfs !== undefined) rows.push(["Plancher de bruit (slow-time)", f.noise_dbfs.toFixed(1) + " dBFS"]);
  if (f.drift_db !== undefined) rows.push(["Dérive LO (tang./rad.)", `<span class="${cls(f.drift_db, 6, 15)}">${f.drift_db.toFixed(1)} dB</span>`]);
  if (f.motion !== undefined) rows.push(["Indice de mouvement", `<span class="${f.motion >= thresholds.motion ? "warn" : ""}">${f.motion.toFixed(1)}</span>`]);
  if (f.periodicity !== undefined) rows.push(["Périodicité (conc./ACF)", `${f.concentration.toFixed(2)} / ${f.acf.toFixed(2)}`]);
  if (st.queue !== undefined) rows.push(["File RX / pertes", `<span class="${(st.dropped || st.suspect_overflows) ? "bad" : "good"}">${st.queue} / ${st.dropped} (+${st.suspect_overflows} susp.)</span>`]);
  rows.push(["Calcul par fenêtre", (st.proc_ms || 0).toFixed(1) + " ms"]);
  if (snap.ml_score !== null && snap.ml_score !== undefined) rows.push(["Score IA", (100 * snap.ml_score).toFixed(0) + " %"]);
  $("health").innerHTML = rows.map(([a, b]) => `<tr><td>${a}</td><td>${b}</td></tr>`).join("");
  const rec = st.recording;
  recording = !!rec;
  $("rec-btn").textContent = rec ? "■ Arrêter" : "● Démarrer";
  $("rec-btn").classList.toggle("rec", !!rec);
  $("rec-info").textContent = rec ? `${rec.path} — ${rec.elapsed_s.toFixed(0)} s` : "";
}

function setSource(snap, scenarios) {
  const s = snap.source; if (!s) return;
  let t = s.kind;
  if (s.scenario) t += ` · scénario « ${s.scenario} »`;
  if (s.uri) t += ` · ${s.uri} · ${(s.f_c / 1e9).toFixed(2)} GHz · RX ${s.rx_gain_db} dB`;
  if (s.file) t += ` · ${s.file.split(/[\\/]/).pop()}`;
  $("src-desc").textContent = t;
  if (scenarios && scenarios.length) {
    $("scen-row").hidden = false;
    const sel = $("scen");
    if (!sel.options.length) for (const x of scenarios) sel.add(new Option(x, x));
    if (s.scenario) sel.value = s.scenario;
  }
}

// ---------------------------------------------------------------- data flow
let last = null;
function render() {
  if (!last) return;
  setStatus(last); setHealth(last);
  if (last.status === "running" && last.display) {
    drawWave(last.display); drawSpec(last.display, last.decision); drawIQ(last.display);
  }
  drawWaterfall(); drawHist();
}

function onSnapshot(snap, h) {
  last = snap;
  if (snap.thresholds) thresholds = snap.thresholds;
  if (h) { hist.push(h); if (hist.length > 600) hist.shift(); }
  if (snap.status === "running" && snap.display) {
    wf.push(snap.display.spec_db); wfF = snap.display.spec_f; if (wf.length > 240) wf.shift();
  }
  setSource(snap);
  requestAnimationFrame(render);
}

function connect() {
  const ws = new WebSocket(`ws://${location.host}/ws`);
  ws.onopen = () => $("conn-dot").classList.add("on");
  ws.onclose = () => { $("conn-dot").classList.remove("on"); setTimeout(connect, 1500); };
  ws.onmessage = (ev) => {
    const m = JSON.parse(ev.data);
    if (m.type === "full") {
      hist = m.history || []; wf = m.waterfall || []; wfF = m.waterfall_f;
      window.__band = m.config.breath_band; window.__hop = m.config.hop_s;
      setSource(m.snapshot, m.scenarios);
      onSnapshot(m.snapshot, null);
    } else onSnapshot(m.snapshot, m.hist);
  };
}

// ---------------------------------------------------------------- controls
async function post(url, body) {
  const r = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) });
  return r.json();
}
document.querySelectorAll("#rec-label button").forEach((b) => b.onclick = () => {
  document.querySelectorAll("#rec-label button").forEach((x) => x.classList.remove("on"));
  b.classList.add("on"); recLabel = parseInt(b.dataset.v, 10);
});
$("rec-btn").onclick = async () => {
  if (recording) await post("/api/record/stop");
  else await post("/api/record/start", { label: recLabel, tag: $("rec-tag").value, notes: $("rec-notes").value });
};
document.querySelectorAll(".annots button").forEach((b) => b.onclick = () => post("/api/annotate", { text: b.dataset.a }));
$("scen").onchange = async (e) => { hist = []; wf = []; await post("/api/scenario", { scenario: e.target.value }); };
window.addEventListener("resize", () => requestAnimationFrame(render));
connect();
