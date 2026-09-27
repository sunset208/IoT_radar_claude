"use strict";
// Tableau de bord radar — dessin canvas maison (aucune dépendance externe).
//
// Deux cadences :
//  * indices rapides (présence, activité, forme d'onde) ~10 Hz, animés en continu
//    (requestAnimationFrame) : bande défilante et jauges fluides ;
//  * analyses spectrales (spectre, IQ, spectrogramme, historique) toutes les 0.5 s.

const $ = (id) => document.getElementById(id);
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const STATE_COLORS = { RESPIRATION: "--ok", MOUVEMENT: "--warn", PRESENCE: "--pres", VIDE: "--idle" };
const STATE_LABELS = {
  RESPIRATION: "RESPIRATION DÉTECTÉE", MOUVEMENT: "MOUVEMENT", PRESENCE: "PRÉSENCE — SIGNE DE VIE",
  VIDE: "AUCUNE DÉTECTION",
};
const WAVE_S = 30;       // durée de la bande défilante

let hist = [];          // [{t, score, state, snr, bpm, ml, pres, act}]
let wf = [];            // colonnes spectrogramme (dB)
let wfF = null;
let thresholds = { snr_db: 10, presence_db: 6, activity_db: 8, motion: 6 };
let recLabel = 1;
let recording = false;
let cfg = { scales_s: [20], window_s: 20, hop_s: 0.5, breath_band: [0.1, 0.8] };

// forme d'onde rapide : échantillons à pas constant (wave_dt), dernier à waveEndT
let wave = [];
let waveDt = 0.1;
let waveEndT = null;
let waveRecv = 0;       // performance.now() à la réception du dernier paquet
let waveScale = null;   // mm par unité brute (null → unités arbitraires)
let fast = null;
let gaugeVals = { pres: -5, act: -5 };
let last = null;        // dernier snapshot d'analyse
let extras = {};

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
const fmt = (v) => Math.abs(v) >= 10 ? v.toFixed(0) : Math.abs(v) >= 1 ? v.toFixed(1) : v.toFixed(2);

function line(g, xs, ys, sx, sy, color, w = 1.5) {
  g.strokeStyle = color; g.lineWidth = w; g.beginPath();
  let started = false;
  for (let i = 0; i < xs.length; i++) {
    if (ys[i] === null || ys[i] === undefined) { started = false; continue; }
    const X = sx(xs[i]), Y = sy(ys[i]);
    if (!started) { g.moveTo(X, Y); started = true; } else g.lineTo(X, Y);
  }
  g.stroke();
}

function viridis(t) {
  const c = [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]];
  const x = t * (c.length - 1), i = Math.min(c.length - 2, Math.floor(x)), f = x - i;
  return [0, 1, 2].map((k) => Math.round(c[i][k] + (c[i + 1][k] - c[i][k]) * f));
}

// image (colonnes × lignes) en fausses couleurs dans un rectangle
function heatmap(g, cols, x, y, w, h, vmin, vmax) {
  if (!cols.length || !cols[0].length) return;
  const nc = cols.length, nb = cols[0].length;
  const iw = Math.max(1, Math.round(w)), ih = Math.max(1, Math.round(h));
  const img = g.createImageData(iw, ih);
  for (let px = 0; px < iw; px++) {
    const c = cols[Math.min(nc - 1, Math.floor(px / iw * nc))];
    for (let py = 0; py < ih; py++) {
      const b = Math.min(nb - 1, Math.floor((1 - py / ih) * nb));
      const v = Math.max(0, Math.min(1, ((c[b] ?? vmin) - vmin) / (vmax - vmin)));
      const [r, gg, bl] = viridis(v);
      const o = (py * iw + px) * 4;
      img.data[o] = r; img.data[o + 1] = gg; img.data[o + 2] = bl; img.data[o + 3] = 255;
    }
  }
  // putImageData ignore la transformée du contexte : passage par un canvas hors écran
  const tmp = document.createElement("canvas"); tmp.width = iw; tmp.height = ih;
  tmp.getContext("2d").putImageData(img, 0, 0);
  g.imageSmoothingEnabled = false;
  g.drawImage(tmp, x, y, w, h);
}

// ---------------------------------------------------------------- bande défilante (60 fps)
function drawWave() {
  const { g, W, H } = setup($("c-wave"));
  if (!wave.length || waveEndT === null) return;
  // Temps d'affichage : on suit l'horloge locale depuis le dernier paquet, avec
  // 0.2 s de retard pour que le point le plus récent arrive en douceur au bord.
  const now = Math.min(waveEndT, waveEndT - 0.2 + (performance.now() - waveRecv) / 1000);
  const n = wave.length;
  const i0 = Math.max(0, n - Math.ceil(WAVE_S / waveDt) - 4);
  const xs = [], ys = [];
  for (let i = i0; i < n; i++) {
    const t = waveEndT - (n - 1 - i) * waveDt - now;
    if (t < -WAVE_S || t > 0.05) continue;
    xs.push(t); ys.push(waveScale ? wave[i] * waveScale : wave[i]);
  }
  let m = 1e-9;
  for (const v of ys) m = Math.max(m, Math.abs(v));
  m *= 1.15;
  const fr = frame(g, W, H, -WAVE_S, 0, -m, m, { nx: 6 });
  line(g, xs, ys, fr.sx, fr.sy, css("--plot1"), 1.8);
}

function setGauge(el, v, thr, on, lo = -5, hi = 30) {
  const frac = (x) => Math.max(0, Math.min(1, (x - lo) / (hi - lo)));
  el.querySelector(".gfill").style.width = (100 * frac(v)).toFixed(1) + "%";
  el.querySelector(".gthr").style.left = (100 * frac(thr)).toFixed(1) + "%";
  el.querySelector(".gv").textContent = v > -90 ? v.toFixed(1) + " dB" : "—";
  el.classList.toggle("on", !!on);
}

function animate() {
  drawWave();
  if (fast) {
    // lissage exponentiel entre deux ticks (10 Hz) → jauges fluides
    gaugeVals.pres += 0.35 * (fast.presence_db - gaugeVals.pres);
    gaugeVals.act += 0.35 * (fast.activity_db - gaugeVals.act);
    setGauge($("g-pres"), gaugeVals.pres, thresholds.presence_db, fast.presence);
    setGauge($("g-act"), gaugeVals.act, thresholds.activity_db, fast.activity);
  }
  requestAnimationFrame(animate);
}

// ---------------------------------------------------------------- analyses
function drawSpec(d, dec) {
  const { g, W, H } = setup($("c-spec"));
  if (!d) return;
  const xs = d.spec_f, ys = d.spec_db;
  const top = Math.max(20, Math.max(...ys) + 3);
  const fr = frame(g, W, H, 0, 3, -5, top, { nx: 6 });
  const bb = cfg.breath_band || [0.1, 0.8];
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
  const { g, W, H } = setup($("c-wf"));
  if (!wf.length || !wfF) return;
  const L = 38, B = 18, T = 6, pw = W - L - 8, ph = H - T - B;
  heatmap(g, wf, L, T, pw, ph, -3, 27);
  g.fillStyle = css("--muted"); g.textAlign = "right";
  const fmax = wfF[wfF.length - 1];
  for (let i = 0; i <= 3; i++) { const f = fmax * i / 3; g.fillText(f.toFixed(1) + " Hz", L - 2, T + ph - ph * i / 3 + 4); }
  g.textAlign = "center";
  g.fillText(`← ${(wf.length * (cfg.hop_s || 0.5)).toFixed(0)} s`, L + 30, H - 4);
  g.fillText("maintenant", L + pw - 30, H - 4);
}

function drawHist() {
  const { g, W, H } = setup($("c-hist"));
  if (!hist.length) return;
  const t1 = hist[hist.length - 1].t, t0 = Math.max(hist[0].t, t1 - 300);
  const fr = frame(g, W, H, t0 - t1, 0, 0, 1, { nx: 5 });
  const xs = hist.map((h) => h.t - t1);
  for (let i = 0; i < hist.length - 1; i++) {        // bande d'état
    g.fillStyle = css(STATE_COLORS[hist[i].state] || "--idle");
    g.globalAlpha = 0.25;
    g.fillRect(fr.sx(xs[i]), fr.T, Math.max(1, fr.sx(xs[i + 1]) - fr.sx(xs[i])), fr.ph);
  }
  g.globalAlpha = 1;
  const clip = (v) => v === null || v === undefined ? null : Math.min(1, Math.max(0, v));
  line(g, xs, hist.map((h) => clip(h.snr / 30)), fr.sx, fr.sy, css("--plot2"), 1);
  line(g, xs, hist.map((h) => clip(h.pres === null || h.pres === undefined ? null : h.pres / 30)), fr.sx, fr.sy, css("--pres"), 1);
  line(g, xs, hist.map((h) => h.score), fr.sx, fr.sy, css("--plot1"), 1.8);
  const hasMl = hist.some((h) => h.ml !== null && h.ml !== undefined);
  if (hasMl) line(g, xs, hist.map((h) => h.ml), fr.sx, fr.sy, css("--warn"), 1.2);
  g.fillStyle = css("--muted"); g.textAlign = "left";
  g.fillText("— confiance   — SNR/30 dB   — présence/30 dB" + (hasMl ? "   — IA" : ""), fr.L + 4, fr.T + 12);
}

// ---------------------------------------------------------------- statut / santé
function setState(state) {
  const card = $("status-card");
  card.className = "status card s-" + state;
  $("state").textContent = STATE_LABELS[state] || state;
}

function setStatus(snap) {
  const card = $("status-card");
  if (snap.status === "error") {
    card.className = "status card s-error"; $("state").textContent = "ERREUR";
    $("state-sub").textContent = snap.error || ""; return;
  }
  if (snap.status === "warmup" || snap.status === "starting") {
    if (!fast) { card.className = "status card"; $("state").textContent = "INITIALISATION"; }
    $("state-sub").textContent = `remplissage de la 1re fenêtre : ${Math.round(100 * (snap.progress || 0))} %`;
    return;
  }
  if (snap.status === "idle") {
    if (!extras.sfcw) { card.className = "status card"; $("state").textContent = snap.message || "EN ATTENTE"; }
    return;
  }
  const d = snap.decision; if (!d) return;
  if (!fast) setState(d.state);
  const f = d.features;
  const sub = [];
  if (f.motion >= thresholds.motion) sub.push("bouffées de mouvement");
  if (d.raw_positive && d.state !== "RESPIRATION") sub.push("confirmation en cours…");
  if (d.state === "RESPIRATION" && d.confirmed_by_s) sub.push(`confirmée sur la fenêtre de ${d.confirmed_by_s} s`);
  for (const w of d.warnings || []) sub.push("⚠ " + w);
  $("state-sub").textContent = sub.join(" · ");
  $("k-bpm").textContent = d.breath_bpm ? d.breath_bpm.toFixed(1) : "—";
  $("k-score").textContent = (100 * d.score).toFixed(0) + " %";
  $("k-snr").textContent = f.snr_db.toFixed(1);
  $("k-mm").textContent = (d.state === "RESPIRATION" && f.disp_mm_pp) ? f.disp_mm_pp.toFixed(1) : "—";
  $("k-hr").textContent = d.heart_bpm ? d.heart_bpm.toFixed(0) : "—";
  // détecteur multi-échelle : SNR de chaque fenêtre vs son seuil
  $("scales").innerHTML = (d.scales || []).map((s) => {
    const cls = (d.confirmed_by_s === s.window_s && d.state === "RESPIRATION") ? "conf" : s.positive ? "pos" : "";
    return `<span class="${cls}" title="fenêtre ${s.window_s} s — seuil ${s.snr_thr.toFixed(1)} dB">${s.window_s} s : ${s.snr_db.toFixed(1)}/${s.snr_thr.toFixed(0)} dB</span>`;
  }).join("");
  const tr = snap.truth;
  $("truth").textContent = tr ? "Vérité terrain : " + Object.entries(tr).filter(([, v]) => v !== null).map(([k, v]) => `${k}=${typeof v === "number" ? v.toFixed?.(1) ?? v : v}`).join(", ") : "";
}

function setHealth(snap) {
  if (snap.status === "idle") return;        // mode SFCW : santé dans le panneau SFCW
  const st = snap.stats || {}, f = snap.decision ? snap.decision.features : {};
  const rows = [];
  const cls = (v, w, b) => v >= b ? "bad" : v >= w ? "warn" : "good";
  if (st.adc_rms_dbfs !== undefined) {
    rows.push(["ADC RMS / crête", `<span class="${cls(st.adc_peak_dbfs, -3, -1)}">${st.adc_rms_dbfs.toFixed(1)} / ${st.adc_peak_dbfs.toFixed(1)} dBFS</span>`]);
  }
  if (f.static_dbfs !== undefined) rows.push(["Clutter statique", f.static_dbfs.toFixed(1) + " dBFS"]);
  if (f.noise_dbfs !== undefined) rows.push(["Plancher de bruit (slow-time)", f.noise_dbfs.toFixed(1) + " dBFS"]);
  if (f.drift_db !== undefined) rows.push(["Dérive LO (tang./rad.)", `<span class="${cls(f.drift_db, 6, 15)}">${f.drift_db.toFixed(1)} dB</span>`]);
  if (fast) rows.push(["Présence rad. / tang.", `${fast.presence_r_db.toFixed(1)} / ${fast.presence_t_db.toFixed(1)} dB`]);
  if (f.motion !== undefined) rows.push(["Indice de mouvement (20 s)", `<span class="${f.motion >= thresholds.motion ? "warn" : ""}">${f.motion.toFixed(1)}</span>`]);
  if (f.periodicity !== undefined) rows.push(["Périodicité (conc./ACF)", `${f.concentration.toFixed(2)} / ${f.acf.toFixed(2)}`]);
  if (st.queue !== undefined) rows.push(["File RX / pertes", `<span class="${(st.dropped || st.suspect_overflows) ? "bad" : "good"}">${st.queue} / ${st.dropped} (+${st.suspect_overflows} susp.)</span>`]);
  rows.push(["Calcul par analyse", (st.proc_ms || 0).toFixed(1) + " ms"]);
  if (snap.ml_score !== null && snap.ml_score !== undefined) rows.push(["Score IA", (100 * snap.ml_score).toFixed(0) + " %"]);
  $("health").innerHTML = rows.map(([a, b]) => `<tr><td>${a}</td><td>${b}</td></tr>`).join("");
  const rec = st.recording;
  recording = !!rec;
  $("rec-btn").textContent = rec ? "■ Arrêter" : "● Démarrer";
  $("rec-btn").classList.toggle("rec", !!rec);
  $("rec-info").textContent = rec ? `${rec.path} — ${rec.elapsed_s.toFixed(0)} s` : "";
}

function setSource(snap, scenarios) {
  const s = snap && snap.source; if (!s || s.kind === "idle") return;
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

// ---------------------------------------------------------------- panneaux annexes
function drawSfcw(sf) {
  $("sfcw-card").hidden = false;
  const sfcwMode = last && last.status === "idle";
  document.body.classList.toggle("mode-sfcw", !!sfcwMode);
  if (sfcwMode) sfcwStatus(sf);
  const st = sf.stats || {};
  const bits = [];
  if (st.reads_per_step) bits.push(`${st.reads_per_step.toFixed(1)} lectures/pas`);
  if (st.adc_peak_dbfs !== undefined) bits.push(`crête ADC ${st.adc_peak_dbfs.toFixed(1)} dBFS`);
  if (st.timeouts) bits.push(`${st.timeouts} pas ratés`);
  if (sf.static_ref) bits.push("fond figé");
  if (sfcwMode && sf.source) {
    const so = sf.source;
    let t = so.kind;
    if (so.scenario) t += ` · scénario « ${so.scenario} »`;
    if (so.uri) t += ` · ${so.uri} · RX ${so.rx_gain_db} dB`;
    if (so.file) t += ` · ${so.file.split(/[\\/]/).pop()}`;
    $("src-desc").textContent = t;
  }
  $("sfcw-stats").textContent = bits.join(" · ");
  const info = [];
  if (sf.status && sf.status !== "running") info.push(sf.status);
  if (sf.sweep_hz) info.push(`${sf.sweep_hz.toFixed(1)} balayages/s`);
  if (sf.f_range) info.push(`${(sf.f_range[0] / 1e9).toFixed(2)}–${(sf.f_range[1] / 1e9).toFixed(2)} GHz, ${sf.n_steps} pas`);
  if (sf.best_range_m !== null && sf.best_range_m !== undefined) info.push(`cible ≈ ${sf.best_range_m.toFixed(2)} m (SNR resp. ${sf.best_snr_db.toFixed(1)} dB)`);
  if (sf.error) info.push("⚠ " + sf.error);
  $("sfcw-info").textContent = info.length ? "(" + info.join(" · ") + ")" : "";
  // carte distance × temps
  let { g, W, H } = setup($("c-rt"));
  const L = 38, T = 6, B = 18, pw = W - L - 8, ph = H - T - B;
  if (sf.rt_map && sf.rt_map.length) {
    heatmap(g, sf.rt_map, L, T, pw, ph, sf.rt_vmin ?? -10, sf.rt_vmax ?? 20);
    g.fillStyle = css("--muted"); g.textAlign = "right";
    const rmax = sf.range_m[sf.range_m.length - 1];
    for (let i = 0; i <= 4; i++) { const r = rmax * i / 4; g.fillText(r.toFixed(1) + " m", L - 2, T + ph - ph * i / 4 + 4); }
    g.textAlign = "center"; g.fillText(`← ${(sf.rt_span_s || 0).toFixed(0)} s (énergie de mouvement, dB)`, L + pw / 2, H - 4);
  }
  // profil distance (fond retiré) + énergie respiratoire par case
  ({ g, W, H } = setup($("c-prof")));
  if (sf.profile_db && sf.range_m) {
    const top = Math.max(10, Math.max(...sf.profile_db) + 3);
    const fr = frame(g, W, H, 0, sf.range_m[sf.range_m.length - 1], Math.min(-10, Math.min(...sf.profile_db)), top, { nx: 4 });
    line(g, sf.range_m, sf.profile_db, fr.sx, fr.sy, css("--plot1"), 1.5);
    if (sf.static_db) line(g, sf.range_m, sf.static_db, fr.sx, fr.sy, css("--muted"), 1);
    g.fillStyle = css("--muted"); g.textAlign = "left"; g.fillText("— |profil − fond| (dB)  — fond", fr.L + 4, fr.T + 12);
  }
  ({ g, W, H } = setup($("c-rbins")));
  if (sf.breath_snr_db && sf.range_m) {
    const top = Math.max(15, Math.max(...sf.breath_snr_db) + 2);
    const fr = frame(g, W, H, 0, sf.range_m[sf.range_m.length - 1], 0, top, { nx: 6, ny: 3 });
    const bw = Math.max(2, fr.pw / sf.range_m.length - 1);
    for (let i = 0; i < sf.range_m.length; i++) {
      const v = sf.breath_snr_db[i];
      g.fillStyle = css(v >= (sf.breath_thr_db || 10) ? "--ok" : "--idle");
      g.fillRect(fr.sx(sf.range_m[i]) - bw / 2, fr.sy(Math.max(0, v)), bw, fr.sy(0) - fr.sy(Math.max(0, v)));
    }
    g.fillStyle = css("--muted"); g.textAlign = "left"; g.fillText("SNR respiratoire par case de distance (dB)", fr.L + 4, fr.T + 12);
  }
}

// mode SFCW : la carte d'état résume la décision distance × temps
function sfcwStatus(sf) {
  setState(sf.state || "VIDE");
  const sub = [];
  if (sf.state === "RESPIRATION" && sf.best_range_m != null) sub.push(`cible à ${sf.best_range_m.toFixed(2)} m`);
  if (sf.motion_extent_m) sub.push(`mouvement entre ${sf.motion_extent_m[0].toFixed(1)} et ${sf.motion_extent_m[1].toFixed(1)} m`);
  if (sf.status === "error") sub.push("⚠ " + (sf.error || "erreur"));
  $("state-sub").textContent = sub.join(" · ");
  $("k-bpm").textContent = sf.state === "RESPIRATION" && sf.best_bpm ? sf.best_bpm.toFixed(1) : "—";
  $("k-score").textContent = sf.sweep_hz ? sf.sweep_hz.toFixed(1) : "—";
  $("k-score").nextElementSibling.textContent = "balayages/s";
  $("k-snr").textContent = sf.best_snr_db != null ? sf.best_snr_db.toFixed(1) : "—";
  $("k-mm").textContent = sf.best_range_m != null ? sf.best_range_m.toFixed(2) : "—";
  $("k-mm").nextElementSibling.textContent = "distance (m)";
  $("k-hr").textContent = sf.resolution_m ? (100 * sf.resolution_m).toFixed(0) : "—";
  $("k-hr").nextElementSibling.textContent = "résolution (cm)";
  const rec = sf.recording;
  recording = !!rec;
  $("rec-btn").textContent = rec ? "■ Arrêter" : "● Démarrer";
  $("rec-btn").classList.toggle("rec", !!rec);
  $("rec-info").textContent = rec ? `${rec.path} — ${rec.elapsed_s.toFixed(0)} s` : "";
}

let mr60Hist = [];
function drawMr60(m) {
  $("mr60-card").hidden = false;
  const info = [m.model || "60 GHz", m.port || ""];
  if (m.status && m.status !== "running") info.push(m.status);
  if (m.last_rx_s !== null && m.last_rx_s !== undefined) info.push(`dernière trame il y a ${m.last_rx_s.toFixed(1)} s`);
  if (m.error) info.push("⚠ " + m.error);
  $("mr60-info").textContent = "(" + info.filter(Boolean).join(" · ") + ")";
  $("m-bpm").textContent = m.breath_bpm != null ? m.breath_bpm.toFixed(0) : "—";
  $("m-hr").textContent = m.heart_bpm != null ? m.heart_bpm.toFixed(0) : "—";
  $("m-dist").textContent = m.distance_m != null ? m.distance_m.toFixed(2) : "—";
  $("m-pres").textContent = m.presence == null ? "—" : m.presence ? "oui" : "non";
  if (m.history) mr60Hist = m.history;
  const { g, W, H } = setup($("c-mr60"));
  if (mr60Hist.length < 2) return;
  const t1 = mr60Hist[mr60Hist.length - 1].t, t0 = Math.max(mr60Hist[0].t, t1 - 120);
  const fr = frame(g, W, H, t0 - t1, 0, 0, 40, { nx: 4, ny: 4 });
  const xs = mr60Hist.map((h) => h.t - t1);
  line(g, xs, mr60Hist.map((h) => h.breath_bpm), fr.sx, fr.sy, css("--ok"), 1.6);
  line(g, xs, mr60Hist.map((h) => h.heart_bpm == null ? null : h.heart_bpm / 3), fr.sx, fr.sy, css("--bad"), 1.2);
  g.fillStyle = css("--muted"); g.textAlign = "left"; g.fillText("— resp/min   — cœur/3", fr.L + 4, fr.T + 12);
}

function setExtras(ex) {
  extras = ex || {};
  if (extras.sfcw) drawSfcw(extras.sfcw);
  if (extras.mr60) drawMr60(extras.mr60);
}

// ---------------------------------------------------------------- flux de données
function renderAnalysis() {
  if (!last) return;
  setStatus(last); setHealth(last);
  if (last.status === "running" && last.display) {
    drawSpec(last.display, last.decision); drawIQ(last.display);
  }
  drawWaterfall(); drawHist();
}

function onSnapshot(snap) {
  last = snap;
  if (snap.thresholds) thresholds = { ...thresholds, ...snap.thresholds };
  if (snap.status === "running" && snap.display) {
    wf.push(snap.display.spec_db); wfF = snap.display.spec_f; if (wf.length > 240) wf.shift();
  }
  setSource(snap);
  requestAnimationFrame(renderAnalysis);
}

function onFast(f) {
  if (!f) return;
  const first = !fast;
  fast = f;
  if (first) { gaugeVals.pres = f.presence_db; gaugeVals.act = f.activity_db; }
  waveDt = f.wave_dt || waveDt;
  const sc = f.wave_scale_mm || null;
  if (first || (sc === null) !== (waveScale === null)) {
    $("wave-unit").textContent = sc
      ? "(déplacement, mm — 30 s, temps réel)" : "(projection filtrée 0.12–1 Hz, µFS — 30 s, temps réel)";
  }
  waveScale = sc;   // toute la bande est remise à l'échelle d'un bloc : pas de saut
  if (!last || last.status !== "error") setState(f.state);
}

function onWave(samples, endT) {
  if (!samples || !samples.length) return;
  // saut en arrière (relecture en boucle, redémarrage) : on repart de zéro
  if (waveEndT !== null && endT < waveEndT - 1) wave = [];
  for (const v of samples) wave.push(v);
  const keep = Math.ceil((WAVE_S + 5) / waveDt);
  if (wave.length > keep) wave.splice(0, wave.length - keep);
  waveEndT = endT; waveRecv = performance.now();
}

function connect() {
  const ws = new WebSocket(`ws://${location.host}/ws`);
  ws.onopen = () => $("conn-dot").classList.add("on");
  ws.onclose = () => { $("conn-dot").classList.remove("on"); setTimeout(connect, 1500); };
  ws.onmessage = (ev) => {
    const m = JSON.parse(ev.data);
    if (m.type === "full") {
      hist = m.history || []; wf = m.waterfall || []; wfF = m.waterfall_f;
      cfg = { ...cfg, ...(m.config || {}) };
      wave = []; waveEndT = null; fast = null;
      setSource(m.snapshot, m.scenarios);
      onFast(m.fast);
      onWave(m.wave, m.wave_end_t);
      onSnapshot(m.snapshot);
      setExtras(m.extras);
      return;
    }
    if (m.fast) onFast(m.fast);
    if (m.wave) onWave(m.wave, m.wave_end_t);
    if (m.hist) { for (const h of m.hist) hist.push(h); if (hist.length > 700) hist.splice(0, hist.length - 700); }
    if (m.snapshot) onSnapshot(m.snapshot);
    if (m.extras) setExtras(m.extras);
  };
}

// ---------------------------------------------------------------- commandes
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
$("sfcw-bg").onclick = () => post("/api/sfcw/background");
$("scen").onchange = async (e) => { hist = []; wf = []; await post("/api/scenario", { scenario: e.target.value }); };
window.addEventListener("resize", () => { requestAnimationFrame(renderAnalysis); if (extras) setExtras(extras); });
connect();
requestAnimationFrame(animate);
