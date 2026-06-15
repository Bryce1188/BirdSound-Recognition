from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from birdclef_lsw.dashboard import build_dashboard_payload
from birdclef_lsw.inference import predict_audio_file


LSW_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = LSW_ROOT.parent
REPORT_DIR = "\u62a5\u544a"
ARTIFACTS_DIR = "\u5b9e\u9a8c\u4ea7\u7269"

app = FastAPI(title="BirdCLEF2026 \u9e1f\u7c7b\u58f0\u97f3\u8bc6\u522b\u5b9e\u9a8c\u7cfb\u7edf")
app.mount("/artifacts", StaticFiles(directory=str(LSW_ROOT / REPORT_DIR / ARTIFACTS_DIR)), name="artifacts")
app.mount("/web_test_audio", StaticFiles(directory=str(WORKSPACE_ROOT / "web_test_audio")), name="web_test_audio")


def _predict_sample_file(sample_name: str) -> dict[str, object]:
    sample_path = (WORKSPACE_ROOT / "web_test_audio" / sample_name).resolve()
    sample_root = (WORKSPACE_ROOT / "web_test_audio").resolve()
    if sample_root not in sample_path.parents or not sample_path.is_file():
        raise HTTPException(status_code=404, detail=f"Sample audio not found: {sample_name}")
    return predict_audio_file(LSW_ROOT, sample_path)


INDEX_HTML = """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>BirdCLEF2026 \u9e1f\u7c7b\u58f0\u97f3\u8bc6\u522b</title>
  <style>
    :root { --ink:#14233b; --muted:#53677f; --line:#d4dfec; --soft:#eef4fb; --accent:#1d6fa5; --accent2:#2a9d8f; --bg:#f3f6fb; --panel:#fff; --warn:#d90429; }
    * { box-sizing: border-box; }
    html { font-family: "Microsoft YaHei", "Noto Sans CJK SC", "PingFang SC", "Segoe UI", Arial, sans-serif; }
    body { margin: 0; color: var(--ink); background: var(--bg); }
    header { padding: 22px 28px 18px; background: linear-gradient(135deg, #fff, #eef6fb); border-bottom: 1px solid var(--line); }
    h1 { margin: 0 0 8px; font-size: 28px; }
    h2 { margin: 0 0 12px; font-size: 18px; }
    .sub { margin: 0; color: var(--muted); font-size: 14px; line-height: 1.6; }
    main { max-width: 1320px; margin: 0 auto; padding: 18px; display: grid; gap: 16px; }
    section { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 16px; }
    .hero-grid, .grid, .figures, .two-col, .sample-grid { display: grid; gap: 14px; }
    .hero-grid { grid-template-columns: repeat(4, minmax(0, 1fr)); margin-top: 18px; }
    .two-col { grid-template-columns: 1.1fr 1fr; }
    .grid { grid-template-columns: repeat(4, minmax(0, 1fr)); }
    .metric, .figure-card, .sample-card, .rank-card { border: 1px solid var(--line); border-radius: 8px; background: #fff; }
    .metric { padding: 14px; min-height: 94px; background: linear-gradient(180deg, #fff, #fbfdff); }
    .metric strong { display: block; font-size: 12px; color: var(--muted); font-weight: 700; text-transform: uppercase; }
    .metric span { display: block; margin-top: 8px; font-size: 22px; font-weight: 700; }
    .metric small { display: block; margin-top: 8px; color: var(--muted); line-height: 1.5; }
    .banner { border: 1px solid var(--line); border-radius: 8px; background: var(--soft); padding: 12px 14px; font-size: 14px; line-height: 1.7; margin-bottom: 12px; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th, td { border-bottom: 1px solid var(--line); padding: 8px; text-align: left; vertical-align: top; }
    th { background: #eef3f8; color: #334155; }
    button { border: 0; border-radius: 6px; padding: 10px 14px; background: var(--accent); color: white; cursor: pointer; font-weight: 600; }
    button:disabled { opacity: 0.55; cursor: wait; }
    input[type=file] { display: block; margin: 8px 0 12px; }
    .figures { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .sample-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .figure-card, .sample-card { padding: 10px; }
    .figure-card img, .prediction-figure img { width: 100%; border: 1px solid var(--line); border-radius: 6px; background: #fff; }
    .figure-card strong, .sample-card strong { display: block; margin-bottom: 8px; font-size: 13px; }
    .sample-card audio { width: 100%; margin-top: 8px; }
    .rank-list { display: grid; gap: 10px; }
    .rank-card { padding: 12px; }
    .rank-head { display: flex; justify-content: space-between; align-items: baseline; gap: 12px; font-size: 16px; font-weight: 800; }
    .rank-name small { display: block; margin-top: 3px; color: var(--muted); font-size: 12px; font-weight: 600; }
    .score-badge { flex: 0 0 auto; border-radius: 999px; background: #e8f4ec; color: #176a49; padding: 5px 10px; font-size: 13px; font-weight: 800; }
    .rank-meta { margin-top: 6px; color: var(--muted); font-size: 12px; line-height: 1.6; }
    .bar { height: 8px; margin-top: 10px; border-radius: 999px; background: #e5edf6; overflow: hidden; }
    .bar > span { display: block; height: 100%; background: linear-gradient(90deg, var(--accent), var(--accent2)); }
    .prediction-figure { display: grid; gap: 12px; grid-template-columns: repeat(3, minmax(0, 1fr)); margin-top: 14px; }
    .pill { display: inline-block; padding: 3px 8px; border-radius: 999px; background: #e8f4ec; color: #1f7a4b; font-size: 12px; font-weight: 700; }
    .status { color: var(--muted); font-size: 13px; }
    .ok { color: #1f7a4b; font-weight: 700; }
    .warn { color: var(--warn); font-weight: 700; }
    @media (max-width: 1040px) { .hero-grid, .grid, .two-col, .prediction-figure { grid-template-columns: 1fr; } }
    @media (max-width: 860px) { .figures, .sample-grid { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <header>
    <h1>BirdCLEF2026 \u9e1f\u7c7b\u58f0\u97f3\u8bc6\u522b\u5b9e\u9a8c\u7cfb\u7edf</h1>
    <p class="sub">FastAPI \u9875\u9762 · \u975e Gradio · BirdCLEF2026 \u58f0\u666f\u6570\u636e\u3001\u4f20\u7edf\u673a\u5668\u5b66\u4e60\u4e0e\u6df1\u5ea6\u5b66\u4e60\u5bf9\u7167\u5b9e\u9a8c\u3001\u6392\u540d\u7c7b\u6307\u6807\u4e0e\u53ef\u89e3\u91ca\u58f0\u5b66\u7279\u5f81\u5c55\u793a</p>
    <div id="hero" class="hero-grid"></div>
  </header>
  <main>
    <section class="two-col">
      <div>
        <h2>\u97f3\u9891\u8bc6\u522b</h2>
        <div class="banner">\u4e0a\u4f20\u9e1f\u9e23\u97f3\u9891\u540e\uff0c\u7cfb\u7edf\u4f1a\u8fd4\u56de\u5f53\u524d\u6a21\u578b\u8bad\u7ec3\u6807\u7b7e\u8303\u56f4\u5185\u7684\u9e1f\u7c7b\u9884\u6d4b\u7ed3\u679c\uff0c\u5e76\u5c55\u793a\u4e2d\u6587\u53c2\u8003\u540d\u3001\u82f1\u6587\u5e38\u7528\u540d\u3001\u5b66\u540d\u548c\u58f0\u5b66\u7279\u5f81\u56fe\u3002</div>
        <input id="file" type="file" accept=".wav,.ogg,.mp3,.flac,audio/*" />
        <button id="predict">\u4e0a\u4f20\u5e76\u8bc6\u522b</button>
        <p id="predict-status" class="status">\u8bf7\u9009\u62e9\u4e00\u6bb5\u9e1f\u9e23\u97f3\u9891\u3002</p>
        <div id="prediction"></div>
      </div>
      <div>
        <h2>\u6f14\u793a\u6837\u4f8b\u97f3\u9891</h2>
        <div id="sample-audio" class="sample-grid"></div>
      </div>
    </section>
    <section>
      <h2>\u9884\u6d4b\u53ef\u89c6\u5316</h2>
      <div id="prediction-figures" class="prediction-figure"></div>
    </section>
    <section>
      <h2>\u5b9e\u9a8c\u6307\u6807\u770b\u677f</h2>
      <div id="summary-banner" class="banner"></div>
      <div id="summary" class="grid"></div>
      <div id="models"></div>
    </section>
    <section class="two-col">
      <div><h2>\u9608\u503c\u4f18\u5316\u7ed3\u679c</h2><div id="thresholds"></div></div>
      <div><h2>\u8bfe\u7a0b\u8981\u6c42\u6838\u5bf9</h2><div id="checklist"></div></div>
    </section>
    <section>
      <h2>\u5b9e\u9a8c\u56fe\u8868</h2>
      <div id="figures" class="figures"></div>
    </section>
  </main>
  <script>
    const escapeHtml = (value) => String(value ?? "").replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");
    const fmt = (value) => Number.isFinite(Number(value)) ? Number(value).toFixed(4) : String(value ?? "");
    const statusText = (value) => value === "classical_completed" ? "\u4f20\u7edf\u6a21\u578b\u5b9e\u9a8c\u5df2\u5b8c\u6210" : escapeHtml(value || "\u5f85\u751f\u6210");

    async function loadDashboard() {
      const data = await fetch("/api/dashboard").then(r => r.json());
      const models = [...(data.models || [])].sort((a, b) => Number(b.lrap || 0) - Number(a.lrap || 0));
      const best = models[0] || {};
      const manifest = data.manifest || {};
      document.querySelector("#hero").innerHTML = `
        <div class="metric"><strong>\u5b9e\u9a8c\u4e3b\u9898</strong><span>Bird Sound Recognition</span><small>BirdCLEF2026 course project</small></div>
        <div class="metric"><strong>\u6570\u636e\u96c6</strong><span>${escapeHtml(data.dataset || "BirdCLEF2026")}</span><small>${escapeHtml(manifest.sample_count || "")} \u4e2a\u6837\u672c\u7a97\u53e3 / ${escapeHtml(manifest.label_column_count || "")} \u4e2a\u8bad\u7ec3\u6807\u7b7e</small></div>
        <div class="metric"><strong>\u5f53\u524d\u72b6\u6001</strong><span>${statusText(data.status)}</span><small>\u751f\u6210\u65f6\u95f4\uff1a${escapeHtml(data.generated_at || "")}</small></div>
        <div class="metric"><strong>\u6700\u4f18\u4f20\u7edf\u6a21\u578b</strong><span>${escapeHtml(manifest.best_classical_model || "N/A")}</span><small>\u6307\u6807\u91cd\u70b9\uff1aLRAP / Top-k / Hamming Loss</small></div>`;
      document.querySelector("#summary-banner").innerHTML = `\u5f53\u524d\u4e3b\u7ed3\u679c\u4ee5 <span class="pill">${escapeHtml(manifest.best_classical_model || "extra_trees")}</span> \u4e3a\u4e3b\uff0c\u6df1\u5ea6\u6a21\u578b\u4f5c\u4e3a\u5bf9\u7167\u5b9e\u9a8c\u3002\u9875\u9762\u5c55\u793a LRAP\u3001Micro F1\u3001Top-3 \u547d\u4e2d\u7387\u548c Hamming Loss \u7b49\u8bfe\u7a0b\u5b9e\u9a8c\u6307\u6807\u3002`;
      document.querySelector("#summary").innerHTML = `
        <div class="metric"><strong>\u6700\u4f73 LRAP</strong><span>${fmt(best.lrap)}</span><small>${escapeHtml(best.model || "")}</small></div>
        <div class="metric"><strong>\u6700\u4f73 Micro F1</strong><span>${fmt(best.micro_f1)}</span><small>Top-3 \u547d\u4e2d\u7387 ${fmt(best.top3_hit_rate)}</small></div>
        <div class="metric"><strong>\u6700\u4f18\u9608\u503c</strong><span>${fmt(best.best_threshold)}</span><small>Hamming Loss ${fmt(best.hamming_loss)}</small></div>
        <div class="metric"><strong>\u7279\u5f81\u7ef4\u5ea6</strong><span>${escapeHtml(manifest.feature_column_count || "161")}</span><small>MFCC / pitch / volume / timbre / onset</small></div>`;
      const rows = models.map(row => `<tr><td>${escapeHtml(row.model)}</td><td>${fmt(row.lrap)}</td><td>${fmt(row.micro_f1)}</td><td>${fmt(row.macro_f1)}</td><td>${fmt(row.top1_hit_rate)}</td><td>${fmt(row.top3_hit_rate)}</td><td>${fmt(row.hamming_loss)}</td><td>${fmt(row.best_threshold)}</td></tr>`).join("");
      document.querySelector("#models").innerHTML = `<table><thead><tr><th>\u6a21\u578b</th><th>LRAP</th><th>Micro F1</th><th>Macro F1</th><th>Top-1</th><th>Top-3</th><th>Hamming Loss</th><th>\u9608\u503c</th></tr></thead><tbody>${rows}</tbody></table>`;
      const thresholdRows = (data.thresholds || []).map(row => `<tr><td>${escapeHtml(row.model)}</td><td>${fmt(row.threshold)}</td><td>${fmt(row.lrap)}</td><td>${fmt(row.micro_f1)}</td><td>${fmt(row.prediction_density)}</td></tr>`).join("");
      document.querySelector("#thresholds").innerHTML = `<table><thead><tr><th>\u6a21\u578b</th><th>\u9608\u503c</th><th>LRAP</th><th>Micro F1</th><th>\u9884\u6d4b\u5bc6\u5ea6</th></tr></thead><tbody>${thresholdRows}</tbody></table>`;
      const checks = (data.teacher_checklist || []).map(item => `<tr><td>${escapeHtml(item.requirement)}</td><td class="ok">${escapeHtml(item.status)}</td><td>${escapeHtml(item.evidence)}</td></tr>`).join("");
      document.querySelector("#checklist").innerHTML = `<table><thead><tr><th>\u8981\u6c42</th><th>\u72b6\u6001</th><th>\u8bc1\u636e</th></tr></thead><tbody>${checks}</tbody></table>`;
      document.querySelector("#figures").innerHTML = (data.figures || []).map(fig => `<div class="figure-card"><strong>${escapeHtml(fig.name)}</strong><img src="${fig.url}" alt="${escapeHtml(fig.name)}" /></div>`).join("");
      document.querySelector("#sample-audio").innerHTML = (data.sample_audio || []).map(audio => `<div class="sample-card"><strong>${escapeHtml(audio.name)}</strong><button type="button" data-sample="${escapeHtml(audio.name)}">\u4f7f\u7528\u8be5\u6837\u4f8b\u8bc6\u522b</button><audio controls preload="none" src="${audio.url}"></audio></div>`).join("");
      document.querySelectorAll("[data-sample]").forEach(button => button.addEventListener("click", () => predictSample(button.dataset.sample)));
    }

    function displayName(row) {
      if (row.chinese_name) return `${row.chinese_name} (${row.common_name || row.species_name || row.label})`;
      return row.display_name || row.common_name || row.scientific_name || row.label;
    }

    function confidenceText(value) {
      const number = Number(value || 0);
      return `\u7f6e\u4fe1\u5ea6 ${(number * 100).toFixed(1)}%`;
    }

    function renderPrediction(data) {
      const predictionTitle = data.prediction_scope === "birds" ? "\u9e1f\u7c7b\u9884\u6d4b\u7ed3\u679c" : "\u58f0\u666f\u5019\u9009\u7ed3\u679c";
      const predictionNote = data.prediction_scope === "birds"
        ? "\u4e0b\u9762\u662f\u6a21\u578b\u8ba4\u4e3a\u66f4\u50cf\u7684\u9e1f\u7c7b\uff0c\u6309\u7f6e\u4fe1\u5ea6\u4ece\u9ad8\u5230\u4f4e\u6392\u5e8f\u3002\u5982\u679c\u524d\u51e0\u540d\u5206\u6570\u5f88\u63a5\u8fd1\uff0c\u8bf4\u660e\u6a21\u578b\u5bf9\u8fd9\u6bb5\u97f3\u9891\u7684\u533a\u5206\u5ea6\u4e0d\u9ad8\uff0c\u7ed3\u679c\u66f4\u9002\u5408\u4f5c\u4e3a\u5019\u9009\u53c2\u8003\u3002"
        : "\u5f53\u524d\u97f3\u9891\u5728\u9e1f\u7c7b\u6807\u7b7e\u4e2d\u7f6e\u4fe1\u5ea6\u8f83\u4f4e\uff0c\u56e0\u6b64\u663e\u793a\u58f0\u666f\u5019\u9009\u3002";
      document.querySelector("#prediction").innerHTML = `
        <div class="banner"><strong>${predictionTitle}</strong><br>${predictionNote}</div>
        <div class="rank-list">${data.top_predictions.map(row => `
          <div class="rank-card">
            <div class="rank-head"><span class="rank-name">${escapeHtml(displayName(row))}<small>${escapeHtml(row.scientific_name || "")}</small></span><span class="score-badge">${confidenceText(row.score)}</span></div>
            <div class="bar"><span style="width:${Math.max(2, Math.round(Number(row.score || 0) * 100))}%"></span></div>
            <div class="rank-meta">
              \u6807\u7b7e\u7f16\u53f7\uff1a${escapeHtml(row.label_id || row.species_code || row.label)}
              ${row.class_name ? ` | \u7c7b\u7fa4\uff1a${escapeHtml(row.class_display_name || row.class_name)} / ${escapeHtml(row.class_name)}` : ""}
            </div>
          </div>`).join("")}</div>`;
      document.querySelector("#prediction-figures").innerHTML = Object.entries(data.figures).map(([name, url]) => `<div class="figure-card"><strong>${escapeHtml(name)}</strong><img src="${url}" alt="${escapeHtml(name)}" /></div>`).join("");
      document.querySelector("#predict-status").textContent = `\u6a21\u578b\uff1a${data.model}\uff1b\u7a97\u53e3\u6570\uff1a${data.window_count}\uff1b\u65f6\u957f\uff1a${data.duration_seconds}s`;
    }

    async function predict() {
      const input = document.querySelector("#file");
      const status = document.querySelector("#predict-status");
      const button = document.querySelector("#predict");
      if (!input.files.length) { status.textContent = "\u8bf7\u5148\u9009\u62e9\u97f3\u9891\u6587\u4ef6\u3002"; return; }
      const form = new FormData();
      form.append("file", input.files[0]);
      button.disabled = true;
      status.textContent = "\u6b63\u5728\u63a8\u7406\u548c\u751f\u6210\u53ef\u89c6\u5316...";
      try {
        const data = await fetch("/api/predict", { method: "POST", body: form }).then(async r => {
          if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
          return r.json();
        });
        renderPrediction(data);
      } catch (error) {
        status.innerHTML = `<span class="warn">${error.message}</span>`;
      } finally {
        button.disabled = false;
      }
    }

    async function predictSample(sampleName) {
      const status = document.querySelector("#predict-status");
      status.textContent = `\u6b63\u5728\u8f7d\u5165\u6837\u4f8b ${sampleName} ...`;
      try {
        const data = await fetch(`/api/predict-sample?name=${encodeURIComponent(sampleName)}`).then(async r => {
          if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
          return r.json();
        });
        renderPrediction(data);
      } catch (error) {
        status.innerHTML = `<span class="warn">${error.message}</span>`;
      }
    }

    document.querySelector("#predict").addEventListener("click", predict);
    loadDashboard().then(() => {
      const demo = new URLSearchParams(window.location.search).get("demo");
      if (demo) predictSample(demo);
    }).catch(error => { document.querySelector("#summary").textContent = error.message; });
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse(INDEX_HTML, media_type="text/html; charset=utf-8")


@app.get("/api/dashboard")
def dashboard() -> dict[str, object]:
    return build_dashboard_payload(LSW_ROOT)


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, object]:
    suffix = Path(file.filename or "upload.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        temp_path = Path(handle.name)
        shutil.copyfileobj(file.file, handle)
    try:
        return predict_audio_file(LSW_ROOT, temp_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{exc.__class__.__name__}: {exc}") from exc
    finally:
        temp_path.unlink(missing_ok=True)


@app.get("/api/predict-sample")
def predict_sample(name: str = Query(..., min_length=1)) -> dict[str, object]:
    return _predict_sample_file(name)
