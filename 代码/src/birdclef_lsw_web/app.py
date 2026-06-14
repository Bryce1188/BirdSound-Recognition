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

app = FastAPI(title="BirdCLEF2026 鸟类声音识别实验系统")
app.mount("/artifacts", StaticFiles(directory=str(LSW_ROOT / "报告" / "实验产物")), name="artifacts")
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
  <title>BirdCLEF2026 鸟类声音识别</title>
  <style>
    :root {
      color-scheme: light;
      --ink: #1f2937;
      --muted: #64748b;
      --line: #d7dee8;
      --soft: #eef4fb;
      --accent: #1d6fa5;
      --accent-2: #2a9d8f;
      --warn: #d90429;
      --bg: #f3f6fb;
      --panel: #ffffff;
    }
    * { box-sizing: border-box; }
    html {
      font-family: "Microsoft YaHei", "Noto Sans CJK SC", "Source Han Sans SC", "PingFang SC",
        "Hiragino Sans GB", "SimHei", "WenQuanYi Micro Hei", "Segoe UI", Arial, sans-serif;
      -webkit-font-smoothing: antialiased;
      text-rendering: optimizeLegibility;
    }
    body { margin: 0; font-family: inherit; color: var(--ink); background: var(--bg); }
    header { padding: 22px 28px 18px; background: linear-gradient(135deg, #ffffff, #eef6fb); border-bottom: 1px solid var(--line); }
    h1 { margin: 0 0 8px; font-size: 28px; letter-spacing: 0; }
    .sub { margin: 0; color: var(--muted); font-size: 14px; line-height: 1.6; }
    main { max-width: 1320px; margin: 0 auto; padding: 18px; display: grid; gap: 16px; }
    section { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 16px; }
    h2 { margin: 0 0 12px; font-size: 18px; }
    .hero-grid, .grid, .figures, .two-col, .sample-grid { display: grid; gap: 14px; }
    .hero-grid { grid-template-columns: repeat(4, minmax(0, 1fr)); margin-top: 18px; }
    .two-col { grid-template-columns: 1.1fr 1fr; }
    .grid { grid-template-columns: repeat(4, minmax(0, 1fr)); }
    .metric { border: 1px solid var(--line); border-radius: 8px; padding: 14px; min-height: 94px; background: linear-gradient(180deg, #ffffff, #fbfdff); }
    .metric strong { display: block; font-size: 12px; color: var(--muted); font-weight: 700; text-transform: uppercase; }
    .metric span { display: block; margin-top: 8px; font-size: 22px; font-weight: 700; }
    .metric small { display: block; margin-top: 8px; color: var(--muted); line-height: 1.5; }
    .banner { border: 1px solid var(--line); border-radius: 8px; background: var(--soft); padding: 12px 14px; font-size: 14px; line-height: 1.7; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th, td { border-bottom: 1px solid var(--line); padding: 8px; text-align: left; vertical-align: top; }
    th { background: #eef3f8; color: #334155; }
    button { border: 0; border-radius: 6px; padding: 10px 14px; background: var(--accent); color: white; cursor: pointer; font-weight: 600; }
    button:disabled { opacity: 0.55; cursor: wait; }
    input[type=file] { display: block; margin: 8px 0 12px; }
    .figures { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .sample-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .figure-card, .sample-card, .rank-card { border: 1px solid var(--line); border-radius: 8px; padding: 10px; background: #fff; }
    .figure-card img, .prediction-figure img { width: 100%; border: 1px solid var(--line); border-radius: 6px; background: #fff; }
    .figure-card strong, .sample-card strong { display: block; margin-bottom: 8px; font-size: 13px; }
    .sample-card audio { width: 100%; margin-top: 8px; }
    .rank-list { display: grid; gap: 10px; }
    .rank-card { padding: 12px; }
    .rank-head { display: flex; justify-content: space-between; gap: 12px; font-size: 14px; font-weight: 700; }
    .rank-meta { margin-top: 6px; color: var(--muted); font-size: 12px; }
    .bar { height: 8px; margin-top: 10px; border-radius: 999px; background: #e5edf6; overflow: hidden; }
    .bar > span { display: block; height: 100%; background: linear-gradient(90deg, var(--accent), var(--accent-2)); }
    .prediction-figure { display: grid; gap: 12px; grid-template-columns: repeat(3, minmax(0, 1fr)); margin-top: 14px; }
    .pill { display: inline-block; padding: 3px 8px; border-radius: 999px; background: #e8f4ec; color: #1f7a4b; font-size: 12px; font-weight: 700; }
    .status { color: var(--muted); font-size: 13px; }
    .ok { color: #1f7a4b; font-weight: 700; }
    .warn { color: var(--warn); font-weight: 700; }
    .mono { font-family: Consolas, "SFMono-Regular", "Microsoft YaHei", "Noto Sans CJK SC", monospace; }
    @media (max-width: 1040px) { .hero-grid, .grid, .two-col, .prediction-figure { grid-template-columns: 1fr; } }
    @media (max-width: 860px) { .figures, .sample-grid { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <header>
    <h1>BirdCLEF2026 鸟类声音识别实验系统</h1>
    <p class="sub">FastAPI 页面 · 非 Gradio · BirdCLEF2026 声景数据、传统机器学习与深度学习对照实验、排名类指标与可解释声学特征展示</p>
    <div id="hero" class="hero-grid"></div>
  </header>
  <main>
    <section class="two-col">
      <div>
        <h2>音频识别</h2>
        <div class="banner">
          页面用于课程验收演示：上传鸟鸣音频后，系统输出 Top-5 预测类别、波形、Log-Mel 频谱和 MFCC 特征图。
        </div>
        <input id="file" type="file" accept=".wav,.ogg,.mp3,.flac,audio/*" />
        <button id="predict">上传并识别</button>
        <p id="predict-status" class="status">请选择一段鸟鸣音频。</p>
        <div id="prediction"></div>
      </div>
      <div>
        <h2>演示样例音频</h2>
        <div id="sample-audio" class="sample-grid"></div>
      </div>
    </section>
    <section>
      <h2>预测可视化</h2>
      <div id="prediction-figures" class="prediction-figure"></div>
    </section>
    <section>
      <h2>实验指标看板</h2>
      <div id="summary-banner" class="banner"></div>
      <div id="summary" class="grid"></div>
      <div id="models"></div>
    </section>
    <section class="two-col">
      <div>
        <h2>阈值优化结果</h2>
        <div id="thresholds"></div>
      </div>
      <div>
        <h2>老师要求核对</h2>
        <div id="checklist"></div>
      </div>
    </section>
    <section>
      <h2>实验图表</h2>
      <div id="figures" class="figures"></div>
    </section>
  </main>
  <script>
    const escapeHtml = (value) => String(value ?? "").replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");
    const fmt = (value) => Number.isFinite(Number(value)) ? Number(value).toFixed(4) : String(value ?? "");
    async function loadDashboard() {
      const data = await fetch("/api/dashboard").then(r => r.json());
      const models = [...(data.models || [])].sort((a, b) => Number(b.lrap || 0) - Number(a.lrap || 0));
      const best = models[0] || {};
      const manifest = data.manifest || {};
      document.querySelector("#hero").innerHTML = `
        <div class="metric"><strong>实验主题</strong><span>Bird Sound Recognition</span><small>BirdCLEF2026 course project</small></div>
        <div class="metric"><strong>数据集</strong><span>${escapeHtml(data.dataset)}</span><small>${escapeHtml(manifest.sample_count || "")} 个样本窗口 / ${escapeHtml(manifest.label_column_count || "")} 个鸟类标签</small></div>
        <div class="metric"><strong>当前状态</strong><span>${escapeHtml(data.status)}</span><small>生成时间：${escapeHtml(data.generated_at || "")}</small></div>
        <div class="metric"><strong>最优传统模型</strong><span>${escapeHtml(manifest.best_classical_model || "N/A")}</span><small>指标重点：${escapeHtml((manifest.metric_focus || []).join(" / "))}</small></div>`;
      document.querySelector("#summary-banner").innerHTML = `当前主结果以 <span class="pill">${escapeHtml(manifest.best_classical_model || "extra_trees")}</span> 为主，深度模型作为对照实验。页面展示 LRAP、Micro F1、Top-3 命中率和 Hamming Loss 等课程实验指标。`;
      document.querySelector("#summary").innerHTML = `
        <div class="metric"><strong>最佳 LRAP</strong><span>${fmt(best.lrap)}</span><small>${escapeHtml(best.model || "")}</small></div>
        <div class="metric"><strong>最佳 Micro F1</strong><span>${fmt(best.micro_f1)}</span><small>Top-3 命中率 ${fmt(best.top3_hit_rate)}</small></div>
        <div class="metric"><strong>最优阈值</strong><span>${fmt(best.best_threshold)}</span><small>Hamming Loss ${fmt(best.hamming_loss)}</small></div>
        <div class="metric"><strong>特征维度</strong><span>${escapeHtml(manifest.feature_column_count || "161")}</span><small>MFCC / pitch / volume / timbre / onset</small></div>`;
      const rows = models.map(row => `
        <tr><td>${escapeHtml(row.model)}</td><td>${fmt(row.lrap)}</td><td>${fmt(row.micro_f1)}</td><td>${fmt(row.macro_f1)}</td><td>${fmt(row.top1_hit_rate)}</td><td>${fmt(row.top3_hit_rate)}</td><td>${fmt(row.hamming_loss)}</td><td>${fmt(row.best_threshold)}</td></tr>`).join("");
      document.querySelector("#models").innerHTML = `<table><thead><tr><th>模型</th><th>LRAP</th><th>Micro F1</th><th>Macro F1</th><th>Top-1</th><th>Top-3</th><th>Hamming Loss</th><th>阈值</th></tr></thead><tbody>${rows}</tbody></table>`;
      const thresholdRows = (data.thresholds || []).map(row => `
        <tr><td>${escapeHtml(row.model)}</td><td>${fmt(row.threshold)}</td><td>${fmt(row.lrap)}</td><td>${fmt(row.micro_f1)}</td><td>${fmt(row.prediction_density)}</td></tr>`).join("");
      document.querySelector("#thresholds").innerHTML = `<table><thead><tr><th>模型</th><th>阈值</th><th>LRAP</th><th>Micro F1</th><th>预测密度</th></tr></thead><tbody>${thresholdRows}</tbody></table>`;
      const checks = (data.teacher_checklist || []).map(item => `
        <tr><td>${escapeHtml(item.requirement)}</td><td class="ok">${escapeHtml(item.status)}</td><td>${escapeHtml(item.evidence)}</td></tr>`).join("");
      document.querySelector("#checklist").innerHTML = `<table><thead><tr><th>要求</th><th>状态</th><th>证据</th></tr></thead><tbody>${checks}</tbody></table>`;
      document.querySelector("#figures").innerHTML = (data.figures || []).map(fig => `<div class="figure-card"><strong>${escapeHtml(fig.name)}</strong><img src="${fig.url}" alt="${escapeHtml(fig.name)}" /></div>`).join("");
      document.querySelector("#sample-audio").innerHTML = (data.sample_audio || []).map(audio => `
        <div class="sample-card">
          <strong>${escapeHtml(audio.name)}</strong>
          <button type="button" data-sample="${escapeHtml(audio.name)}">使用该样例识别</button>
          <audio controls preload="none" src="${audio.url}"></audio>
        </div>`).join("");
      document.querySelectorAll("[data-sample]").forEach(button => {
        button.addEventListener("click", () => predictSample(button.dataset.sample));
      });
    }
    function renderPrediction(data) {
      document.querySelector("#prediction").innerHTML = `<div class="rank-list">${data.top_predictions.map(row => `
          <div class="rank-card">
            <div class="rank-head"><span>${escapeHtml(row.label)}</span><span>${fmt(row.score)}</span></div>
            <div class="bar"><span style="width:${Math.max(2, Math.round(Number(row.score || 0) * 100))}%"></span></div>
            <div class="rank-meta">${escapeHtml(row.common_name)} | ${escapeHtml(row.scientific_name)} | ${escapeHtml(row.class_name)}</div>
          </div>`).join("")}</div>`;
      document.querySelector("#prediction-figures").innerHTML = Object.entries(data.figures).map(([name, url]) => `<div class="figure-card"><strong>${escapeHtml(name)}</strong><img src="${url}" alt="${escapeHtml(name)}" /></div>`).join("");
      document.querySelector("#predict-status").textContent = `模型：${data.model}；窗口数：${data.window_count}；时长：${data.duration_seconds}s`;
    }
    async function predict() {
      const input = document.querySelector("#file");
      const status = document.querySelector("#predict-status");
      const button = document.querySelector("#predict");
      if (!input.files.length) { status.textContent = "请先选择音频文件。"; return; }
      const form = new FormData();
      form.append("file", input.files[0]);
      button.disabled = true;
      status.textContent = "正在推理和生成可视化...";
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
      status.textContent = `正在载入样例 ${sampleName} ...`;
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
    loadDashboard()
      .then(() => {
        const params = new URLSearchParams(window.location.search);
        const demo = params.get("demo");
        if (demo) predictSample(demo);
      })
      .catch(error => { document.querySelector("#summary").textContent = error.message; });
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX_HTML


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
