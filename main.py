import os
import base64
import time
import math
import random
from io import BytesIO

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI(title="AI Image Compressor", description="The world's most advanced lossy image compression algorithm")

# ---------------------------------------------------------------------------
# OpenAI client (lazy init so missing key shows a nice error at request time)
# ---------------------------------------------------------------------------

def get_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY environment variable is not set. Please set it and restart the server."
        )
    return OpenAI(api_key=api_key)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class CompressRequest(BaseModel):
    image_base64: str          # data:image/...;base64,<data>  OR  raw base64
    filename: str = "image.jpg"

class CompressResponse(BaseModel):
    prompt: str
    original_bytes: int
    compressed_bytes: int
    compression_ratio: float   # 0-100 %
    compression_speed: str     # fun fake metric
    algorithm: str

class DecompressRequest(BaseModel):
    prompt: str

class DecompressResponse(BaseModel):
    image_url: str
    generation_time_ms: int
    disclaimer: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FAKE_SPEEDS = [
    "42.7 THz quantum tunnelling",
    "∞ bps (theoretical max)",
    "1.21 GW flux-capacitor burst",
    "π × 10⁹ photon/s",
    "299,792,457 m/s (almost light speed!)",
    "9000+ (it's over nine thousand!)",
    "Ludicrous Speed™",
    "Plaid Mode engaged",
]

ALGORITHMS = [
    "NeuroLossy™ v4.2 (patent pending)",
    "QuantumFourier-GPT Transform",
    "Semantic Wavelet Collapse (SWC-7)",
    "Dimensional Reduction via LLM (DRLL)",
    "Stochastic Prompt Entropy Encoding",
    "Heuristic AI Backbone Compression (HABC)",
]

def strip_data_prefix(b64: str) -> str:
    """Remove data:image/...;base64, prefix if present."""
    if "," in b64:
        return b64.split(",", 1)[1]
    return b64

def format_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    elif n < 1024 ** 2:
        return f"{n/1024:.1f} KB"
    elif n < 1024 ** 3:
        return f"{n/1024**2:.2f} MB"
    return f"{n/1024**3:.3f} GB"


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.post("/compress", response_model=CompressResponse)
async def compress(req: CompressRequest):
    client = get_client()

    raw_b64 = strip_data_prefix(req.image_base64)
    try:
        raw_bytes = base64.b64decode(raw_b64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image data.")

    original_bytes = len(raw_bytes)

    # Build data-url for OpenAI vision
    ext = req.filename.rsplit(".", 1)[-1].lower() if "." in req.filename else "jpeg"
    mime_map = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "gif": "gif", "webp": "webp"}
    mime = mime_map.get(ext, "jpeg")
    data_url = f"data:image/{mime};base64,{raw_b64}"

    try:
        vision_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an extremely meticulous image analyst whose job is to compress images "
                        "into text descriptions so detailed that the image can be perfectly reconstructed "
                        "from your words alone. Describe EVERYTHING: subjects, colors (use hex codes when useful), "
                        "lighting, shadows, textures, composition, background, foreground, mood, art style, "
                        "camera angle, depth of field, any text visible, approximate dimensions and positions "
                        "of elements, emotional tone, and any fine details. "
                        "Be systematic: describe center, then edges, then background. "
                        "Target around 200-350 words. Output ONLY the description, no preamble."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Please compress this image into a perfect textual representation:",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url, "detail": "high"},
                        },
                    ],
                },
            ],
            max_tokens=800,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"OpenAI Vision API error: {str(e)}")

    prompt = vision_response.choices[0].message.content.strip()
    compressed_bytes = len(prompt.encode("utf-8"))

    ratio = (1 - compressed_bytes / max(original_bytes, 1)) * 100
    ratio = max(0.0, min(99.9999, ratio))  # clamp

    return CompressResponse(
        prompt=prompt,
        original_bytes=original_bytes,
        compressed_bytes=compressed_bytes,
        compression_ratio=round(ratio, 4),
        compression_speed=random.choice(FAKE_SPEEDS),
        algorithm=random.choice(ALGORITHMS),
    )


@app.post("/decompress", response_model=DecompressResponse)
async def decompress(req: DecompressRequest):
    client = get_client()

    dalle_prompt = (
        "Reconstruct the following image as faithfully as possible based on this compressed textual description. "
        "Render it as a high-quality, photorealistic or stylistically accurate image. "
        "Description:\n\n" + req.prompt
    )

    t0 = time.time()
    try:
        image_response = client.images.generate(
            model="dall-e-3",
            prompt=dalle_prompt,
            size="1024x1024",
            quality="standard",
            n=1,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"DALL-E 3 API error: {str(e)}")

    elapsed_ms = int((time.time() - t0) * 1000)
    image_url = image_response.data[0].url

    return DecompressResponse(
        image_url=image_url,
        generation_time_ms=elapsed_ms,
        disclaimer="以下圖片可能與原圖有些許不同 😇",
    )


# ---------------------------------------------------------------------------
# Frontend — single-page HTML
# ---------------------------------------------------------------------------

HTML = r"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>AI 圖片壓縮器™ — 業界最先進的有損壓縮技術</title>
<style>
  :root {
    --base: #0A0A0F;
    --surface: #14141E;
    --surface2: #1E1E2E;
    --accent: #6366F1;
    --accent2: #818CF8;
    --accent-glow: rgba(99,102,241,0.35);
    --ink: #E8E8F0;
    --ink-dim: #9898B0;
    --danger: #F43F5E;
    --success: #22D3A0;
    --warning: #F59E0B;
    --radius: 14px;
    --radius-sm: 8px;
  }

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: var(--base);
    color: var(--ink);
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    min-height: 100vh;
    overflow-x: hidden;
  }

  /* ── Header ─────────────────────────────────────────────── */
  header {
    text-align: center;
    padding: 3rem 1rem 1.5rem;
    position: relative;
  }
  header::after {
    content: '';
    display: block;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--accent), transparent);
    margin-top: 1.5rem;
    opacity: 0.5;
  }
  .logo-tag {
    display: inline-block;
    background: var(--accent);
    color: #fff;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    padding: 3px 10px;
    border-radius: 99px;
    margin-bottom: 0.75rem;
  }
  h1 {
    font-size: clamp(1.8rem, 5vw, 3rem);
    font-weight: 800;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #fff 30%, var(--accent2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }
  .subtitle {
    color: var(--ink-dim);
    font-size: 0.95rem;
    margin-top: 0.5rem;
  }

  /* ── Layout ─────────────────────────────────────────────── */
  .layout {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1.5rem;
    max-width: 1200px;
    margin: 0 auto;
    padding: 1.5rem 1.5rem 3rem;
  }
  @media (max-width: 768px) { .layout { grid-template-columns: 1fr; } }

  /* ── Cards ──────────────────────────────────────────────── */
  .card {
    background: var(--surface);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: var(--radius);
    padding: 1.5rem;
  }
  .card-title {
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--ink-dim);
    margin-bottom: 1rem;
  }

  /* ── Upload Zone ─────────────────────────────────────────── */
  #upload-zone {
    border: 2px dashed rgba(99,102,241,0.4);
    border-radius: var(--radius-sm);
    padding: 2.5rem 1rem;
    text-align: center;
    cursor: pointer;
    transition: border-color 0.2s, background 0.2s;
    position: relative;
    overflow: hidden;
  }
  #upload-zone.drag-over {
    border-color: var(--accent);
    background: rgba(99,102,241,0.08);
  }
  #upload-zone .upload-icon { font-size: 2.5rem; margin-bottom: 0.5rem; }
  #upload-zone p { color: var(--ink-dim); font-size: 0.9rem; }
  #upload-zone strong { color: var(--ink); }
  #file-input { display: none; }

  /* ── Image Preview ───────────────────────────────────────── */
  .img-wrap {
    position: relative;
    margin-top: 1rem;
    border-radius: var(--radius-sm);
    overflow: hidden;
    background: var(--surface2);
    min-height: 200px;
    display: flex;
    align-items: center;
    justify-content: center;
  }
  .img-wrap img {
    width: 100%;
    height: auto;
    display: block;
    border-radius: var(--radius-sm);
  }
  .img-placeholder {
    color: var(--ink-dim);
    font-size: 0.85rem;
    padding: 3rem;
    text-align: center;
  }

  /* ── Compress Button ─────────────────────────────────────── */
  .btn {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.75rem 1.75rem;
    border-radius: 99px;
    border: none;
    cursor: pointer;
    font-size: 0.95rem;
    font-weight: 600;
    transition: transform 0.15s, box-shadow 0.15s, opacity 0.15s;
  }
  .btn:active { transform: scale(0.97); }
  .btn-primary {
    background: var(--accent);
    color: #fff;
    box-shadow: 0 0 24px var(--accent-glow);
    width: 100%;
    justify-content: center;
    margin-top: 1rem;
  }
  .btn-primary:hover:not(:disabled) { box-shadow: 0 0 40px var(--accent-glow); }
  .btn-primary:disabled { opacity: 0.45; cursor: not-allowed; }
  .btn-secondary {
    background: var(--surface2);
    color: var(--ink);
    border: 1px solid rgba(255,255,255,0.1);
  }

  /* ── Central Compression Ratio Hero ─────────────────────── */
  .ratio-hero {
    text-align: center;
    padding: 2rem 1rem;
    position: relative;
  }
  .ratio-label {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--ink-dim);
    margin-bottom: 0.5rem;
  }
  .ratio-number {
    font-size: clamp(3.5rem, 10vw, 6rem);
    font-weight: 900;
    letter-spacing: -0.04em;
    background: linear-gradient(135deg, var(--accent2), var(--success));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1;
    transition: all 0.5s ease;
  }
  .ratio-number.animating {
    animation: count-up 1.2s ease-out forwards;
  }
  @keyframes count-up {
    from { opacity: 0.3; transform: scale(0.85); }
    to   { opacity: 1;   transform: scale(1); }
  }
  .ratio-sub {
    font-size: 0.85rem;
    color: var(--ink-dim);
    margin-top: 0.35rem;
  }

  /* ── Stats Grid ─────────────────────────────────────────── */
  .stats-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 0.75rem;
    margin: 1rem 0;
  }
  .stat-card {
    background: var(--surface2);
    border-radius: var(--radius-sm);
    padding: 0.85rem 1rem;
    border: 1px solid rgba(255,255,255,0.06);
  }
  .stat-label {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--ink-dim);
    margin-bottom: 0.25rem;
  }
  .stat-value {
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--ink);
  }
  .stat-value.green { color: var(--success); }
  .stat-value.purple { color: var(--accent2); }

  /* ── Progress / Loading ──────────────────────────────────── */
  .progress-wrap {
    display: none;
    margin-top: 1rem;
  }
  .progress-wrap.active { display: block; }
  .progress-bar-track {
    height: 4px;
    background: var(--surface2);
    border-radius: 99px;
    overflow: hidden;
    margin-bottom: 0.6rem;
  }
  .progress-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--accent), var(--success));
    border-radius: 99px;
    width: 0%;
    transition: width 0.4s ease;
  }
  .progress-msg {
    font-size: 0.8rem;
    color: var(--ink-dim);
    text-align: center;
    min-height: 1.2em;
  }

  /* ── Right panel states ──────────────────────────────────── */
  #right-panel .idle-msg {
    color: var(--ink-dim);
    font-size: 0.9rem;
    text-align: center;
    padding: 3rem 1rem;
  }

  /* ── Prompt display ──────────────────────────────────────── */
  .prompt-box {
    background: rgba(99,102,241,0.06);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: var(--radius-sm);
    padding: 1rem;
    font-size: 0.8rem;
    line-height: 1.65;
    color: var(--ink-dim);
    max-height: 180px;
    overflow-y: auto;
    margin-top: 0.75rem;
    white-space: pre-wrap;
    word-break: break-word;
  }
  .prompt-box::-webkit-scrollbar { width: 4px; }
  .prompt-box::-webkit-scrollbar-track { background: transparent; }
  .prompt-box::-webkit-scrollbar-thumb { background: var(--accent); border-radius: 99px; }

  /* ── Algorithm badge ─────────────────────────────────────── */
  .algo-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(99,102,241,0.15);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 99px;
    padding: 4px 12px;
    font-size: 0.72rem;
    color: var(--accent2);
    margin-top: 0.5rem;
    font-weight: 600;
  }

  /* ── Decompress progress ─────────────────────────────────── */
  #decompress-progress {
    display: none;
    text-align: center;
    padding: 3rem 1rem;
  }
  #decompress-progress.active { display: block; }
  .spinner {
    width: 48px; height: 48px;
    border: 3px solid rgba(99,102,241,0.2);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
    margin: 0 auto 1rem;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* ── Comparison section ──────────────────────────────────── */
  #comparison {
    display: none;
    margin-top: 1.5rem;
  }
  #comparison.active { display: block; }
  .compare-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
    align-items: start;
  }
  .compare-col { text-align: center; }
  .compare-col img {
    width: 100%;
    height: auto;
    border-radius: var(--radius-sm);
    border: 1px solid rgba(255,255,255,0.08);
  }
  .compare-label {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
  }
  .compare-label.orig { color: var(--success); }
  .compare-label.regen { color: var(--accent2); }
  .disclaimer {
    text-align: center;
    font-size: 0.8rem;
    color: var(--ink-dim);
    margin-top: 0.75rem;
    padding: 0.5rem 1rem;
    background: rgba(245,158,11,0.08);
    border-radius: var(--radius-sm);
    border: 1px solid rgba(245,158,11,0.2);
  }

  /* ── Error toast ─────────────────────────────────────────── */
  #error-toast {
    display: none;
    position: fixed;
    bottom: 2rem;
    left: 50%;
    transform: translateX(-50%);
    background: #1e0a0e;
    border: 1px solid var(--danger);
    color: var(--danger);
    padding: 0.75rem 1.5rem;
    border-radius: 99px;
    font-size: 0.85rem;
    font-weight: 600;
    z-index: 999;
    max-width: 90vw;
    text-align: center;
    box-shadow: 0 4px 32px rgba(244,63,94,0.3);
  }
  #error-toast.show { display: block; animation: slide-up 0.3s ease; }
  @keyframes slide-up {
    from { transform: translateX(-50%) translateY(20px); opacity: 0; }
    to   { transform: translateX(-50%) translateY(0);    opacity: 1; }
  }

  /* ── Footer ─────────────────────────────────────────────── */
  footer {
    text-align: center;
    padding: 2rem 1rem;
    color: var(--ink-dim);
    font-size: 0.75rem;
    border-top: 1px solid rgba(255,255,255,0.05);
  }
  footer a { color: var(--accent2); text-decoration: none; }
</style>
</head>
<body>

<header>
  <div class="logo-tag">⚡ World's Most Advanced</div>
  <h1>AI 圖片壓縮器™</h1>
  <p class="subtitle">業界最先進的有損壓縮技術 · 基於 GPT-4o 量子神經語義折疊演算法 · 99.97% 壓縮率保證*</p>
</header>

<div class="layout">
  <!-- ── LEFT PANEL ─────────────────────────────────── -->
  <div>
    <div class="card">
      <div class="card-title">📤 上傳原始圖片</div>

      <div id="upload-zone">
        <div class="upload-icon">🖼️</div>
        <p><strong>點擊或拖放圖片</strong></p>
        <p>支援 JPG · PNG · WebP · GIF</p>
        <input type="file" id="file-input" accept="image/*" />
      </div>

      <div class="img-wrap" id="orig-img-wrap">
        <div class="img-placeholder">尚未上傳圖片</div>
      </div>

      <button class="btn btn-primary" id="compress-btn" disabled>
        ⚡ 開始壓縮（量子語義折疊）
      </button>

      <div class="progress-wrap" id="compress-progress">
        <div class="progress-bar-track">
          <div class="progress-bar-fill" id="compress-bar"></div>
        </div>
        <div class="progress-msg" id="compress-msg">初始化量子態...</div>
      </div>
    </div>

    <!-- Stats (hidden until compressed) -->
    <div class="card" id="stats-card" style="display:none; margin-top:1rem;">
      <div class="card-title">📊 壓縮統計報告</div>

      <div class="ratio-hero">
        <div class="ratio-label">壓縮率</div>
        <div class="ratio-number" id="ratio-display">—</div>
        <div class="ratio-sub" id="ratio-sub">等待壓縮...</div>
      </div>

      <div class="stats-grid">
        <div class="stat-card">
          <div class="stat-label">📦 原始大小</div>
          <div class="stat-value" id="stat-orig">—</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">💾 壓縮後大小</div>
          <div class="stat-value green" id="stat-comp">—</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">⚡ 壓縮速度</div>
          <div class="stat-value purple" id="stat-speed" style="font-size:0.8rem;">—</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">🧬 演算法</div>
          <div class="stat-value" id="stat-algo" style="font-size:0.72rem; color:var(--accent2);">—</div>
        </div>
      </div>

      <div style="margin-top:0.75rem;">
        <div class="card-title">🗜️ 壓縮後的「數據流」（即 prompt）</div>
        <div class="prompt-box" id="prompt-display">等待壓縮...</div>
      </div>

      <button class="btn btn-primary" id="decompress-btn" style="margin-top:1rem;">
        🔮 解壓縮（AI 還原）
      </button>
    </div>
  </div>

  <!-- ── RIGHT PANEL ────────────────────────────────── -->
  <div id="right-panel">
    <div class="card">
      <div class="card-title">🔮 AI 解壓縮結果</div>

      <div class="idle-msg" id="right-idle">
        壓縮完成後按「解壓縮」<br />即可體驗業界最先進的有損還原技術 ✨
      </div>

      <div id="decompress-progress">
        <div class="spinner"></div>
        <div class="progress-msg" id="decompress-msg">還原記憶碎片...</div>
      </div>

      <div id="comparison">
        <div class="compare-row">
          <div class="compare-col">
            <div class="compare-label orig">📷 原始圖片</div>
            <img id="compare-orig" src="" alt="原始圖片" />
          </div>
          <div class="compare-col">
            <div class="compare-label regen">🤖 AI 還原版</div>
            <img id="compare-regen" src="" alt="AI 還原圖片" />
          </div>
        </div>
        <div class="disclaimer" id="disclaimer-msg">以下圖片可能與原圖有些許不同 😇</div>
        <div style="text-align:center; margin-top:0.75rem;">
          <span class="algo-badge" id="gen-time">⏱ 生成耗時：—</span>
        </div>
      </div>
    </div>
  </div>
</div>

<div id="error-toast"></div>

<footer>
  <p>* 壓縮率為有損壓縮，圖片將由 GPT-4o 轉換為純文字，再由 DALL-E 3 重新生成。</p>
  <p style="margin-top:0.3rem;">「以最先進的 AI 把你的圖片壓縮成一首詩，再用 AI 畫一張完全不一樣的圖給你。」— r/ProgrammerHumor</p>
</footer>

<script>
  // ── State ──────────────────────────────────────────────────
  let currentFile = null;
  let currentB64 = null;
  let currentPrompt = null;
  let origDataUrl = null;

  // ── DOM refs ───────────────────────────────────────────────
  const uploadZone   = document.getElementById('upload-zone');
  const fileInput    = document.getElementById('file-input');
  const origImgWrap  = document.getElementById('orig-img-wrap');
  const compressBtn  = document.getElementById('compress-btn');
  const compressProg = document.getElementById('compress-progress');
  const compressBar  = document.getElementById('compress-bar');
  const compressMsg  = document.getElementById('compress-msg');
  const statsCard    = document.getElementById('stats-card');
  const ratioDsp     = document.getElementById('ratio-display');
  const ratioSub     = document.getElementById('ratio-sub');
  const statOrig     = document.getElementById('stat-orig');
  const statComp     = document.getElementById('stat-comp');
  const statSpeed    = document.getElementById('stat-speed');
  const statAlgo     = document.getElementById('stat-algo');
  const promptDsp    = document.getElementById('prompt-display');
  const decompressBtn= document.getElementById('decompress-btn');
  const rightIdle    = document.getElementById('right-idle');
  const decompProg   = document.getElementById('decompress-progress');
  const decompMsg    = document.getElementById('decompress-msg');
  const comparison   = document.getElementById('comparison');
  const compareOrig  = document.getElementById('compare-orig');
  const compareRegen = document.getElementById('compare-regen');
  const disclaimerMsg= document.getElementById('disclaimer-msg');
  const genTime      = document.getElementById('gen-time');
  const errorToast   = document.getElementById('error-toast');

  // ── Upload handling ────────────────────────────────────────
  uploadZone.addEventListener('click', () => fileInput.click());
  uploadZone.addEventListener('dragover', e => { e.preventDefault(); uploadZone.classList.add('drag-over'); });
  uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('drag-over'));
  uploadZone.addEventListener('drop', e => {
    e.preventDefault(); uploadZone.classList.remove('drag-over');
    if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
  });
  fileInput.addEventListener('change', () => { if (fileInput.files[0]) handleFile(fileInput.files[0]); });

  function handleFile(file) {
    currentFile = file;
    const reader = new FileReader();
    reader.onload = ev => {
      origDataUrl = ev.target.result;
      currentB64 = origDataUrl; // includes data: prefix
      origImgWrap.innerHTML = `<img src="${origDataUrl}" alt="原始圖片" />`;
      compressBtn.disabled = false;
      // Reset results
      statsCard.style.display = 'none';
      comparison.classList.remove('active');
      rightIdle.style.display = 'block';
      decompProg.classList.remove('active');
      currentPrompt = null;
    };
    reader.readAsDataURL(file);
  }

  // ── Compression flow ───────────────────────────────────────
  const compressMsgs = [
    '正在壓縮量子態...',
    '折疊高維語義空間...',
    '提取神經特徵向量...',
    '量化視覺熵值...',
    '應用霍夫曼-GPT 編碼...',
    '壓縮維度中（第 7 維）...',
    '優化語義基底...',
    '計算壓縮率...',
  ];
  let msgIdx = 0, msgTimer = null;

  function startMsgCycle(msgs, el, bar) {
    msgIdx = 0; el.textContent = msgs[0];
    let prog = 5; bar.style.width = prog + '%';
    msgTimer = setInterval(() => {
      msgIdx = (msgIdx + 1) % msgs.length;
      el.textContent = msgs[msgIdx];
      prog = Math.min(prog + (Math.random() * 12 + 5), 88);
      bar.style.width = prog + '%';
    }, 900);
  }
  function stopMsgCycle(bar) {
    clearInterval(msgTimer);
    bar.style.width = '100%';
  }

  compressBtn.addEventListener('click', async () => {
    if (!currentB64) return;
    compressBtn.disabled = true;
    compressProg.classList.add('active');
    startMsgCycle(compressMsgs, compressMsg, compressBar);

    try {
      const res = await fetch('/compress', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image_base64: currentB64,
          filename: currentFile?.name || 'image.jpg',
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || '壓縮失敗');

      stopMsgCycle(compressBar);
      compressMsg.textContent = '✅ 壓縮完成！';

      // Update stats
      currentPrompt = data.prompt;
      statsCard.style.display = 'block';

      ratioDsp.textContent = data.compression_ratio.toFixed(2) + '%';
      ratioDsp.classList.remove('animating');
      void ratioDsp.offsetWidth;
      ratioDsp.classList.add('animating');
      ratioSub.textContent = `節省了 ${data.compression_ratio.toFixed(4)}% 的儲存空間 🎉`;

      statOrig.textContent = formatBytes(data.original_bytes);
      statComp.textContent = formatBytes(data.compressed_bytes);
      statSpeed.textContent = data.compression_speed;
      statAlgo.textContent = data.algorithm;
      promptDsp.textContent = data.prompt;

    } catch (err) {
      stopMsgCycle(compressBar);
      compressMsg.textContent = '❌ 壓縮失敗';
      showError(err.message);
    } finally {
      compressProg.classList.remove('active');
      compressBtn.disabled = false;
    }
  });

  // ── Decompression flow ─────────────────────────────────────
  const decompMsgs = [
    '還原記憶碎片...',
    '喚醒量子像素...',
    '從虛空召喚圖片...',
    '重建視覺現實...',
    '折疊維度中...',
    '正在聯絡 DALL-E 宇宙...',
    '等待 AI 靈感降臨...',
    '描繪無中生有...',
    '最後潤色（幾乎完成了）...',
  ];

  decompressBtn.addEventListener('click', async () => {
    if (!currentPrompt) return showError('請先壓縮圖片！');
    rightIdle.style.display = 'none';
    comparison.classList.remove('active');
    decompProg.classList.add('active');

    const decompBar = document.createElement('div');
    const fakeBar = { style: { width: '0%' } };
    startMsgCycle(decompMsgs, decompMsg, fakeBar);

    try {
      const res = await fetch('/decompress', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: currentPrompt }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || '解壓縮失敗');

      clearInterval(msgTimer);
      decompMsg.textContent = '✅ 還原完成！';

      // Show comparison
      compareOrig.src = origDataUrl;
      compareRegen.src = data.image_url;
      disclaimerMsg.textContent = data.disclaimer;
      genTime.textContent = `⏱ DALL-E 3 生成耗時：${(data.generation_time_ms/1000).toFixed(1)}s`;

      decompProg.classList.remove('active');
      comparison.classList.add('active');

    } catch (err) {
      clearInterval(msgTimer);
      decompMsg.textContent = '❌ 解壓縮失敗';
      showError(err.message);
      decompProg.classList.remove('active');
      rightIdle.style.display = 'block';
    }
  });

  // ── Utilities ──────────────────────────────────────────────
  function formatBytes(n) {
    if (n < 1024) return n + ' B';
    if (n < 1024*1024) return (n/1024).toFixed(1) + ' KB';
    return (n/1024/1024).toFixed(2) + ' MB';
  }

  let toastTimer = null;
  function showError(msg) {
    errorToast.textContent = '❌ ' + msg;
    errorToast.classList.add('show');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => errorToast.classList.remove('show'), 5000);
  }
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse(content=HTML)
