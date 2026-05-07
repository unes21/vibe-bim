"""
app.py — VIBE Backend (Flask API Server)
=========================================
Voice Interface for BIM Environments (VIBE)
https://github.com/unes21/vibe-bim

Overview
--------
This Flask application serves as the middleware between the browser-based
voice interface and the Autodesk Revit / Dynamo execution layer.

Architecture
------------
Browser  →  POST /api/llm/category   →  NLP intent resolution
         →  POST /api/llm/mode       →  write-mode resolution (append / overwrite)
         →  POST /api/write          →  mark elements as pending in intent.json
                                         ↓
                              Dynamo (VIBE_executor.dyn) polls intent.json
                              and applies parameter writes inside Revit.

Two-tier intent resolution cascade
----------------------------------
    1. Rule-based keyword matching         (O(|V|), <0.1 ms in benchmark, always available)
    2. Ollama-hosted local LLM (offline)   (requires local ollama daemon)

If both tiers fail, the system returns source="unresolved" with a
descriptive error field — never a silent stall. The architecture is
fully self-contained: no external cloud dependency, no API key required,
no network egress beyond the local Ollama daemon. This is the design
choice that distinguishes VIBE from cloud-only systems and is required
for production BIM environments where data sovereignty matters.

File-lock protocol
------------------
Flask and Dynamo share a single JSON file (intent.json).  Concurrent access
is managed via an atomic lock file (intent.json.lock), using os.O_CREAT |
os.O_EXCL to ensure mutually exclusive writes on both Windows and Unix.

Configuration
-------------
    JSON_PATH           — set via VIBE_JSON_PATH env var (default below)
    OLLAMA_URL          — Ollama HTTP endpoint (default: http://localhost:11434)
    OLLAMA_MODEL        — Ollama model name (default: llama3.1:8b)
    OLLAMA_TIMEOUT_SEC  — Ollama request timeout (default: 30)
    VIBE_LOG_PATH       — benchmark CSV log path (default: ./vibe_bench.csv)

Usage
-----
    pip install flask requests python-dotenv
    # Install ollama separately (https://ollama.ai) and run:
    #     ollama pull llama3.1:8b
    #     ollama serve     (auto-starts on Windows install)
    python app.py

Author  : Ayberk Enis
Project : VIBE - Voice Interface for BIM Environments
License : MIT
"""

from flask import Flask, render_template_string, jsonify, request
import csv
import json
import os
import time
from datetime import datetime

# ---------------------------------------------------------------------------
# Optional .env loader — reads VIBE_JSON_PATH, OLLAMA_URL, OLLAMA_MODEL,
# and any other VIBE_* variables from a .env file next to app.py.
# This avoids the need to set environment variables manually in PowerShell
# every time the server is restarted. If python-dotenv is not installed
# the system silently falls back to OS-level environment variables.
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    _here = os.path.dirname(os.path.abspath(__file__))
    load_dotenv(os.path.join(_here, ".env"))
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Optional dependencies — system degrades gracefully if absent
# ---------------------------------------------------------------------------
try:
    import requests  # used for Ollama HTTP calls
except ImportError:
    requests = None

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
JSON_PATH = os.getenv(
    "VIBE_JSON_PATH",
    r"C:\ProjectX\revit_data.json"   # fallback — override via env var
)
LOCK_PATH = JSON_PATH + ".lock"

OLLAMA_URL        = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL      = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_TIMEOUT    = float(os.getenv("OLLAMA_TIMEOUT_SEC", "30"))

LOG_PATH = os.getenv("VIBE_LOG_PATH", os.path.join(os.getcwd(), "vibe_bench.csv"))

# ---------------------------------------------------------------------------
# Category taxonomy
# ---------------------------------------------------------------------------
CATEGORY_KEYS = [
    "roof", "wall", "floor", "door", "window",
    "ceiling", "stair", "column", "beam", "railing",
    "room", "furniture", "light",
]

CATEGORY_MAP = {
    "floor":     ["floors", "floor", "döşemeler", "döşeme", "zemin döşemeleri", "zemin"],
    "wall":      ["walls", "wall", "basic wall", "duvarlar", "duvar"],
    "door":      ["doors", "door", "kapılar", "kapı"],
    "roof":      ["roofs", "roof", "çatılar", "çatı", "cati"],
    "window":    ["windows", "window", "pencereler", "pencere"],
    "ceiling":   ["ceilings", "ceiling", "tavanlar", "tavan"],
    "stair":     ["stairs", "stair", "merdivenler", "merdiven"],
    "column":    ["structural columns", "columns", "column", "yapısal kolonlar", "kolonlar"],
    "beam":      ["structural framing", "yapısal çerçeveleme", "beam", "kiriş", "kiris"],
    "railing":   ["railings", "railing", "korkuluklar", "korkuluk"],
    "room":      ["rooms", "room", "odalar", "oda"],
    "furniture": ["furniture", "specialty equipment", "mobilyalar", "mobilya"],
    "light":     ["lighting fixtures", "lights", "light fixtures", "aydınlatma", "aydinlatma"],
}

# ---------------------------------------------------------------------------
# File-lock helpers
# ---------------------------------------------------------------------------

def acquire_lock(timeout_sec: float = 2.0, poll: float = 0.05) -> bool:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            fd = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            return True
        except FileExistsError:
            time.sleep(poll)
    return False


def release_lock() -> None:
    try:
        if os.path.exists(LOCK_PATH):
            os.remove(LOCK_PATH)
    except OSError:
        pass


def safe_read_json(max_retry: int = 12, poll: float = 0.05) -> dict:
    if not os.path.exists(JSON_PATH):
        return {}
    for _ in range(max_retry):
        if not acquire_lock(timeout_sec=1.5, poll=poll):
            time.sleep(poll)
            continue
        try:
            with open(JSON_PATH, "r", encoding="utf-8") as f:
                content = f.read().strip()
            if not content:
                return {}
            return json.loads(content)
        except json.JSONDecodeError:
            time.sleep(poll)
        finally:
            release_lock()
    return {}


def safe_write_json(data: dict) -> None:
    if not acquire_lock(timeout_sec=2.0, poll=0.05):
        raise RuntimeError(
            "Cannot acquire file lock for writing "
            "(Dynamo may be holding it — retry shortly)."
        )
    try:
        tmp = JSON_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        os.replace(tmp, JSON_PATH)   # atomic on both Windows and Unix
    finally:
        release_lock()


# ---------------------------------------------------------------------------
# Category matching helper
# ---------------------------------------------------------------------------

def match_category(entry, category_key: str) -> bool:
    """
    Return True if a JSON *entry* (dict) belongs to *category_key*.

    Resolution priority:
      1. entry["cat_key"]       — direct match (preferred; populated by the
                                  Dynamo ingest graph for racbasicsampleproject)
      2. entry["tip"]           — Turkish field name used by VIBE_executor.dyn
      3. entry["element_type"]  — legacy English field name (older ingest graphs)

    The first non-empty field is matched against CATEGORY_MAP aliases.
    """
    if not isinstance(entry, dict):
        return False

    # Priority 1: direct cat_key match (set by ingest graph)
    direct = (entry.get("cat_key") or "").strip().lower()
    if direct == category_key.lower():
        return True

    # Priority 2/3: element type string from either schema
    raw = entry.get("tip") or entry.get("element_type") or ""
    normalized = raw.strip().lower()
    if not normalized:
        return False
    for alias in CATEGORY_MAP.get(category_key, []):
        alias = alias.lower()
        if normalized == alias or normalized.startswith(alias):
            return True
    return False


# ---------------------------------------------------------------------------
# Tier 1: Rule-based intent extraction
# ---------------------------------------------------------------------------

def rule_extract_category(text: str):
    t = (text or "").lower()
    rules = [
        ("roof",     ["çatı", "cati", "roof"]),
        ("wall",     ["duvar", "wall"]),
        ("floor",    ["zemin", "döşeme", "doseme", "floor"]),
        ("door",     ["kapı", "kapi", "door"]),
        ("window",   ["pencere", "window"]),
        ("ceiling",  ["tavan", "ceiling"]),
        ("stair",    ["merdiven", "stair"]),
        ("column",   ["kolon", "column"]),
        ("beam",     ["kiriş", "kiris", "beam"]),
        ("railing",  ["korkuluk", "railing"]),
        ("room",     ["oda", "room"]),
        ("furniture",["mobilya", "furniture"]),
        ("light",    ["aydınlatma", "aydinlatma", "light"]),
    ]
    for key, keywords in rules:
        if any(kw in t for kw in keywords):
            return key
    return None


def rule_extract_mode(text: str):
    t = (text or "").lower()
    append_keywords    = ["yanına", "yanina", "ekle", "üzerine ekle", "ustune ekle", "append"]
    overwrite_keywords = ["sil", "sıfır", "sifir", "baştan", "bastan", "yeniden", "overwrite"]
    if any(kw in t for kw in append_keywords):
        return "append"
    if any(kw in t for kw in overwrite_keywords):
        return "overwrite"
    return None


# ---------------------------------------------------------------------------
# Tier 2: Ollama local LLM (offline fallback)
# ---------------------------------------------------------------------------

def _ollama_call(prompt: str) -> str | None:
    """
    Call a locally-hosted Ollama model via its HTTP API.

    Ollama's /api/chat endpoint returns a JSON body with a message.content
    field. We set format="json" to encourage JSON-only output for reliable
    downstream parsing.

    Returns the raw content string, or None on any error / unavailability.
    """
    if requests is None:
        return None
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "format": "json",
                "options": {"temperature": 0},
            },
            timeout=OLLAMA_TIMEOUT,
        )
        if resp.status_code != 200:
            return None
        body = resp.json()
        msg = body.get("message", {})
        content = msg.get("content")
        if isinstance(content, str):
            return content.strip()
        return None
    except Exception:
        return None


def _ollama_available() -> bool:
    """Lightweight probe — used by /api/health, does not generate."""
    if requests is None:
        return False
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2.0)
        return r.status_code == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Cascade orchestrator — category
# ---------------------------------------------------------------------------

def llm_extract_category(text: str) -> dict:
    """
    Two-tier category resolution cascade:
        rules → ollama → None

    The 'source' field always reports which tier produced the result, so
    downstream logging can measure tier coverage and offline-fallback
    effectiveness independently.
    """
    # --- Tier 1: rules ---
    cat = rule_extract_category(text)
    if cat:
        return {"category": cat, "source": "rules"}

    prompt = (
        "Return JSON only. No explanation.\n"
        f"Allowed categories: {CATEGORY_KEYS}\n"
        "Extract the most appropriate BIM category from the user sentence.\n"
        'Example output: {"category":"roof"} or {"category":null}\n\n'
        f'User: "{text}"'
    )

    # --- Tier 2: Ollama local ---
    raw = _ollama_call(prompt)
    if raw:
        try:
            cat = json.loads(raw).get("category")
            if cat in CATEGORY_KEYS:
                return {"category": cat, "source": "ollama"}
        except json.JSONDecodeError:
            pass

    return {"category": None, "source": "unresolved"}


# ---------------------------------------------------------------------------
# Cascade orchestrator — write mode
# ---------------------------------------------------------------------------

def llm_extract_mode(text: str) -> dict:
    """
    Two-tier write-mode resolution cascade with append fallback:
        rules → ollama → append (non-destructive default)
    """
    # --- Tier 1: rules ---
    mode = rule_extract_mode(text)
    if mode:
        return {"mode": mode, "source": "rules"}

    prompt = (
        "Return JSON only. No explanation.\n"
        "append = add after existing value\n"
        "overwrite = replace existing value\n"
        'Example: {"mode":"append"} or {"mode":null}\n\n'
        f'User: "{text}"'
    )

    # --- Tier 2: Ollama local ---
    raw = _ollama_call(prompt)
    if raw:
        try:
            mode = json.loads(raw).get("mode")
            if mode in ("append", "overwrite"):
                return {"mode": mode, "source": "ollama"}
        except json.JSONDecodeError:
            pass

    # --- Tier 3: non-destructive default ---
    return {"mode": "append", "source": "default_append"}


# ---------------------------------------------------------------------------
# Benchmark logging (opt-in, CSV)
# ---------------------------------------------------------------------------

_LOG_HEADER = [
    "timestamp_iso", "endpoint", "input_text",
    "resolved_value", "source", "latency_ms",
]


def _append_log_row(row: list) -> None:
    """Append a row to the benchmark CSV, creating the file if absent."""
    try:
        new_file = not os.path.exists(LOG_PATH)
        with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if new_file:
                writer.writerow(_LOG_HEADER)
            writer.writerow(row)
    except Exception:
        pass  # benchmark logging must never break the main flow


# ---------------------------------------------------------------------------
# Frontend (single-page voice interface) — unchanged
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    html = """<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>VIBE Revit Voice Agent</title>
<style>
  body{background:#0a0b10;color:#e2e8f0;font-family:Inter,system-ui,Arial;margin:0}
  .wrap{max-width:980px;margin:30px auto;padding:20px}
  .card{background:#151921;border:1px solid #273244;border-radius:16px;padding:16px;margin-bottom:14px}
  .row{display:flex;gap:12px;flex-wrap:wrap;align-items:center}
  .btn{background:#f63366;border:none;color:white;border-radius:12px;padding:12px 14px;font-weight:800;cursor:pointer}
  .btn:disabled{opacity:.5;cursor:not-allowed}
  .btn2{background:#1e293b;border:1px solid #334155;color:#cbd5e1;border-radius:12px;padding:10px 12px;cursor:pointer}
  .pill{display:inline-block;padding:6px 10px;border-radius:999px;background:#0f172a;border:1px solid #273244;font-size:12px;color:#94a3b8}
  #chat{display:flex;flex-direction:column;gap:10px;min-height:220px;max-height:360px;overflow:auto}
  .msg{max-width:80%;padding:10px 12px;border-radius:14px;line-height:1.35;font-size:14px}
  .ai{align-self:flex-start;background:#f63366;color:white}
  .user{align-self:flex-end;background:#334155;color:white}
  .hint{font-size:12px;color:#94a3b8}
  .danger{color:#fda4af}
</style>
</head>
<body>
<div class="wrap">
  <h2 style="margin:0 0 6px 0;color:#f63366">VIBE Revit Voice Agent</h2>
  <div class="hint">Akış: "çatı için not ekle" → "not nedir?" → "yanına mı / sıfırdan mı?" → Dynamo'ya gönder.</div>

  <div class="card">
    <div class="row">
      <span class="pill" id="statePill">State: idle</span>
      <span class="pill" id="catPill">Kategori: -</span>
      <span class="pill" id="modePill">Mode: -</span>
      <span class="pill" id="ttsPill">TTS: -</span>
      <span class="pill" id="lockPill">🔒 0</span>
      <span class="pill" id="llmPill">LLM: -</span>
      <span class="pill" id="timerPill">⏱ -</span>
    </div>

    <div style="height:10px"></div>
    <div id="chat"></div>

    <div style="height:12px"></div>
    <div class="row">
      <button class="btn" id="talkBtn" onclick="startConversation()">🎤 Konuş</button>
      <button class="btn2" onclick="stopAll()">⏹️ Durdur</button>
      <button class="btn2" onclick="hardRefresh()">🔄 Hard refresh</button>
    </div>

    <div class="hint" style="margin-top:10px">
      Chrome önerilir. Mikrofon izni ver. TTS hata verirse yine de sistem çalışır.
    </div>
    <div class="hint danger" id="err"></div>
  </div>
</div>

<script>
const chat      = document.getElementById("chat");
const err       = document.getElementById("err");
const statePill = document.getElementById("statePill");
const catPill   = document.getElementById("catPill");
const modePill  = document.getElementById("modePill");
const ttsPill   = document.getElementById("ttsPill");
const lockPill  = document.getElementById("lockPill");
const llmPill   = document.getElementById("llmPill");
const timerPill = document.getElementById("timerPill");
const talkBtn   = document.getElementById("talkBtn");

function addMsg(text, who){
  const d = document.createElement("div");
  d.className = "msg " + who;
  d.textContent = text;
  chat.appendChild(d);
  chat.scrollTop = chat.scrollHeight;
}
function setState(s){ statePill.textContent = "State: " + s; }
function setCat(c)  { catPill.textContent   = "Kategori: " + (c || "-"); }
function setMode(m) { modePill.textContent  = "Mode: " + (m || "-"); }
function setLLM(s)  { llmPill.textContent   = "LLM: " + (s || "-"); }
function setTimer(s){ timerPill.textContent = "⏱ " + s; }

function hardRefresh(){ location.href = location.pathname + "?v=" + Date.now(); }

let recognition = null;

function stopAll(){
  try{ if(recognition) recognition.abort(); }catch(e){}
  try{ window.speechSynthesis.cancel(); }catch(e){}
  setState("idle");
  talkBtn.disabled = false;
}

function speakTR(text){
  return new Promise((resolve) => {
    try{
      if(!("speechSynthesis" in window)){ ttsPill.textContent="TTS: n/a"; resolve(); return; }
      ttsPill.textContent="TTS: speaking";
      window.speechSynthesis.cancel();
      try{ window.speechSynthesis.resume(); }catch(e){}
      const u = new SpeechSynthesisUtterance(text);
      u.lang = "tr-TR";
      u.rate = 1.0;
      u.onend = () => { ttsPill.textContent="TTS: idle"; resolve(); };
      u.onerror = () => {
        try{
          const u2 = new SpeechSynthesisUtterance(text);
          u2.lang = "en-US";
          u2.onend = () => { ttsPill.textContent="TTS: idle"; resolve(); };
          u2.onerror = () => { ttsPill.textContent="TTS: error"; resolve(); };
          window.speechSynthesis.speak(u2);
        }catch(e2){ ttsPill.textContent="TTS: error"; resolve(); }
      };
      window.speechSynthesis.speak(u);
    }catch(e){ ttsPill.textContent="TTS: error"; resolve(); }
  });
}

function listenOnceTR(){
  return new Promise((resolve, reject) => {
    err.textContent = "";
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if(!SR){ reject(new Error("SpeechRecognition yok. Chrome kullan.")); return; }
    recognition = new SR();
    recognition.lang = "tr-TR";
    recognition.interimResults = false;
    recognition.continuous = false;
    recognition.onresult = (e) => resolve(e.results[0][0].transcript.trim());
    recognition.onerror  = (e) => reject(new Error("Mikrofon hata: " + e.error));
    recognition.start();
  });
}

async function apiData(){
  const res = await fetch("/api/data");
  return await res.json();
}

async function refreshLocks(){
  try{
    const data = await apiData();
    let locked = 0;
    for(const k of Object.keys(data||{})){
      const o = data[k];
      if(o && typeof o === "object" && o.Kilit) locked++;
    }
    lockPill.textContent = "🔒 " + locked;
  }catch(e){}
}
setInterval(refreshLocks, 1500);
refreshLocks();

async function apiExtractCategory(text){
  const res = await fetch("/api/llm/category", {
    method:"POST", headers:{"Content-Type":"application/json"},
    body: JSON.stringify({text})
  });
  return await res.json();
}
async function apiExtractMode(text){
  const res = await fetch("/api/llm/mode", {
    method:"POST", headers:{"Content-Type":"application/json"},
    body: JSON.stringify({text})
  });
  return await res.json();
}
async function apiWrite(cat, note, mode, source, duration){
  const res = await fetch("/api/write", {
    method:"POST", headers:{"Content-Type":"application/json"},
    body: JSON.stringify({cat, note, mode, source, duration})
  });
  return await res.json();
}

async function startConversation(){
  talkBtn.disabled = true;
  const t0 = Date.now();
  try{
    setState("listen_command");
    addMsg("Dinliyorum… (örn: 'çatı için not ekle')", "ai");
    await speakTR("Dinliyorum. Örnek: çatı için not ekle.");

    const cmd = await listenOnceTR();
    addMsg(cmd, "user");

    setState("category");
    const catObj = await apiExtractCategory(cmd);
    setLLM((catObj.source || "-") + (catObj.error ? " | " + catObj.error : ""));
    const cat = catObj.category || null;

    if(!cat){
      const duration = (Date.now() - t0) / 1000;
      setTimer(duration.toFixed(1) + "s");
      // Failure'ı logla
      await fetch("/api/log", {
        method:"POST", headers:{"Content-Type":"application/json"},
        body: JSON.stringify({command: cmd, category: null, source: catObj.source, success: false, duration, note: "kategori bulunamadi"})
      });
      setState("idle");
      addMsg("Kategoriyi anlayamadım. 'çatı', 'duvar', 'zemin' gibi tekrar söyle.", "ai");
      await speakTR("Kategoriyi anlayamadım. Çatı, duvar, zemin gibi tekrar söyle.");
      talkBtn.disabled = false;
      return;
    }
    setCat(cat);

    setState("ask_note");
    addMsg("Not nedir?", "ai");
    await speakTR("Not nedir? Söyleyebilirsin.");
    const note = await listenOnceTR();
    addMsg(note, "user");

    setState("ask_mode");
    addMsg("Yanına mı ekleyeyim, yoksa sıfırdan mı?", "ai");
    await speakTR("Yanına mı ekleyeyim, yoksa sıfırdan mı?");
    const modeAns = await listenOnceTR();
    addMsg(modeAns, "user");

    setState("mode");
    const modeObj = await apiExtractMode(modeAns);
    setLLM((modeObj.source || "-") + (modeObj.error ? " | " + modeObj.error : ""));
    let mode = modeObj.mode || null;

    if(!mode){
      mode = "append";
      addMsg("Cevabı net anlayamadım. Varsayılan: yanına ekliyorum.", "ai");
      await speakTR("Cevabı net anlayamadım. Varsayılan olarak yanına ekliyorum.");
    }
    setMode(mode);

    setState("sending");
    addMsg("Dynamo'ya gönderiyorum…", "ai");
    await speakTR("Tamam. Dynamo'ya gönderiyorum.");

    const duration = (Date.now() - t0) / 1000;
    setTimer(duration.toFixed(1) + "s");

    const out = await apiWrite(cat, note, mode, catObj.source, duration);
    addMsg(out.msg || "Gönderildi.", "ai");
    await speakTR(out.msg || "Gönderildi.");

    setState("done");
    refreshLocks();
    talkBtn.disabled = false;

  }catch(e){
    const duration = (Date.now() - t0) / 1000;
    setTimer(duration.toFixed(1) + "s");
    err.textContent = String(e.message || e);
    addMsg("Hata: " + (e.message || e), "ai");
    try{ await speakTR("Bir hata oluştu. Mikrofon izni ve ses ayarlarını kontrol et."); }catch(_){}
    setState("idle");
    talkBtn.disabled = false;
  }
}
</script>
</body>
</html>"""
    return render_template_string(html), 200, {
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Pragma": "no-cache",
    }

# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@app.route("/api/llm/category", methods=["POST"])
def api_llm_category():
    """Resolve BIM category from a natural-language command string."""
    text = (request.json or {}).get("text", "").strip()
    t0 = time.perf_counter()
    result = llm_extract_category(text)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    result["latency_ms"] = round(latency_ms, 2)
    _append_log_row([
        datetime.utcnow().isoformat(timespec="seconds"),
        "category", text, result.get("category"),
        result.get("source"), round(latency_ms, 2),
    ])
    return jsonify(result)


@app.route("/api/llm/mode", methods=["POST"])
def api_llm_mode():
    """Resolve write mode (append / overwrite) from a natural-language string."""
    text = (request.json or {}).get("text", "").strip()
    t0 = time.perf_counter()
    result = llm_extract_mode(text)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    result["latency_ms"] = round(latency_ms, 2)
    _append_log_row([
        datetime.utcnow().isoformat(timespec="seconds"),
        "mode", text, result.get("mode"),
        result.get("source"), round(latency_ms, 2),
    ])
    return jsonify(result)


@app.route("/api/data")
def api_data():
    return jsonify(safe_read_json())


@app.route("/api/write", methods=["POST"])
def api_write():
    body = request.json or {}
    cat  = body.get("cat")
    note = (body.get("note") or "").strip()
    mode = body.get("mode") or "overwrite"

    if not cat or not note:
        return jsonify({"msg": "Missing 'cat' or 'note' in request body.", "count": 0}), 400

    data = safe_read_json()
    count = 0
    updated_ids = []
    timestamp = datetime.now().strftime("%H:%M")
    value = f"{note} ({timestamp})"

    for element_id, entry in data.items():
        if not isinstance(entry, dict):
            continue
        if match_category(entry, cat):
            # NB: Field names ("Kilit", "Asistan_Notu") are Turkish to stay
            # wire-compatible with VIBE_executor.dyn, which reads/writes these
            # exact keys. Do not rename without updating the Dynamo script.
            existing = entry.get("Asistan_Notu", "")
            if mode == "append" and existing:
                entry["Asistan_Notu"] = f"{existing} | {value}"
            else:
                entry["Asistan_Notu"] = value
            entry["Kilit"]     = True
            entry["LastError"] = ""
            count += 1
            updated_ids.append(element_id)

    if count == 0:
        sample_types = sorted({
            (e.get("tip") or e.get("element_type") or "")
            for e in data.values()
            if isinstance(e, dict)
            and (e.get("tip") or e.get("element_type"))
        })[:10]
        return jsonify({
            "msg": (
                f"No elements found for category '{cat}'. "
                f"Sample type values in model: {sample_types}"
            ),
            "count": 0,
            "ids": [],
        })

    safe_write_json(data)
    return jsonify({
        "msg": f"Done — {count} '{cat}' element(s) marked as pending. Dynamo will write.",
        "count": count,
        "ids": updated_ids,
    })


@app.route("/api/log", methods=["POST"])
def api_log():
    """Optional UI log endpoint used when a voice conversation stops before /api/write."""
    body = request.json or {}
    _append_log_row([
        datetime.utcnow().isoformat(timespec="seconds"),
        "ui_log",
        body.get("command", ""),
        body.get("category", ""),
        body.get("source", ""),
        body.get("note", ""),
    ])
    return jsonify({"ok": True})


@app.route("/api/reset_pending", methods=["POST"])
def api_reset_pending():
    data = safe_read_json()
    count = sum(
        1 for entry in data.values()
        if isinstance(entry, dict) and entry.get("Kilit")
    )
    for entry in data.values():
        if isinstance(entry, dict):
            entry["Kilit"] = False
    safe_write_json(data)
    return jsonify({"msg": f"{count} pending flag(s) cleared."})


@app.route("/api/category_status", methods=["POST"])
def api_category_status():
    """
    Inspect the current state of all elements matching a category.

    Used by the end-to-end benchmark harness to verify that Dynamo has
    actually executed a pending write. The caller supplies an optional
    note_substr to narrow the check to elements tagged by a specific
    benchmark command (avoiding false positives from earlier writes).

    Request body (JSON)
    -------------------
    cat         : str  — category key (e.g. "wall")
    note_substr : str  — optional marker embedded in Asistan_Notu

    Returns
    -------
    total_in_category : int   — elements of this category present in JSON
    with_note         : int   — elements whose Asistan_Notu contains note_substr
    still_locked      : int   — of those, how many still have Kilit == True
    with_last_write   : int   — of those, how many have a LastWrite timestamp
    with_error        : int   — of those, how many have a non-empty LastError
    errors            : list  — first 10 error strings, for diagnostics
    """
    body = request.json or {}
    cat  = body.get("cat")
    note_substr = body.get("note_substr", "") or ""

    data = safe_read_json()
    result = {
        "total_in_category": 0,
        "with_note":         0,
        "still_locked":      0,
        "with_last_write":   0,
        "with_error":        0,
        "errors":            [],
    }
    for eid, entry in data.items():
        if not isinstance(entry, dict):
            continue
        if not match_category(entry, cat):
            continue
        result["total_in_category"] += 1
        asistan_notu = entry.get("Asistan_Notu", "") or ""
        if note_substr and note_substr in asistan_notu:
            result["with_note"] += 1
            if entry.get("Kilit") is True:
                result["still_locked"] += 1
            if entry.get("LastWrite"):
                result["with_last_write"] += 1
            err = entry.get("LastError", "") or ""
            if err:
                result["with_error"] += 1
                if len(result["errors"]) < 10:
                    result["errors"].append({"eid": eid, "err": err[:200]})
    return jsonify(result)


@app.route("/api/health")
def api_health():
    """Report status of both intent-resolution tiers."""
    return jsonify({
        "ollama_url":         OLLAMA_URL,
        "ollama_model":       OLLAMA_MODEL,
        "ollama_available":   _ollama_available(),
        "requests_installed": requests is not None,
        "json_path":          JSON_PATH,
        "json_exists":        os.path.exists(JSON_PATH),
        "lock_exists":        os.path.exists(LOCK_PATH),
        "log_path":           LOG_PATH,
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, port=5000)