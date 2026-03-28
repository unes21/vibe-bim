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

File-lock protocol
------------------
Flask and Dynamo share a single JSON file (intent.json).  Concurrent access
is managed via an atomic lock file (intent.json.lock), using os.O_CREAT |
os.O_EXCL to ensure mutually exclusive writes on both Windows and Unix.

Configuration
-------------
Set JSON_PATH via the VIBE_JSON_PATH environment variable, or edit the
fallback path below.  Set GROQ_API_KEY to enable LLM-based intent
extraction; the system falls back gracefully to rule-based extraction
if the key is absent.

Usage
-----
    pip install flask groq
    set GROQ_API_KEY=<your_key>        # Windows
    export GROQ_API_KEY=<your_key>     # Unix/macOS
    python app.py

Author  : Ayberk Enis
Project : VIBE - Voice Interface for BIM Environments
License : MIT
"""

from flask import Flask, render_template_string, jsonify, request
import json
import os
import time
from datetime import datetime

# ---------------------------------------------------------------------------
# Optional Groq import — system works without it (rule-based fallback)
# ---------------------------------------------------------------------------
try:
    from groq import Groq
except ImportError:
    Groq = None

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Configuration
# JSON_PATH: absolute path to the shared intent file that Dynamo also reads.
# Prefer a local (non-cloud-synced) path to avoid file-locking conflicts.
# ---------------------------------------------------------------------------
JSON_PATH = os.getenv(
    "VIBE_JSON_PATH",
    r"C:\ProjectX\revit_data.json"   # fallback — override via env var
)
LOCK_PATH = JSON_PATH + ".lock"

GROQ_KEY    = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_KEY) if (Groq and GROQ_KEY) else None

# ---------------------------------------------------------------------------
# Category taxonomy
# Maps internal category keys to Revit element type strings in both
# English and Turkish (for bilingual command resolution).
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
# Uses atomic O_CREAT | O_EXCL file creation for cross-platform locking.
# Compatible with Windows (where Unix fcntl is unavailable).
# ---------------------------------------------------------------------------

def acquire_lock(timeout_sec: float = 2.0, poll: float = 0.05) -> bool:
    """
    Attempt to acquire the shared file lock within *timeout_sec* seconds.

    Returns True on success, False on timeout.
    """
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
    """Remove the lock file, suppressing errors if already absent."""
    try:
        if os.path.exists(LOCK_PATH):
            os.remove(LOCK_PATH)
    except OSError:
        pass


def safe_read_json(max_retry: int = 12, poll: float = 0.05) -> dict:
    """
    Read and parse the intent JSON file under the file lock.

    Retries up to *max_retry* times if the lock is held by Dynamo.
    Returns an empty dict if the file is absent or unparseable.
    """
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
    """
    Write *data* to the intent JSON file atomically.

    Uses a temporary file + os.replace() to prevent partial reads
    by the Dynamo polling script.

    Raises RuntimeError if the lock cannot be acquired.
    """
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

def match_category(element_type: str, category_key: str) -> bool:
    """
    Return True if *element_type* (Revit element category string) matches
    any alias for *category_key* in CATEGORY_MAP.
    """
    normalized = (element_type or "").strip().lower()
    for alias in CATEGORY_MAP.get(category_key, []):
        alias = alias.lower()
        if normalized == alias or normalized.startswith(alias):
            return True
    return False


# ---------------------------------------------------------------------------
# Rule-based intent extraction (no API key required)
# ---------------------------------------------------------------------------

def rule_extract_category(text: str):
    """
    Resolve a BIM element category from *text* using keyword matching.

    Supports both English and Turkish keywords.
    Returns a category key string or None.
    """
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
    """
    Resolve the write mode (append / overwrite) from *text*.

    Returns "append", "overwrite", or None.
    """
    t = (text or "").lower()
    append_keywords   = ["yanına", "yanina", "ekle", "üzerine ekle", "ustune ekle", "append"]
    overwrite_keywords = ["sil", "sıfır", "sifir", "baştan", "bastan", "yeniden", "overwrite"]
    if any(kw in t for kw in append_keywords):
        return "append"
    if any(kw in t for kw in overwrite_keywords):
        return "overwrite"
    return None


# ---------------------------------------------------------------------------
# LLM-based intent extraction (Groq / Llama 3 — optional)
# Falls back to rule-based if GROQ_API_KEY is not set.
# ---------------------------------------------------------------------------

def llm_extract_category(text: str) -> dict:
    """
    Extract BIM category from *text*, preferring rule-based resolution.

    Escalates to the Groq-hosted Llama 3 (70B) model only when rules
    fail to produce a confident match.

    Returns a dict with keys: category (str|None), source (str).
    """
    # 1. Rule-based attempt
    cat = rule_extract_category(text)
    if cat:
        return {"category": cat, "source": "rules"}

    # 2. LLM escalation
    if not groq_client:
        return {"category": None, "source": "no_groq_key"}

    prompt = (
        "Return JSON only. No explanation.\n"
        f"Allowed categories: {CATEGORY_KEYS}\n"
        "Extract the most appropriate BIM category from the user sentence.\n"
        'Example output: {"category":"roof"} or {"category":null}\n\n'
        f'User: "{text}"'
    )
    try:
        response = groq_client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        payload = json.loads(response.choices[0].message.content.strip())
        cat = payload.get("category")
        if cat in CATEGORY_KEYS:
            return {"category": cat, "source": "llm"}
        return {"category": None, "source": "llm_null"}
    except Exception as exc:
        return {"category": None, "source": "llm_error", "error": str(exc)[:200]}


def llm_extract_mode(text: str) -> dict:
    """
    Extract write mode ("append" or "overwrite") from *text*.

    Returns a dict with keys: mode (str|None), source (str).
    """
    # 1. Rule-based attempt
    mode = rule_extract_mode(text)
    if mode:
        return {"mode": mode, "source": "rules"}

    # 2. LLM escalation
    if not groq_client:
        return {"mode": None, "source": "no_groq_key"}

    prompt = (
        "Return JSON only. No explanation.\n"
        "append = add after existing value\n"
        "overwrite = replace existing value\n"
        'Example: {"mode":"append"} or {"mode":null}\n\n'
        f'User: "{text}"'
    )
    try:
        response = groq_client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        payload = json.loads(response.choices[0].message.content.strip())
        mode = payload.get("mode")
        if mode in ("append", "overwrite"):
            return {"mode": mode, "source": "llm"}
        return {"mode": None, "source": "llm_null"}
    except Exception as exc:
        return {"mode": None, "source": "llm_error", "error": str(exc)[:200]}


# ---------------------------------------------------------------------------
# Frontend (single-page voice interface)
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Serve the VIBE voice interface."""
    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>VIBE — Voice Interface for BIM</title>
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
  <h2 style="margin:0 0 4px 0;color:#f63366">VIBE &mdash; Voice Interface for BIM</h2>
  <div class="hint">Workflow: speak a command (e.g. "add a note to the roof") → specify the note → choose append or overwrite → sent to Dynamo.</div>

  <div class="card">
    <div class="row">
      <span class="pill" id="statePill">State: idle</span>
      <span class="pill" id="catPill">Category: —</span>
      <span class="pill" id="modePill">Mode: —</span>
      <span class="pill" id="ttsPill">TTS: —</span>
      <span class="pill" id="lockPill">🔒 0 pending</span>
      <span class="pill" id="llmPill">LLM: —</span>
    </div>

    <div style="height:10px"></div>
    <div id="chat"></div>

    <div style="height:12px"></div>
    <div class="row">
      <button class="btn" id="talkBtn" onclick="startConversation()">🎤 Speak</button>
      <button class="btn2" onclick="stopAll()">⏹ Stop</button>
      <button class="btn2" onclick="location.reload()">🔄 Refresh</button>
    </div>

    <div class="hint" style="margin-top:10px">
      Chrome recommended. Grant microphone permission.
      TTS errors are non-fatal — the pipeline still runs.
    </div>
    <div class="hint danger" id="err"></div>
  </div>
</div>

<script>
const chat     = document.getElementById("chat");
const err      = document.getElementById("err");
const statePill = document.getElementById("statePill");
const catPill  = document.getElementById("catPill");
const modePill = document.getElementById("modePill");
const ttsPill  = document.getElementById("ttsPill");
const lockPill = document.getElementById("lockPill");
const llmPill  = document.getElementById("llmPill");
const talkBtn  = document.getElementById("talkBtn");

function addMsg(text, who){
  const d = document.createElement("div");
  d.className = "msg " + who;
  d.textContent = text;
  chat.appendChild(d);
  chat.scrollTop = chat.scrollHeight;
}
function setState(s){ statePill.textContent = "State: " + s; }
function setCat(c){ catPill.textContent = "Category: " + (c || "—"); }
function setMode(m){ modePill.textContent = "Mode: " + (m || "—"); }
function setLLM(s){ llmPill.textContent = "LLM: " + (s || "—"); }

let recognition = null;

function stopAll(){
  try{ if(recognition) recognition.abort(); }catch(e){}
  try{ window.speechSynthesis.cancel(); }catch(e){}
  setState("idle");
  talkBtn.disabled = false;
}

function speak(text){
  return new Promise((resolve) => {
    try{
      window.speechSynthesis.cancel();
      const u = new SpeechSynthesisUtterance(text);
      // Try language matching the command; fall back to en-US
      u.lang = "en-US";
      u.rate = 1.0;
      u.onend = () => { ttsPill.textContent="TTS: idle"; resolve(); };
      u.onerror = () => { ttsPill.textContent="TTS: error"; resolve(); };
      ttsPill.textContent = "TTS: speaking";
      window.speechSynthesis.speak(u);
    }catch(e){ ttsPill.textContent="TTS: error"; resolve(); }
  });
}

function listenOnce(lang="en-US"){
  return new Promise((resolve, reject) => {
    err.textContent = "";
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if(!SR){ reject(new Error("SpeechRecognition not available — use Chrome.")); return; }
    recognition = new SR();
    recognition.lang = lang;
    recognition.interimResults = false;
    recognition.continuous = false;
    recognition.onresult = (e) => resolve(e.results[0][0].transcript.trim());
    recognition.onerror = (e) => reject(new Error("Microphone error: " + e.error));
    recognition.start();
  });
}

async function refreshPending(){
  try{
    const data = await fetch("/api/data").then(r=>r.json());
    let count = 0;
    for(const k of Object.keys(data||{})){
      if(data[k]?.pending === true) count++;
    }
    lockPill.textContent = "🔒 " + count + " pending";
  }catch(e){}
}
setInterval(refreshPending, 1500);
refreshPending();

async function post(url, body){
  const r = await fetch(url,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
  return r.json();
}

async function startConversation(){
  talkBtn.disabled = true;
  try{
    setState("listening — command");
    addMsg("Listening… (e.g. 'add a note to the roof')", "ai");
    await speak("Listening. Say a command, for example: add a note to the roof.");

    const cmd = await listenOnce();
    addMsg(cmd, "user");

    setState("resolving category");
    const catObj = await post("/api/llm/category", {text: cmd});
    setLLM(catObj.source || "—");
    const cat = catObj.category || null;

    if(!cat){
      setState("idle");
      addMsg("Could not identify a BIM category. Try: roof, wall, floor, door…", "ai");
      await speak("Could not identify the category. Please try again.");
      talkBtn.disabled = false;
      return;
    }
    setCat(cat);

    setState("listening — note");
    addMsg("What is the note?", "ai");
    await speak("What is the note?");
    const note = await listenOnce();
    addMsg(note, "user");

    setState("listening — mode");
    addMsg("Append to existing, or overwrite?", "ai");
    await speak("Should I append this to the existing value, or overwrite it?");
    const modeAns = await listenOnce();
    addMsg(modeAns, "user");

    setState("resolving mode");
    const modeObj = await post("/api/llm/mode", {text: modeAns});
    setLLM(modeObj.source || "—");
    let mode = modeObj.mode || "append";
    if(!modeObj.mode){
      addMsg("Mode unclear — defaulting to append.", "ai");
      await speak("Mode was unclear. Defaulting to append.");
    }
    setMode(mode);

    setState("sending to Dynamo");
    addMsg("Sending to Dynamo…", "ai");
    await speak("Sending to Dynamo.");

    const out = await post("/api/write", {cat, note, mode});
    addMsg(out.msg || "Done.", "ai");
    await speak(out.msg || "Done.");

    setState("done");
    refreshPending();
    talkBtn.disabled = false;

  }catch(e){
    err.textContent = String(e.message || e);
    addMsg("Error: " + (e.message || e), "ai");
    try{ await speak("An error occurred. Check microphone permissions."); }catch(_){}
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
    return jsonify(llm_extract_category(text))


@app.route("/api/llm/mode", methods=["POST"])
def api_llm_mode():
    """Resolve write mode (append / overwrite) from a natural-language string."""
    text = (request.json or {}).get("text", "").strip()
    return jsonify(llm_extract_mode(text))


@app.route("/api/data")
def api_data():
    """Return the current contents of the intent JSON file."""
    return jsonify(safe_read_json())


@app.route("/api/write", methods=["POST"])
def api_write():
    """
    Mark all elements of the given category as pending in the intent file.

    Request body (JSON)
    -------------------
    cat  : str  — category key (e.g. "wall")
    note : str  — the assistant note to write
    mode : str  — "append" or "overwrite"

    Returns the number of elements marked and their ElementIds.
    """
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
        element_type = entry.get("element_type", "")
        if match_category(element_type, cat):
            existing = entry.get("assistant_value", "")
            if mode == "append" and existing:
                entry["assistant_value"] = f"{existing} | {value}"
            else:
                entry["assistant_value"] = value
            entry["pending"]   = True
            entry["LastError"] = ""
            count += 1
            updated_ids.append(element_id)

    if count == 0:
        sample_types = sorted({
            e.get("element_type", "") for e in data.values()
            if isinstance(e, dict) and e.get("element_type")
        })[:10]
        return jsonify({
            "msg": (
                f"No elements found for category '{cat}'. "
                f"Sample element_type values in model: {sample_types}"
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


@app.route("/api/reset_pending", methods=["POST"])
def api_reset_pending():
    """Clear all pending flags in the intent file (emergency reset)."""
    data = safe_read_json()
    count = sum(
        1 for entry in data.values()
        if isinstance(entry, dict) and entry.get("pending")
    )
    for entry in data.values():
        if isinstance(entry, dict):
            entry["pending"] = False
    safe_write_json(data)
    return jsonify({"msg": f"{count} pending flag(s) cleared."})


@app.route("/api/health")
def api_health():
    """Return system health status."""
    return jsonify({
        "groq_enabled":    bool(groq_client),
        "groq_key_present": bool(GROQ_KEY),
        "json_path":       JSON_PATH,
        "json_exists":     os.path.exists(JSON_PATH),
        "lock_exists":     os.path.exists(LOCK_PATH),
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, port=5000)
