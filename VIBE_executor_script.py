"""
VIBE_executor_script.py
========================
Python script node content for the VIBE Dynamo workspace (VIBE_executor.dyn).

HOW TO USE
----------
This file exists for code readability and version control on GitHub.
To use it inside Revit:
  1. Open VIBE_executor.dyn in Dynamo (Player or Editor).
  2. The Python Script node already contains this code — no manual copy needed.
  3. Connect IN[0] to a File Path node pointing to your intent.json file.
  4. Connect IN[1] to a DateTime.Now node (triggers periodic polling).
  5. Set Dynamo Run Mode to "Periodic" (1000 ms recommended).

WHAT IT DOES
------------
Polls intent.json every ~1000 ms. For each element entry where
"pending" == True, the script:
  1. Resolves the Revit element by ElementId.
  2. Locates the target parameter (explicit name or default: Comments).
  3. Writes the "assistant_value" string to that parameter.
  4. Marks the entry "pending": False and records a timestamp.

FILE-LOCK PROTOCOL
------------------
Flask and Dynamo share intent.json. Concurrent access is managed through
a companion lock file (intent.json.lock). Since IronPython runs on Windows
where Unix fcntl is unavailable, locking is implemented as an atomic
file-existence check with a configurable timeout and poll interval.

INPUTS (Dynamo IN ports)
------------------------
IN[0] : str  — Absolute path to intent.json (match app.py JSON_PATH)
IN[1] : any  — Periodic trigger (connect DateTime.Now)

OUTPUT (Dynamo OUT port)
------------------------
OUT   : tuple(str, list[str])
        [0] status summary string
        [1] list of error strings (empty on clean run)

AUTHOR  : Ayberk Enis
PROJECT : VIBE — Voice Interface for BIM Environments
GITHUB  : https://github.com/unes21/vibe-bim
LICENSE : MIT
"""

import clr
import json
import os
import time

clr.AddReference("RevitServices")
from RevitServices.Persistence import DocumentManager
from RevitServices.Transactions import TransactionManager

clr.AddReference("RevitAPI")
from Autodesk.Revit.DB import ElementId, BuiltInParameter

# ---------------------------------------------------------------------------
# Revit document handle
# ---------------------------------------------------------------------------
doc = DocumentManager.Instance.CurrentDBDocument

# ---------------------------------------------------------------------------
# Dynamo inputs
# ---------------------------------------------------------------------------
json_path = IN[0]   # absolute path to the shared intent JSON file
_trigger  = IN[1]   # periodic trigger value — used to force re-evaluation

lock_path = json_path + ".lock"

# ---------------------------------------------------------------------------
# File-lock helpers
# Atomic file-existence lock; compatible with IronPython on Windows.
# ---------------------------------------------------------------------------

def acquire_lock(timeout_sec=2.0, poll_interval=0.05):
    """
    Attempt to acquire the lock file within *timeout_sec* seconds.

    Returns True on success, False on timeout.
    """
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if not os.path.exists(lock_path):
            try:
                with open(lock_path, "w") as f:
                    f.write("locked")
                return True
            except Exception:
                pass
        time.sleep(poll_interval)
    return False


def release_lock():
    """Remove the lock file, suppressing any errors."""
    try:
        if os.path.exists(lock_path):
            os.remove(lock_path)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Safe JSON read / write
# ---------------------------------------------------------------------------

def safe_read_json(max_retries=10, poll_interval=0.05):
    """
    Read and parse the intent JSON file under the file lock.

    Returns an empty dict if the file is absent or cannot be parsed.
    """
    if not os.path.exists(json_path):
        return {}
    for _ in range(max_retries):
        if not acquire_lock(timeout_sec=1.5, poll_interval=poll_interval):
            time.sleep(poll_interval)
            continue
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
            if not content:
                return {}
            return json.loads(content)
        except Exception:
            time.sleep(poll_interval)
        finally:
            release_lock()
    return {}


def safe_write_json(data):
    """
    Write *data* to the intent JSON file atomically via a .tmp rename.

    Raises RuntimeError if the lock cannot be acquired.
    """
    if not acquire_lock(timeout_sec=2.0, poll_interval=0.05):
        raise RuntimeError(
            "Cannot acquire file lock for writing "
            "(Flask server may be holding it — will retry on next poll)."
        )
    try:
        tmp_path = json_path + ".tmp"
        payload  = json.dumps(data, indent=4, ensure_ascii=False)
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(payload)
        if os.path.exists(json_path):
            try:
                os.remove(json_path)
            except Exception:
                pass
        os.rename(tmp_path, json_path)
    finally:
        release_lock()


# ---------------------------------------------------------------------------
# Parameter resolution
# ---------------------------------------------------------------------------

def resolve_parameter(element, param_name=None):
    """
    Locate a writable parameter on *element*.

    Resolution order:
      1. Explicit *param_name* from the intent payload (if supplied).
      2. BuiltInParameter.ALL_MODEL_INSTANCE_COMMENTS (default target).
      3. Fallback display names: "Comments", "Description".

    Returns an Autodesk.Revit.DB.Parameter, or None if none found.
    """
    if param_name:
        p = element.LookupParameter(param_name)
        if p and not p.IsReadOnly:
            return p

    p = element.get_Parameter(BuiltInParameter.ALL_MODEL_INSTANCE_COMMENTS)
    if p and not p.IsReadOnly:
        return p

    for name in ("Comments", "Description"):
        p = element.LookupParameter(name)
        if p and not p.IsReadOnly:
            return p

    return None


# ---------------------------------------------------------------------------
# Main polling loop
# ---------------------------------------------------------------------------

intent_data = safe_read_json()

# Collect element IDs with pending == True
pending_ids = [
    eid for eid, entry in (intent_data or {}).items()
    if isinstance(entry, dict) and entry.get("pending") is True
]

updated = 0
skipped = 0
errors  = []

if pending_ids:
    TransactionManager.Instance.EnsureInTransaction(doc)

    for element_id_str in pending_ids:
        entry      = intent_data[element_id_str]
        value      = (entry.get("assistant_value") or "").strip()
        param_name = entry.get("parameter_name")  # optional explicit target

        # Nothing to write
        if not value:
            entry["pending"]   = False
            entry["LastError"] = "No assistant_value in intent payload."
            skipped += 1
            continue

        try:
            element = doc.GetElement(ElementId(int(element_id_str)))

            if element is None:
                entry["pending"]   = False
                entry["LastError"] = "Element not found in active document."
                skipped += 1
                continue

            param = resolve_parameter(element, param_name)

            if param is None:
                entry["pending"]   = False
                entry["LastError"] = (
                    "No writable parameter found"
                    + (" for '{}'.".format(param_name) if param_name else ".")
                )
                skipped += 1
                continue

            param.Set(value)
            entry["pending"]    = False
            entry["LastError"]  = ""
            entry["last_write"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            updated += 1

        except Exception as exc:
            entry["LastError"] = str(exc)
            errors.append("ElementId {}: {}".format(element_id_str, exc))

    TransactionManager.Instance.TransactionTaskDone()

# Write results back
try:
    safe_write_json(intent_data)
except Exception as exc:
    errors.append("JSON write error: {}".format(exc))

# ---------------------------------------------------------------------------
# Dynamo output
# ---------------------------------------------------------------------------
status = (
    "VIBE executor — "
    "{} pending | {} updated | {} skipped | {} error(s)".format(
        len(pending_ids), updated, skipped, len(errors)
    )
)

OUT = status, errors
