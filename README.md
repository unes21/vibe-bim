# VIBE — Voice Interface for BIM Environments

> AI-driven voice interaction and execution framework for Autodesk Revit / Dynamo

**Paper:** VIBE: An AI-Driven Framework for Natural Language Interaction and Execution in Building Information Modeling Systems *(under review)*

---

## Repository Structure

```
vibe-bim/
├── app.py                    # Flask API server (NLP + file-lock layer)
├── VIBE_executor.dyn         # Dynamo workspace (open in Revit/Dynamo)
├── VIBE_executor_script.py   # Python script node content (readable version)
└── README.md
```

---

## How It Works

```
Browser (voice)
    │
    ▼
app.py  ──── rule-based NLP ──── resolves intent
    │              │
    │         LLM escalation     (Groq / Llama 3 70B, optional)
    │
    ▼
intent.json  (shared file, file-lock protected)
    │
    ▼
VIBE_executor.dyn  (Dynamo, runs inside Revit, polls every 1000 ms)
    │
    ▼
Revit model  (parameter written directly via Revit API)
```

---

## Quick Start

### 1. Flask server

```bash
pip install flask groq
export GROQ_API_KEY=<your_key>       # optional — enables LLM escalation
export VIBE_JSON_PATH=C:\path\to\intent.json
python app.py
```

Open `http://localhost:5000` in **Chrome** (required for Web Speech API).

### 2. Dynamo script

1. Open Autodesk Revit with the sample model (`rac_advanced_sample_project.rvt`).
2. Launch Dynamo and open `VIBE_executor.dyn`.
3. Set the **File Path** node to the same path as `VIBE_JSON_PATH`.
4. Set **Run Mode → Periodic** (1000 ms).
5. Click **Run**.

### 3. Speak a command

Example flow:
- **You say:** "Add a note to the roof"
- **VIBE asks:** "What is the note?"
- **You say:** "Needs waterproofing inspection"
- **VIBE asks:** "Append or overwrite?"
- **You say:** "Append"
- **Result:** All Roof elements in the model have the note appended to their Comments parameter within ~1–2 seconds.

---

## Configuration

| Environment Variable | Default | Description |
|---|---|---|
| `VIBE_JSON_PATH` | `C:\ProjectX\revit_data.json` | Absolute path to shared intent file |
| `GROQ_API_KEY` | *(not set)* | Groq API key for LLM escalation |

---

## Intent File Format (`intent.json`)

The Dynamo script reads entries where `"pending": true` and writes the `assistant_value` to the element's parameter.

```json
{
  "123456": {
    "element_type": "Roofs",
    "pending": true,
    "assistant_value": "Needs waterproofing inspection (14:32)",
    "parameter_name": null,
    "LastError": "",
    "last_write": ""
  }
}
```

`parameter_name` is optional. If null, the script defaults to the `Comments` (ALL_MODEL_INSTANCE_COMMENTS) built-in parameter.

---

## Supported Element Categories

| Key | Revit Type Strings |
|---|---|
| roof | Roofs, roof, çatı |
| wall | Walls, wall, Basic Wall, duvar |
| floor | Floors, floor, zemin |
| door | Doors, door, kapı |
| window | Windows, window, pencere |
| ceiling | Ceilings, ceiling, tavan |
| stair | Stairs, stair, merdiven |
| column | Structural Columns, column, kolon |
| beam | Structural Framing, beam, kiriş |
| room | Rooms, room, oda |
| furniture | Furniture, mobilya |
| light | Lighting Fixtures, light, aydınlatma |

---

## Test Model

Validation was performed on Revit's official sample model:
`rac_advanced_sample_project.rvt` (included with Autodesk Revit installation).

---

## License

MIT License — see `LICENSE` file.

---

## Citation

If you use VIBE in your research, please cite:

```
Enis, A. (2025). VIBE: Voice Interface for BIM Environments —
An AI-Driven Framework for Natural Language Interaction and
Execution in Building Information Modeling Systems. Under review.
```
