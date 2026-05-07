[README.md](https://github.com/user-attachments/files/27486697/README.md)
# VIBE: Voice Interface for BIM Environments

VIBE is a rule-first natural-language interface for BIM parameter annotation. It connects a browser-based voice/text interface to Autodesk Revit through a Flask middleware layer, a local Ollama-hosted LLM fallback, and a Dynamo executor. The current prototype writes free-text annotations to a Revit `Comments`-mapped field (`Asistan_Notu`) through a shared JSON handoff file.

The main design goal is to avoid a cloud LLM as a hard dependency. Common commands are resolved by deterministic keyword rules; ambiguous commands are escalated to a locally served `llama3.1:8b` model through Ollama.

## Repository structure

```text
vibe-bim/
├── app.py                         # Flask backend and embedded browser UI
├── config.example.env             # Example local configuration
├── requirements.txt               # Python dependencies
├── LICENSE
├── README.md
├── benchmark/
│   ├── run_benchmark.py           # NLP-only benchmark over the 91-command corpus
│   ├── run_e2e_benchmark.py       # End-to-end benchmark: NLP + write + verification
│   ├── run_multi.py               # Repeated benchmark wrapper
│   ├── aggregate_runs.py          # Multi-run aggregation helper
│   └── analyze_metrics.py         # Single-run metric analysis helper
├── dynamo/
│   └── VIBE_executor.dyn          # Dynamo graph used inside Revit
├── data/
│   └── sample_revit_data.json     # Minimal example JSON schema
└── runs/
    └── take3_run*.csv/.txt        # Five replicated benchmark logs
```

## Requirements

Tested environment:

- Python 3.11
- Autodesk Revit 2025
- Dynamo Player inside Revit
- Ollama with `llama3.1:8b`
- Windows host for the Revit/Dynamo workflow

Python dependencies are listed in `requirements.txt`.

## Setup

Clone the repository and install Python dependencies:

```bash
git clone https://github.com/unes21/vibe-bim.git
cd vibe-bim
python -m venv .venv
.venv\Scripts\activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Install Ollama separately and pull the local model:

```bash
ollama pull llama3.1:8b
```

Ollama usually runs automatically on Windows after installation. If needed, start it manually:

```bash
ollama serve
```

## Configuration

Copy the example environment file and edit the JSON path:

```bash
copy config.example.env .env
```

Edit `.env`:

```env
VIBE_JSON_PATH=C:\ProjectX\revit_data.json
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b
OLLAMA_TIMEOUT_SEC=30
VIBE_LOG_PATH=./vibe_bench.csv
```

Do not commit your real `.env` file. It may contain personal local paths.

## Revit and Dynamo preparation

1. Open Autodesk Revit 2025.
2. Load the benchmark model, for example the standard `racbasicsampleproject.rvt` from the Revit sample project library.
3. Open Dynamo Player.
4. Load `dynamo/VIBE_executor.dyn`.
5. Make sure the Dynamo workflow is writing and polling the same JSON file configured by `VIBE_JSON_PATH`.
6. Keep Revit and Dynamo running while the Flask server and benchmarks execute.

The repository does not include Autodesk Revit sample model files (`.rvt`) because they are Autodesk-distributed assets.

## Running the Flask server

From the repository root:

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

Useful API endpoints:

```text
GET  /api/health
POST /api/llm/category
POST /api/llm/mode
GET  /api/data
POST /api/write
POST /api/reset_pending
POST /api/category_status
```

## Running benchmarks

The benchmark scripts live in `benchmark/`. Run them from that directory so side-by-side imports work cleanly:

```bash
cd benchmark
```

NLP-only benchmark:

```bash
python run_benchmark.py --server http://127.0.0.1:5000
```

End-to-end benchmark:

```bash
python run_e2e_benchmark.py --server http://127.0.0.1:5000 --wait-sec 20
```

Five-run repeated benchmark:

```bash
python run_multi.py --runs 5 --label take3 --out-dir ../runs --server http://127.0.0.1:5000
```

Aggregate completed run CSVs:

```bash
python aggregate_runs.py --csv-dir ../runs --out-csv ../runs/aggregated_per_category.csv
```

## Benchmark corpus

The 91-command corpus is embedded in `benchmark/run_benchmark.py` as `TEST_CORPUS`. It covers 13 BIM categories with seven syntactic variants per category:

- direct Turkish keyword
- direct English keyword
- Turkish synonym or alternative phrasing
- accent-stripped Turkish
- out-of-vocabulary Turkish description
- out-of-vocabulary English description
- indirect or inferential English description

## Notes on reproducibility

The included `runs/take3_run01` to `take3_run05` files are replicated benchmark logs. They document the observed NLP tier distribution, per-category results, latency summaries, and dispatch misses.

The local JSON handoff file (`revit_data.json`) is intentionally not included as a full runtime artifact because it contains live Revit element state and local benchmark traces. A small schema-only example is provided in `data/sample_revit_data.json`.

## Security and privacy

Do not commit:

- `.env`
- full `revit_data.json` runtime files
- Revit model files (`.rvt`, `.rfa`)
- Python cache files (`__pycache__`, `.pyc`)
- API keys or external service credentials

The current VIBE architecture does not require a cloud LLM API key for the reported workflow.

## License

This project is released under the MIT License. See `LICENSE`.
