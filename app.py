# Environment and secrets
.env
*.env.local

# Python
__pycache__/
*.py[cod]
*$py.class
.venv/
venv/
env/

# Local runtime files
revit_data.json
revit_data_backup_full.json
*.lock
*.tmp
vibe_bench.csv
vibe_bench_results.csv
vibe_bench_summary.txt
vibe_e2e_results.csv
vibe_e2e_summary.txt
aggregated_per_category.csv
flaky_commands.csv

# OS/editor
.DS_Store
Thumbs.db
.vscode/
.idea/

# Revit model files are not included by default.
*.rvt
*.rfa
