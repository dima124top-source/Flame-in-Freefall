# Flame in Freefall

Microgravity ($0g$) fire safety simulator and combustion advisor built on NASA Physical Sciences Informatics (PSI) datasets (`BASS-II`, `Saffire`, `ACME`, `SoFIE`, `FLEX-2`). Because buoyancy-driven convection vanishes in orbit, low-speed cabin ventilation feeds flames instead of extinguishing them, and molten polymers form boiling droplet clouds.

## What It Does
* **Cabin & Mixture Simulation:** Models single polymers (`PMMA`, `Cotton`, `PVC`, `Nomex`, `PTFE`) or normalized multi-component mixtures across $O_2$ ($15\%–30\%$), airflow ($0–20\text{ cm/s}$), and pressure ($55–105\text{ kPa}$). Accounts for the $0g$ wick effect ($1.28\times$ flame spread multiplier when meltable polymers meet fibrous/charring matrices) and calculates a $1–10$ Fire Hazard Index.
* **Decoupled RAG + Telemetry:** Parses NASA PSI PDF reports into ChromaDB on CPU with boolean material/facility tags (`has_pmma`, `has_saffire`, etc.) to prevent cross-contaminating chemical properties. Time-series CSV logs are plotted directly via Pandas and Plotly (falling back to empirical NASA correlations when local CSVs are absent).
* **Onboard Advisor:** Connects live simulator telemetry with retrieved NASA excerpts to answer questions on flame physics and suppression tactics, enforcing strict elemental guardrails based on active polymer formulas.

## Structure
```text
Flame in Freefall/
├── chroma_data/       # Persisted ChromaDB vector store (auto-generated)
├── data/
│   ├── csv/           # Optional sensor telemetry (cols: time_s, spread_rate_mms, flame_temp_k, co_ppm)
│   └── pdf/           # NASA PSI / NTRS technical reports (PDF)
├── docstore_data/     # Persisted parent document chunks (auto-generated)
├── scripts/
│   ├── app.py         # Streamlit dashboard, physics engine, and RAG UI
│   └── indexer.py     # PDF parser, regex metadata tagger, and CPU indexer
├── .gitignore
├── README.md
└── requirements.txt
```


# Quickstart
## 1. Environment Installation
Run all commands from the repository root (Python 3.11–3.12):


Linux / macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
Windows (PowerShell):

```PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Adding NASA PSI Data
PDF Reports: Download investigation reports from NASA PSI or NASA NTRS (search for BASS-II, Saffire, ACME, SoFIE, or FLEX-2) and place them in data/pdf/.

CSV Telemetry (Optional): Place sensor logs in data/csv/. Include the material name in the filename (e.g., pmma_bass2.csv) and ensure the file contains the columns:
time_s, spread_rate_mms, flame_temp_k, co_ppm.

## 3. Indexing Documents
Build the local ChromaDB and parent document store on CPU before launching the UI (you can also trigger this later from the app sidebar):
```
python scripts/indexer.py
```

## 4. Launch the dashboard (opens at http://localhost:8501):
```
streamlit run scripts/apps.py
```
