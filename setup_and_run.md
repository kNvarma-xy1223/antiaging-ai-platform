# Quick Start — 5 commands to run the platform

Run these in order. Each step must finish before the next.

---

## Terminal 1 — start the server (keep this running)

```powershell
# 1. activate venv
.\venv\Scripts\Activate.ps1

# 2. start server
uvicorn api.main:app --reload
```

Leave this terminal open. Server runs at http://127.0.0.1:8000

---

## Terminal 2 — setup (run once)

```powershell
# 1. activate venv
.\venv\Scripts\Activate.ps1

# 2. install dependencies (first time only)
pip install -r requirements.txt

# 3. populate knowledge graph (10-20 min, internet needed)
python populate_graph.py

# 4. train ML models (2-5 min)
python api/agents/ml_agent.py
```

After step 4, go to Terminal 1, press Ctrl+C, restart:
```powershell
uvicorn api.main:app --reload
```

---

## Validate everything works

Open browser: http://127.0.0.1:8000/docs

Run these in Swagger:
- `GET /health`                    → should return {"status": "healthy"}
- `GET /graph/stats`               → nodes > 3000
- `GET /ml/status`                 → all 3 models trained: true
- `GET /ml/report/RAPAMYCIN`       → Tier 1, success_probability > 0.90
- `GET /ml/report/QUERCETIN`       → Tier 1, known_targets has 11 genes

If all 5 pass — the platform is fully operational.

---

## Common issues

| Problem | Fix |
|---|---|
| `uvicorn: command not found` | Run `pip install uvicorn` |
| `Graph is empty` error | Make sure uvicorn is running, then run `python populate_graph.py` |
| `Embeddings not trained` in Swagger | Run `python agents/ml_agent.py` then restart uvicorn |
| Long path error on Windows pip | See README.md "Windows long-path issue" section |
| Timeout during populate | Normal — script skips and continues. Run again if needed |
