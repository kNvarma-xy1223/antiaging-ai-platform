"""
populate_graph.py  v6  — FINAL
================================
Key fix: Phase 5 now uses /bio/compound-targets/ (correct direction)
  RAPAMYCIN → molecule search → MTOR gene target  ✅
  (old Phase 5 used /bio/drug-targets/ which was backwards)

USAGE:
  Terminal 1: uvicorn api.main:app --reload
  Terminal 2: python populate_graph.py
"""

import requests, time, sys

BASE = "http://127.0.0.1:8000"

def graph_stats():
    try:
        r = requests.get(f"{BASE}/graph/stats", timeout=10)
        s = r.json()
        types = s.get("node_type_counts", {})
        print(f"\n  📊 {s['nodes']} nodes  {s['edges']} edges")
        for k, v in sorted(types.items(), key=lambda x: -x[1]):
            print(f"     {k:<22} {v:>5}  {'█'*min(v//50,30)}")
    except Exception:
        pass

def call(endpoint, label, timeout=90):
    print(f"\n  {'─'*50}")
    print(f"  {label}")
    print(f"  {endpoint}")
    try:
        r = requests.get(f"{BASE}{endpoint}", timeout=timeout)
        if r.status_code == 200:
            d = r.json()
            print(f"  ✅  {d.get('count', '?')} records")
            return True
        print(f"  ❌  HTTP {r.status_code}")
        return False
    except requests.exceptions.ConnectionError:
        print("  ❌  Server not reachable. Run: uvicorn api.main:app --reload")
        sys.exit(1)
    except requests.exceptions.Timeout:
        print(f"  ⏱️  Timed out — skipping")
        return False
    except Exception as e:
        print(f"  ❌  {e}")
        return False

# ── check ─────────────────────────────────────────────────────
print("\n" + "="*55)
print("  POPULATE GRAPH  v6  (FINAL)")
print("="*55)
try:
    requests.get(f"{BASE}/health", timeout=5)
    print("\n  ✅  Server is up")
except Exception:
    print("\n  ❌  Server not running. Run: uvicorn api.main:app --reload")
    sys.exit(1)

graph_stats()

# ═══════════════════════════════════════════════
# PHASE 1 — HAGR databases
# ═══════════════════════════════════════════════
print("\n\n📚  PHASE 1 — HAGR databases\n")
for ep, lbl in [
    ("/hagr/aging-genes",  "GenAge       → aging genes"),
    ("/hagr/cellage",      "CellAge      → senescence genes"),
    ("/hagr/gendr",        "GenDR        → DR genes"),
    ("/hagr/longevitymap", "LongevityMap → longevity variants"),
    ("/hagr/drugage",      "DrugAge      → longevity compounds (ML labels)"),
]:
    call(ep, lbl)
    graph_stats()
    time.sleep(2)

# ═══════════════════════════════════════════════
# PHASE 2 — Gene → drugs (fast, no timeout)
# ═══════════════════════════════════════════════
print("\n\n🧬  PHASE 2 — Gene → ChEMBL drugs [fast mode]\n")
for gene, desc in [
    ("TP53","DNA damage"), ("SIRT1","caloric restriction"),
    ("FOXO3","pro-longevity TF"), ("IGF1","insulin signaling"),
    ("MTOR","longevity regulator"), ("AMPK","energy sensor"),
    ("AKT1","PI3K-AKT"), ("PTEN","tumor suppressor"),
    ("TERT","telomerase"), ("CDKN2A","senescence"), ("BCL2","apoptosis"),
]:
    call(f"/bio/drug-targets/{gene}?fast=true", f"Gene → drugs: {gene} ({desc})")
    time.sleep(1)
graph_stats()

# ═══════════════════════════════════════════════
# PHASE 3 — PPI
# ═══════════════════════════════════════════════
print("\n\n🔗  PHASE 3 — Protein interactions\n")
for gene in ["MTOR","TP53","SIRT1","FOXO3","AKT1","CDKN2A"]:
    call(f"/bio/ppi/{gene}", f"PPI → {gene}", timeout=45)
    time.sleep(2)

# ═══════════════════════════════════════════════
# PHASE 4 — Pathways
# ═══════════════════════════════════════════════
print("\n\n🛤️   PHASE 4 — Pathways\n")
for gene in ["MTOR","TP53","SIRT1","FOXO3","AKT1"]:
    call(f"/bio/pathways/{gene}", f"Pathways → {gene}", timeout=45)
    time.sleep(2)
graph_stats()

# ═══════════════════════════════════════════════
# PHASE 5 — Compound → GENE targets (CORRECT DIRECTION)
# Uses /bio/compound-targets/ endpoint
# compound_name → ChEMBL molecule ID → mechanism → gene targets
# RAPAMYCIN → CHEMBL6456 → MTOR, FKBP1A  ✅
# ═══════════════════════════════════════════════
print("\n\n💊  PHASE 5 — Compound → gene targets (CORRECT DIRECTION)")
print("    endpoint: /bio/compound-targets/{name}")
print("    RAPAMYCIN → molecule search → MTOR gene  ✅\n")

compounds = [
    ("RAPAMYCIN",         "mTOR inhibitor — #1 longevity drug"),
    ("METFORMIN",         "AMPK activator"),
    ("RESVERATROL",       "SIRT1 activator"),
    ("QUERCETIN",         "senolytic"),
    ("CURCUMIN",          "anti-inflammatory"),
    ("BERBERINE",         "AMPK activator"),
    ("ASPIRIN",           "COX inhibitor"),
    ("FISETIN",           "senolytic"),
    ("SPERMIDINE",        "autophagy inducer"),
    ("TORIN 1",           "mTOR inhibitor"),
    ("ACARBOSE",          "NIA ITP drug"),
    ("MELATONIN",         "antioxidant"),
    ("LITHIUM CHLORIDE",  "GSK3 inhibitor"),
    ("NICOTINAMIDE",      "NAD precursor"),
    ("EPIGALLOCATECHIN GALLATE", "EGCG green tea"),
]

done = 0
for name, desc in compounds:
    # /bio/compound-targets/ = correct direction (compound→gene)
    ok = call(f"/bio/compound-targets/{name}", f"Compound → genes: {name} ({desc})")
    if ok: done += 1
    time.sleep(3)

print(f"\n  💊  {done}/{len(compounds)} compounds linked to gene targets")

# ═══════════════════════════════════════════════
# DONE
# ═══════════════════════════════════════════════
print("\n\n" + "="*55)
print("  GRAPH POPULATION COMPLETE")
print("="*55)
graph_stats()
print("""
VALIDATION — check RAPAMYCIN has MTOR as neighbor:
  GET /graph/node/RAPAMYCIN
  Expected: "neighbors" contains "MTOR" and/or "FKBP1A"

NEXT STEPS:
  python agents/ml_agent.py
""")