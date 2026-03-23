"""
ml_agent.py — Anti-Aging AI Platform ML Engine  v3.0  FINAL
============================================================
Path fix: uses graph_storage/ location to anchor all paths.
This works identically whether run via uvicorn or python directly.
"""

import os, sys, json, math, pickle, logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

# ── path setup ────────────────────────────────────────────────────────────────
_THIS_FILE = os.path.abspath(__file__)
_AGENTS_DIR = os.path.dirname(_THIS_FILE)           # .../agents/
_API_DIR    = os.path.dirname(_AGENTS_DIR)           # .../api/  OR  .../agents/ if flat
_ROOT_DIR   = os.path.dirname(_API_DIR)              # project root

# Walk up until we find graph_storage/ — that IS the project root
def _find_root():
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(5):
        if os.path.isdir(os.path.join(d, "graph_storage")) or \
           os.path.isdir(os.path.join(d, "data_cache")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    # fallback to cwd
    return os.getcwd()

_ROOT = _find_root()

for _p in [_AGENTS_DIR, _API_DIR, os.path.join(_ROOT, "api"),
           os.path.join(_ROOT, "api", "agents")]:
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)


class MLAgent:

    MODEL_DIR  = os.path.join(_ROOT, "ml_models")
    GRAPH_PATH = os.path.join(_ROOT, "graph_storage", "graph.pkl")
    CACHE_DIR  = os.path.join(_ROOT, "data_cache")
    EMBED_DIM  = 64

    def __init__(self):
        os.makedirs(self.MODEL_DIR, exist_ok=True)
        self.G              = None
        self.node_list      = []
        self.embeddings     = {}
        self.link_model     = None
        self.success_model  = None
        self.feature_names  = []
        self._compound_lookup: Dict[str, str] = {}
        self._load_graph()
        self._build_compound_lookup()
        self._load_saved_models()
        logger.info(
            f"MLAgent ready — graph: {self.G.number_of_nodes()} nodes | "
            f"root: {_ROOT}"
        )

    # ── graph ─────────────────────────────────────────────────────────────────
    def _load_graph(self):
        import networkx as nx
        if not os.path.exists(self.GRAPH_PATH):
            logger.warning(f"Graph not found at {self.GRAPH_PATH}. Run populate_graph.py first.")
            self.G = nx.Graph()
            return
        try:
            with open(self.GRAPH_PATH, "rb") as f:
                self.G = pickle.load(f)
            self.node_list = list(self.G.nodes())
            logger.info(f"Graph loaded: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges")
        except Exception as e:
            logger.error(f"Graph load error: {e}")
            import networkx as nx
            self.G = nx.Graph()

    def _reload_graph(self):
        self._load_graph()
        self._build_compound_lookup()

    def _build_compound_lookup(self):
        self._compound_lookup = {}
        for node, attr in self.G.nodes(data=True):
            if attr.get("type") in ("drug", "longevity_compound"):
                self._compound_lookup[node.upper()] = node
        logger.info(f"Compound lookup: {len(self._compound_lookup)} nodes")

    # ── persistence ───────────────────────────────────────────────────────────
    def _save(self, obj, filename):
        path = os.path.join(self.MODEL_DIR, filename)
        with open(path, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Saved → {path}")

    def _load(self, filename):
        path = os.path.join(self.MODEL_DIR, filename)
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
        return None

    def _load_saved_models(self):
        emb = self._load("embeddings.pkl")
        if emb: self.embeddings = emb; logger.info(f"Embeddings loaded: {len(emb)} nodes")
        lnk = self._load("link_model.pkl")
        if lnk: self.link_model = lnk; logger.info("Link predictor loaded")
        suc = self._load("success_model.pkl")
        if suc: self.success_model = suc; logger.info("Success predictor loaded")
        ftr = self._load("feature_names.pkl")
        if ftr: self.feature_names = ftr

    # ═══════════════════════════════════════════════════════════════
    # PHASE 1 — SPECTRAL EMBEDDINGS
    # ═══════════════════════════════════════════════════════════════
    def train_embeddings(self) -> Dict[str, List[float]]:
        self._reload_graph()
        n = self.G.number_of_nodes()
        if n < 10:
            raise ValueError(f"Graph has {n} nodes. Run populate_graph.py first.")

        self.node_list = list(self.G.nodes())
        node_idx = {node: i for i, node in enumerate(self.node_list)}
        logger.info(f"Building {n}×{n} adjacency matrix...")

        from scipy.sparse import lil_matrix, csr_matrix, diags
        from scipy.sparse.linalg import svds

        A = lil_matrix((n, n), dtype=np.float32)
        for u, v, data in self.G.edges(data=True):
            i, j = node_idx[u], node_idx[v]
            w = max(0.01, min(float(data.get("weight", 1.0)), 10.0))
            A[i, j] = w; A[j, i] = w

        A = csr_matrix(A)
        deg = np.array(A.sum(axis=1)).flatten(); deg[deg == 0] = 1.0
        D = diags(1.0 / np.sqrt(deg))
        A_norm = D @ A @ D

        k = min(self.EMBED_DIM, n - 2)
        logger.info(f"SVD k={k} on {n}×{n}...")
        U, S, _ = svds(A_norm, k=k)

        emb = U * np.sqrt(np.abs(S))
        norms = np.linalg.norm(emb, axis=1, keepdims=True); norms[norms == 0] = 1.0
        emb /= norms
        if k < self.EMBED_DIM:
            emb = np.hstack([emb, np.zeros((n, self.EMBED_DIM - k))])

        self.embeddings = {self.node_list[i]: emb[i].tolist() for i in range(n)}
        self._save(self.embeddings, "embeddings.pkl")
        logger.info(f"  Embeddings: {len(self.embeddings)} × {self.EMBED_DIM}")
        return self.embeddings

    def _vec(self, node: str):
        v = self.embeddings.get(node.upper())
        return np.array(v, dtype=np.float32) if v is not None else None

    def get_similar_nodes(self, node: str, top_k: int = 10) -> List[Dict]:
        if not self.embeddings:
            return [{"error": "Embeddings not trained. Call POST /ml/train/embeddings"}]
        node = node.upper()
        q = self._vec(node)
        if q is None: return [{"error": f"'{node}' not in embeddings"}]
        results = []
        for other, vec in self.embeddings.items():
            if other == node: continue
            v = np.array(vec, dtype=np.float32)
            denom = np.linalg.norm(q) * np.linalg.norm(v)
            sim = float(np.dot(q, v) / denom) if denom > 0 else 0.0
            results.append({"node": other, "type": self.G.nodes.get(other, {}).get("type", "unknown"), "similarity": round(sim, 4)})
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]

    # ═══════════════════════════════════════════════════════════════
    # PHASE 2 — LINK PREDICTION
    # ═══════════════════════════════════════════════════════════════
    def _hadamard(self, a, b): return a * b

    def _build_link_dataset(self):
        if not self.embeddings: raise ValueError("Train embeddings first.")
        drug_nodes = [n for n,a in self.G.nodes(data=True) if a.get("type") in ("drug","longevity_compound") and n in self.embeddings]
        gene_nodes = [n for n,a in self.G.nodes(data=True) if a.get("type") == "gene" and n in self.embeddings]
        if len(drug_nodes) < 5 or len(gene_nodes) < 5:
            raise ValueError(f"Not enough nodes: drugs={len(drug_nodes)}, genes={len(gene_nodes)}")

        positives = []
        for d in drug_nodes:
            for nb in self.G.neighbors(d):
                if nb in gene_nodes: positives.append((d, nb, 1))
            stored = self.G.nodes[d].get("targets", set())
            if isinstance(stored, set):
                for g in stored:
                    if g in gene_nodes and (d, g, 1) not in positives: positives.append((d, g, 1))

        if len(positives) < 10:
            raise ValueError(f"Only {len(positives)} drug-gene edges. Need at least 10.")

        existing = {(d,g) for d,g,_ in positives}
        import random; random.seed(42)
        negatives = []
        target_neg = min(len(positives)*2, len(drug_nodes)*len(gene_nodes)//2)
        attempts = 0
        while len(negatives) < target_neg and attempts < 1000000:
            d = random.choice(drug_nodes); g = random.choice(gene_nodes)
            if (d,g) not in existing: negatives.append((d,g,0)); existing.add((d,g))
            attempts += 1

        neg_sample = random.sample(negatives, min(len(positives), len(negatives)))
        all_pairs = positives + neg_sample; random.shuffle(all_pairs)
        X = np.array([self._hadamard(np.array(self.embeddings[d],dtype=np.float32), np.array(self.embeddings[g],dtype=np.float32)) for d,g,_ in all_pairs])
        y = np.array([l for _,_,l in all_pairs])
        logger.info(f"Link dataset: {len(positives)} pos + {len(neg_sample)} neg = {len(all_pairs)}")
        return X, y

    def train_link_predictor(self) -> Dict:
        import xgboost as xgb
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, roc_auc_score
        X, y = self._build_link_dataset()
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
        logger.info(f"Training link predictor: {len(X_tr)} train...")
        self.link_model = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05,
            subsample=0.6, colsample_bytree=0.6, reg_alpha=1.0, reg_lambda=2.0,
            eval_metric="logloss", random_state=42, verbosity=0)
        self.link_model.fit(X_tr, y_tr)
        acc = round(float(accuracy_score(y_te, self.link_model.predict(X_te))), 4)
        auc = round(float(roc_auc_score(y_te, self.link_model.predict_proba(X_te)[:,1])), 4)
        self._save(self.link_model, "link_model.pkl")
        logger.info(f"  Link predictor — acc={acc}, AUC={auc}")
        return {"accuracy": acc, "roc_auc": auc, "train_samples": len(X_tr), "test_samples": len(X_te), "status": "trained"}

    def predict_new_targets(self, drug: str, top_k: int = 10, confidence_threshold: float = 0.60) -> List[Dict]:
        if self.link_model is None: return [{"error": "Call POST /ml/train/link first"}]
        if not self.embeddings: return [{"error": "Call POST /ml/train/embeddings first"}]
        drug = drug.strip().upper()
        d_vec = self._vec(drug)
        if d_vec is None: return [{"error": f"'{drug}' not in embeddings"}]
        known = set()
        if drug in self.G:
            for nb in self.G.neighbors(drug):
                if self.G.nodes.get(nb,{}).get("type") == "gene": known.add(nb)
            stored = self.G.nodes[drug].get("targets", set())
            if isinstance(stored, set): known |= stored
        gene_nodes = [n for n,a in self.G.nodes(data=True) if a.get("type") == "gene" and n in self.embeddings]
        predictions = []
        for gene in gene_nodes:
            if gene in known: continue
            g_vec = self._vec(gene)
            if g_vec is None: continue
            prob = float(self.link_model.predict_proba(self._hadamard(d_vec, g_vec).reshape(1,-1))[0][1])
            if prob >= confidence_threshold:
                g_attr = self.G.nodes.get(gene, {})
                predictions.append({"predicted_gene": gene, "confidence": round(prob,4),
                    "novelty_status": "NOT in current graph — novel prediction",
                    "gene_source": g_attr.get("source",""), "longevity_influence": g_attr.get("longevity_influence","")})
        predictions.sort(key=lambda x: x["confidence"], reverse=True)
        return predictions[:top_k]

    # ═══════════════════════════════════════════════════════════════
    # PHASE 3 — DRUG SUCCESS PREDICTOR
    # ═══════════════════════════════════════════════════════════════
    def _resolve_drug_node(self, drug_name: str):
        return self._compound_lookup.get(str(drug_name).strip().upper())

    _ORGANISMS = ["caenorhabditis elegans","mus musculus","drosophila melanogaster","saccharomyces cerevisiae","rattus norvegicus"]

    def _extract_features(self, drug_name: str, organism: str, gene_scores: Dict) -> List[float]:
        drug_node = self._resolve_drug_node(drug_name)
        n_targets = n_hv = 0; best_gs = avg_gs = 0.0; n_pathways = n_ppi = 0
        has_genage = has_cellage = has_gendr = 0
        if drug_node and drug_node in self.G:
            gene_neighbors = [nb for nb in self.G.neighbors(drug_node) if self.G.nodes.get(nb,{}).get("type") == "gene"]
            stored = self.G.nodes[drug_node].get("targets", set())
            if isinstance(stored, set):
                for g in stored:
                    if g not in gene_neighbors and g in self.G and self.G.nodes.get(g,{}).get("type") == "gene":
                        gene_neighbors.append(g)
            n_targets = len(gene_neighbors); gs_vals = []
            for g in gene_neighbors:
                gs = gene_scores.get(g,{}).get("score",0.0); gs_vals.append(gs)
                src = str(self.G.nodes.get(g,{}).get("source","")).lower()
                if "genage" in src: has_genage = 1
                if "cellage" in src: has_cellage = 1
                if "gendr" in src: has_gendr = 1
                for nb2 in self.G.neighbors(g):
                    t = self.G.nodes.get(nb2,{}).get("type","")
                    if t == "pathway": n_pathways += 1
                    if t == "protein": n_ppi += 1
            if gs_vals: best_gs = max(gs_vals); avg_gs = sum(gs_vals)/len(gs_vals); n_hv = sum(1 for s in gs_vals if s >= 0.25)
        if n_hv == 0: mb = 0.0
        elif n_hv == 1: mb = 0.08
        elif n_hv == 2: mb = 0.22
        else: mb = min(0.22 + 0.18*math.log(n_hv-1), 0.65)
        org_lower = str(organism).lower()
        return [n_targets, n_hv, best_gs, avg_gs, n_pathways, n_ppi, has_genage, has_cellage, has_gendr, mb] + [1 if org in org_lower else 0 for org in self._ORGANISMS]

    def _get_gene_scores(self) -> Dict:
        for mod in ["agents.graph_agent", "graph_agent"]:
            try:
                import importlib; m = importlib.import_module(mod)
                return m.GraphAgent()._compute_all_gene_scores()
            except Exception: pass
        return {}

    def _build_success_dataset(self):
        cache_path = os.path.join(self.CACHE_DIR, "drugage.json")
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"DrugAge cache not found. Call GET /hagr/drugage first.")
        with open(cache_path) as f: records = json.load(f)
        gene_scores = self._get_gene_scores()
        matched = sum(1 for r in records if self._resolve_drug_node(r.get("drug","")) is not None)
        logger.info(f"DrugAge: {len(records)} records, {matched} matched ({matched/len(records)*100:.1f}%)")
        feature_names = ["n_aging_gene_targets","n_high_value_targets","best_gene_score","avg_gene_score",
            "n_pathways_covered","n_ppi_connections","has_genage_target","has_cellage_target","has_gendr_target",
            "multi_target_bonus","org_c_elegans","org_mouse","org_fly","org_yeast","org_rat"]
        X, y, skipped = [], [], 0
        for rec in records:
            try: lc = float(str(rec.get("lifespan_change","")).replace("%","").strip())
            except: skipped += 1; continue
            X.append(self._extract_features(rec.get("drug",""), rec.get("organism",""), gene_scores))
            y.append(1 if lc > 0 else 0)
        X_arr = np.array(X)
        nonzero = np.sum(X_arr.any(axis=1))
        logger.info(f"Success dataset: {len(X)} samples, {sum(y)} positive, {nonzero/len(X)*100:.1f}% non-zero rows")
        return X_arr, np.array(y), feature_names

    def train_success_predictor(self) -> Dict:
        import xgboost as xgb
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, roc_auc_score
        X, y, self.feature_names = self._build_success_dataset()
        if len(X) < 20: return {"error": "Not enough data. Call GET /hagr/drugage first."}
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        pos_count = int(sum(y_tr)); neg_count = len(y_tr) - pos_count
        scale = neg_count / pos_count if pos_count > 0 else 1.0
        logger.info(f"Training success predictor: {len(X_tr)} samples...")
        self.success_model = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=scale,
            eval_metric="auc", random_state=42, verbosity=0)
        self.success_model.fit(X_tr, y_tr)
        acc = round(float(accuracy_score(y_te, self.success_model.predict(X_te))), 4)
        auc = round(float(roc_auc_score(y_te, self.success_model.predict_proba(X_te)[:,1])), 4)
        feat_imp = sorted(zip(self.feature_names, self.success_model.feature_importances_.tolist()), key=lambda x: x[1], reverse=True)
        self._save(self.success_model, "success_model.pkl")
        self._save(self.feature_names, "feature_names.pkl")
        logger.info(f"  Success predictor — acc={acc}, AUC={auc}")
        return {"accuracy": acc, "roc_auc": auc, "train_samples": len(X_tr), "test_samples": len(X_te),
            "top_features": [{"feature": f, "importance": round(i,4)} for f,i in feat_imp[:8]], "status": "trained"}

    def predict_drug_success(self, drug: str) -> Dict:
        if self.success_model is None: return {"error": "Call POST /ml/train/success first"}
        gene_scores = self._get_gene_scores()
        drug_up = drug.strip().upper()
        node = self._resolve_drug_node(drug_up) or drug_up
        org = str(self.G.nodes.get(node,{}).get("organism","")) if node in self.G else ""
        feat = np.array([self._extract_features(drug_up, org, gene_scores)])
        prob = float(self.success_model.predict_proba(feat)[0][1])
        conf = "high" if prob >= 0.70 else "medium" if prob >= 0.50 else "low"
        contributions = sorted([{"feature": n, "value": round(v,4)} for n,v in zip(self.feature_names, feat[0].tolist()) if v > 0], key=lambda x: x["value"], reverse=True)
        return {"drug": drug_up, "success_probability": round(prob,4), "confidence": conf,
            "interpretation": f"{'High' if prob>=0.7 else 'Moderate' if prob>=0.5 else 'Low'} probability ({prob:.0%}) that {drug_up} extends lifespan.",
            "supporting_features": contributions[:5], "in_graph": node in self.G}

    # ═══════════════════════════════════════════════════════════════
    # FULL DRUG INTELLIGENCE REPORT
    # ═══════════════════════════════════════════════════════════════
    def full_drug_report(self, drug: str) -> Dict:
        drug_up = drug.strip().upper()
        known_targets = []
        node = self._resolve_drug_node(drug_up) or drug_up
        if node in self.G:
            stored = self.G.nodes[node].get("targets", set())
            if isinstance(stored, set): known_targets = sorted(list(stored))
            for nb in self.G.neighbors(node):
                if self.G.nodes.get(nb,{}).get("type") == "gene" and nb not in known_targets:
                    known_targets.append(nb)
        similar = [s for s in self.get_similar_nodes(drug_up, top_k=8) if s.get("type") in ("drug","longevity_compound") and "error" not in s][:3]
        novel   = [p for p in self.predict_new_targets(drug_up, top_k=5, confidence_threshold=0.55) if "error" not in p]
        success = self.predict_drug_success(drug_up)
        prob    = success.get("success_probability", 0.0)
        n_novel = len(novel)
        if prob >= 0.70 and n_novel >= 2: tier = "Tier 1 — Strong candidate"
        elif prob >= 0.55 or n_novel >= 1: tier = "Tier 2 — Moderate candidate"
        else: tier = "Tier 3 — Low priority"
        return {"drug": drug_up, "overall_tier": tier, "success_probability": prob,
            "known_targets": known_targets, "n_known_targets": len(known_targets),
            "similar_compounds": similar, "novel_target_predictions": novel,
            "n_novel_predictions": n_novel, "success_prediction": success}

    def train_all(self) -> Dict:
        results = {}
        results["embeddings"] = {"n_nodes": len(self.train_embeddings()), "status": "done"}
        results["link_predictor"] = self.train_link_predictor()
        results["success_predictor"] = self.train_success_predictor()
        return results

    def status(self) -> Dict:
        return {
            "graph": {"nodes": self.G.number_of_nodes(), "edges": self.G.number_of_edges(), "ready": self.G.number_of_nodes() > 50},
            "embeddings": {"trained": len(self.embeddings) > 0, "n_nodes": len(self.embeddings)},
            "link_predictor": {"trained": self.link_model is not None},
            "success_predictor": {"trained": self.success_model is not None},
            "paths": {"root": _ROOT, "graph": self.GRAPH_PATH, "models": self.MODEL_DIR},
            "next_step": self._next_step()
        }

    def _next_step(self) -> str:
        if self.G.number_of_nodes() < 50: return "Run populate_graph.py"
        if len(self.embeddings) < 50: return "Call POST /ml/train/embeddings"
        if self.link_model is None: return "Call POST /ml/train/link"
        if self.success_model is None: return "Call POST /ml/train/success"
        return "All ready — call GET /ml/report/{drug}"


# ── standalone ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*55)
    print("  ML AGENT  v3.0  FINAL")
    print("="*55)
    agent = MLAgent()
    s = agent.status()
    print(f"\nRoot     : {_ROOT}")
    print(f"Graph    : {s['graph']['nodes']} nodes, {s['graph']['edges']} edges")
    print(f"Graph.pkl: {agent.GRAPH_PATH}")

    if not s["graph"]["ready"]:
        print("\n  Graph empty.\n    1. uvicorn api.main:app --reload\n    2. python populate_graph.py")
        sys.exit(1)

    print("\n[Phase 1] Training embeddings...")
    embs = agent.train_embeddings()
    print(f"    {len(embs)} nodes embedded")

    print("\n[Phase 2] Training link predictor...")
    m2 = agent.train_link_predictor()
    print(f"    acc={m2['accuracy']}  AUC={m2['roc_auc']}")

    print("\n[Phase 3] Training drug success predictor...")
    m3 = agent.train_success_predictor()
    print(f"    acc={m3['accuracy']}  AUC={m3['roc_auc']}")
    print("  Top features:")
    for fi in m3.get("top_features",[])[:6]:
        print(f"    {fi['feature']:<32} {fi['importance']:.4f}  {'█'*int(fi['importance']*40)}")

    print("\n" + "="*55)
    print("  MODELS SAVED")
    print("="*55)
    print("\nTest: GET /ml/report/RAPAMYCIN")