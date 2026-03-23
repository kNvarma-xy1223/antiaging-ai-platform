from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import time
import math

from agents.data_agent import DataAgent
from agents.graph_agent import GraphAgent, _serialize_node_attrs
from visualize_graph import visualize_graph
from api.agents.ml_agent import MLAgent
ml_agent = MLAgent()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Anti-Aging AI Platform API",
    description=(
        "Multi-agent, multi-omics anti-aging research platform. "
        "Integrates HAGR databases (GenAge, DrugAge, CellAge, LongevityMap, AnAge, GenDR) "
        "with STRING, Reactome, ChEMBL, and ClinicalTrials.gov."
    ),
    version="2.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

data_agent  = DataAgent()
graph_agent = GraphAgent()


# ── Request timing ────────────────────────────────────────────────────────────
@app.middleware("http")
async def add_process_time_header(request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time-Ms"] = str(
        round((time.perf_counter() - start) * 1000, 2)
    )
    return response


def _safe_json(obj):
    """Recursively make any object JSON-safe (converts sets, NaN, etc.)."""
    if isinstance(obj, set):
        return sorted(list(obj))
    if isinstance(obj, dict):
        return {k: _safe_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_json(i) for i in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def safe_response(payload) -> JSONResponse:
    return JSONResponse(content=_safe_json(payload))


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {"message": "Anti-Aging AI Platform Running", "version": "2.0"}


@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy", "timestamp": time.time()}


# ── Q&A ───────────────────────────────────────────────────────────────────────
@app.get("/ask", tags=["Q&A"])
def ask(question: str = Query(..., description="Natural language question about aging")):
    result = data_agent.handle_question(question)
    if result.get("data"):
        try:
            graph_agent.process_query(question, result["data"])
        except Exception as e:
            logger.error(f"Graph ingestion error: {e}")
    return result


# ── HAGR databases ────────────────────────────────────────────────────────────
@app.get("/hagr/aging-genes", tags=["HAGR"])
def aging_genes():
    """GenAge Human — curated genes linked to human aging."""
    data = data_agent.get_aging_genes()
    if data:
        try:
            graph_agent.ingest({"aging_genes": data})
        except Exception as e:
            logger.error(f"Graph ingest error (aging_genes): {e}")
    return {"count": len(data), "data": data}
 
 
@app.get("/hagr/drugage", tags=["HAGR"])
def drugage():
    """DrugAge — compounds that modulate lifespan in model organisms."""
    data = data_agent.get_longevity_drugs()
    if data:
        try:
            graph_agent.ingest({"longevity_drugs": data})
        except Exception as e:
            logger.error(f"Graph ingest error (drugage): {e}")
    return {"count": len(data), "data": data}
 
 
@app.get("/hagr/cellage", tags=["HAGR"])
def cellage():
    """CellAge — genes experimentally linked to cellular senescence."""
    data = data_agent.get_cell_senescence_genes()
    if data:
        try:
            graph_agent.ingest({"cell_senescence": data})
        except Exception as e:
            logger.error(f"Graph ingest error (cellage): {e}")
    return {"count": len(data), "data": data}
 
 
@app.get("/hagr/longevitymap", tags=["HAGR"])
def longevitymap():
    """LongevityMap — human SNPs associated with longevity."""
    data = data_agent.get_longevity_variants()
    if data:
        try:
            graph_agent.ingest({"longevity_variants": data})
        except Exception as e:
            logger.error(f"Graph ingest error (longevitymap): {e}")
    return {"count": len(data), "data": data}
 
 
@app.get("/hagr/anage", tags=["HAGR"])
def anage():
    """AnAge — comparative aging data across 4,000+ animal species."""
    data = data_agent.get_species_aging_data()
    if data:
        try:
            graph_agent.ingest({"species_aging": data})
        except Exception as e:
            logger.error(f"Graph ingest error (anage): {e}")
    return {"count": len(data), "data": data}
 
 
@app.get("/hagr/gendr", tags=["HAGR"])
def gendr():
    """GenDR — genes mediating lifespan extension via dietary restriction."""
    data = data_agent.get_dietary_restriction_genes()
    if data:
        try:
            graph_agent.ingest({"dietary_restriction_genes": data})
        except Exception as e:
            logger.error(f"Graph ingest error (gendr): {e}")
    return {"count": len(data), "data": data}


# ── External Bio APIs ─────────────────────────────────────────────────────────
# ── CHANGE 2 of 2 ────────────────────────────────────────────────────────────
# All /bio/ endpoints now call graph_agent.ingest() directly after fetching.
# Previously ONLY /ask triggered ingest, so calling /bio/drug-targets/MTOR
# then /bio/drug-targets/TP53 directly never accumulated targets in the graph.
# Now every /bio/ call feeds the graph immediately — no /ask required.
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/bio/ppi/{gene}", tags=["External Bio APIs"])
def protein_interactions(gene: str):
    """STRING DB — high-confidence protein-protein interactions (score > 0.7)."""
    interactions = data_agent.get_protein_interactions(gene)
    if interactions:
        try:
            graph_agent.ingest({"protein_interactions": interactions})
        except Exception as e:
            logger.error(f"PPI graph ingest error: {e}")
    return {"gene": gene, "count": len(interactions), "data": interactions}


@app.get("/bio/pathways/{gene}", tags=["External Bio APIs"])
def gene_pathways(gene: str):
    """Reactome — curated biological pathways for a human gene."""
    pathways = data_agent.get_gene_pathways(gene)
    if pathways:
        try:
            graph_agent.ingest({"pathways": pathways})
        except Exception as e:
            logger.error(f"Pathways graph ingest error: {e}")
    return {"gene": gene, "count": len(pathways), "data": pathways}


@app.get("/bio/drug-targets/{gene}", tags=["External Bio APIs"])
def drug_targets(gene: str, fast: bool = False):
    """ChEMBL drug targets. fast=true skips cross-target enrichment."""
    """
    ChEMBL — drug molecules targeting a gene + cross-target enrichment.
    Results are automatically ingested into the knowledge graph.
    Call for multiple genes (MTOR, TP53, SIRT1) then run /graph/scores/drugs
    to see multi-gene drug repurposing scores.
    """
    targets = data_agent.get_drug_targets(gene, fast=fast)
    if targets:
        try:
            graph_agent.ingest({"drug_targets": targets})
        except Exception as e:
            logger.error(f"Drug targets graph ingest error: {e}")
    return {"gene": gene, "count": len(targets), "data": targets}

@app.get("/bio/compound-targets/{compound}", tags=["External Bio APIs"])
def compound_gene_targets(compound: str):
    """
    Correct compound → gene lookup via ChEMBL molecule search.
    Use for DrugAge compounds: RAPAMYCIN, METFORMIN, RESVERATROL etc.
    Creates RAPAMYCIN→MTOR edges (not the backwards CHEMBL84047→RAPAMYCIN).
    """
    targets = data_agent.get_compound_gene_targets(compound)
    if targets:
        try:
            graph_agent.ingest({"drug_targets": targets})
        except Exception as e:
            logger.error(f"Compound targets ingest error: {e}")
    return {"compound": compound, "count": len(targets), "data": targets}



@app.get("/bio/clinical-trials", tags=["External Bio APIs"])
def clinical_trials(query: str = Query(default="aging", description="Search term")):
    """ClinicalTrials.gov — active trials matching a query."""
    data = data_agent.get_clinical_trials(query)
    return {"query": query, "count": len(data), "data": data}


# ── Knowledge Graph ───────────────────────────────────────────────────────────
@app.get("/graph/stats", tags=["Knowledge Graph"])
def graph_stats():
    """Node/edge counts broken down by type."""
    G = graph_agent.G
    type_counts: dict = {}
    for _, attr in G.nodes(data=True):
        t = attr.get("type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1
    return {
        "nodes":            G.number_of_nodes(),
        "edges":            G.number_of_edges(),
        "node_type_counts": type_counts,
    }


@app.get("/graph/visualize", tags=["Knowledge Graph"])
def visualize():
    """Render the current knowledge graph as an image."""
    try:
        path = visualize_graph(graph_agent.G)
        return {"graph_image": path}
    except Exception as e:
        logger.error(f"Graph visualisation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/graph/scores/genes", tags=["Knowledge Graph"])
def gene_scores():
    """Top aging-related genes by multi-factor biological relevance score."""
    return safe_response(graph_agent.compute_gene_scores())


@app.get("/graph/scores/drugs", tags=["Knowledge Graph"])
def drug_scores():
    """
    Drug repurposing candidates scored by 4-pillar engine.
    For best results: call /bio/drug-targets/{gene} for several aging genes
    first (MTOR, TP53, SIRT1, FOXO3), then run this endpoint.
    """
    return safe_response(graph_agent.compute_drug_scores())


# ── Knowledge Graph Debug ─────────────────────────────────────────────────────
@app.get("/graph/node/{node}", tags=["Knowledge Graph Debug"])
def get_node(node: str):
    """Inspect a single node — attributes and direct connections."""
    node = node.upper()
    if node not in graph_agent.G:
        return {"error": "Node not found"}
    attrs = _serialize_node_attrs(dict(graph_agent.G.nodes[node]))
    return safe_response({
        "node":       node,
        "attributes": attrs,
        "neighbors":  list(graph_agent.G.neighbors(node)),
        "degree":     graph_agent.G.degree(node),
    })


@app.get("/graph/top/{node_type}", tags=["Knowledge Graph Debug"])
def top_nodes(node_type: str, k: int = 10):
    """Most-connected nodes of a given type."""
    return graph_agent.get_top_nodes(node_type, k)


@app.get("/graph/subgraph/{node}", tags=["Knowledge Graph Debug"])
def subgraph(node: str):
    """Local neighbourhood graph of a node + all 1-hop neighbours."""
    return safe_response(graph_agent.subgraph(node))


@app.get("/graph/degree-distribution", tags=["Knowledge Graph Debug"])
def degree_distribution():
    """Node connectivity distribution for graph sanity checks."""
    degrees = sorted([d for _, d in graph_agent.G.degree()], reverse=True)
    return {
        "top_degrees": degrees[:20],
        "avg_degree":  round(sum(degrees) / len(degrees), 3) if degrees else 0,
        "max_degree":  degrees[0] if degrees else 0,
        "total_nodes": len(degrees),
    }
    
# ── ML Agent ───────────────────────────────────────────────────────────────
@app.get("/ml/status", tags=["ML Agent"])
def ml_status():
    """Check which ML models are trained and what to do next."""
    return ml_agent.status()
 
 
@app.post("/ml/train/embeddings", tags=["ML Agent"])
def train_embeddings():
    """
    Phase 1: Spectral graph embeddings (scipy SVD — no extra install needed).
    Every node → 64-dim vector. Takes 10-30 seconds on CPU.
    Must run before link predictor.
    """
    ml_agent._reload_graph()   # always reload fresh from disk
    if ml_agent.G.number_of_nodes() < 50:
        raise HTTPException(
            status_code=400,
            detail=f"Graph has {ml_agent.G.number_of_nodes()} nodes. Run populate_graph.py first."
        )
    embs = ml_agent.train_embeddings()
    return {
        "status":    "trained",
        "n_nodes":   len(embs),
        "dim":       ml_agent.EMBED_DIM,
        "next_step": "Call POST /ml/train/link",
    }
 
 
@app.post("/ml/train/link", tags=["ML Agent"])
def train_link():
    """
    Phase 2: XGBoost link predictor.
    Learns which drug-gene connections are real vs random.
    Enables novel target prediction (pharma gold).
    Requires embeddings first.
    """
    if len(ml_agent.embeddings) < 50:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=400,
            detail="Embeddings not trained. Call POST /ml/train/embeddings first."
        )
    return ml_agent.train_link_predictor()
 
 
@app.post("/ml/train/success", tags=["ML Agent"])
def train_success():
    """
    Phase 3: XGBoost drug success predictor.
    Trained on DrugAge 3,400+ records with lifespan change labels.
    Answers: 'Can this drug extend lifespan?'
    Requires GET /hagr/drugage to have been called (populates cache).
    """
    return ml_agent.train_success_predictor()
 
 
@app.post("/ml/train/all", tags=["ML Agent"])
def train_all():
    """
    Train all 3 ML models in sequence:
    embeddings → link predictor → success predictor.
    Total time: 5-20 minutes on CPU.
    """
    return ml_agent.train_all()
 
 
@app.get("/ml/similar/{node}", tags=["ML Agent"])
def similar_nodes(node: str, top_k: int = 10):
    """
    Find biologically similar nodes using embedding cosine similarity.
    Works for any node: gene, drug, pathway, protein.
    Example: /ml/similar/MTOR  → genes closest to MTOR in vector space
    """
    return safe_response(ml_agent.get_similar_nodes(node, top_k=top_k))
 
 
@app.get("/ml/predict/targets/{drug}", tags=["ML Agent"])
def predict_targets(
    drug: str,
    top_k: int = 10,
    threshold: float = 0.60,
):
    """
    THE CORE PHARMA VALUE ENDPOINT.
    Predict NEW gene targets for a drug that are NOT in current graph.
    High confidence + not in literature = publication / patent potential.
    Example: /ml/predict/targets/CHEMBL84047
    """
    return safe_response(
        ml_agent.predict_new_targets(
            drug, top_k=top_k, confidence_threshold=threshold
        )
    )
 
 
@app.get("/ml/predict/success/{drug}", tags=["ML Agent"])
def predict_success(drug: str):
    """
    Predict probability (0→1) that a drug extends lifespan.
    Trained on DrugAge data — no molecular fingerprints needed.
    Example: /ml/predict/success/CHEMBL84047
    """
    return safe_response(ml_agent.predict_drug_success(drug))
 
 
@app.get("/ml/report/{drug}", tags=["ML Agent"])
def drug_report(drug: str):
    """
    Full drug intelligence report:
      - Known targets from graph
      - Similar compounds (embedding similarity)
      - Novel target predictions (link model)
      - Success probability (success model)
      - Overall tier (Tier 1 / 2 / 3)
 
    This is the output pharma pays for.
    Example: /ml/report/CHEMBL84047
    """
    return safe_response(ml_agent.full_drug_report(drug))
 