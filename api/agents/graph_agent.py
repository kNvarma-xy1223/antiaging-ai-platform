# ===============GraphAgent.py: Knowledge Graph Builder & Reasoner================
import networkx as nx
import os
import json
import pickle
import hashlib
import logging
import math
from collections import defaultdict

logger = logging.getLogger(__name__)


def _serialize_node_attrs(attrs: dict) -> dict:
    """Convert non-JSON-serialisable types (sets, NaN) at the API boundary."""
    result = {}
    for k, v in attrs.items():
        if isinstance(v, set):
            result[k] = sorted(list(v))
        elif isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            result[k] = None
        else:
            result[k] = v
    return result


class GraphAgent:

    def __init__(self, storage_dir="graph_storage"):
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)

        self.graph_path = os.path.join(self.storage_dir, "graph.pkl")
        self.cache_path = os.path.join(self.storage_dir, "query_cache.json")

        self.G = self._load_graph()
        self.query_cache = self._load_cache()

        logger.info(
            f"GraphAgent initialized | "
            f"Nodes: {self.G.number_of_nodes()}, "
            f"Edges: {self.G.number_of_edges()}"
        )

    # ============================================================
    # LOAD / SAVE
    # ============================================================

    def _load_graph(self):
        if os.path.exists(self.graph_path):
            try:
                with open(self.graph_path, "rb") as f:
                    G = pickle.load(f)
                logger.info(
                    f"Graph loaded from disk: "
                    f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
                )
                return G
            except Exception as e:
                logger.error(f"Graph load failed: {e}")

        legacy = os.path.join(self.storage_dir, "graph.gpickle")
        if os.path.exists(legacy):
            try:
                with open(legacy, "rb") as f:
                    G = pickle.load(f)
                logger.info(f"Migrated legacy graph.gpickle → graph.pkl")
                self._save_graph_obj(G)
                return G
            except Exception as e:
                logger.error(f"Legacy graph migrate failed: {e}")

        return nx.Graph()

    def _save_graph(self):
        self._save_graph_obj(self.G)

    def _save_graph_obj(self, G):
        try:
            with open(self.graph_path, "wb") as f:
                pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            logger.error(f"Graph save failed: {e}")

    def _load_cache(self):
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path) as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_cache(self):
        try:
            with open(self.cache_path, "w") as f:
                json.dump(self.query_cache, f)
        except Exception as e:
            logger.error(f"Cache save failed: {e}")

    # ============================================================
    # QUERY HASHING
    # ============================================================

    # ── CHANGE 1 of 2 ────────────────────────────────────────────────────────
    # Old: _hash_query(question) — hashed question text ONLY.
    # Problem: /ask?question=drug targets for MTOR processed once, never again.
    #          Adding TP53/SIRT1 drug data doesn't re-trigger ingest because
    #          the question string hasn't changed.
    # Fix: fingerprint = question + data sizes. Same question with MORE data
    #      (e.g. after calling /bio/drug-targets/TP53) forces fresh ingest.
    # ─────────────────────────────────────────────────────────────────────────
    def _hash_query(self, question: str, data: dict = None) -> str:
        fingerprint = question.lower().strip()
        if data:
            try:
                sizes = json.dumps(
                    {k: len(v) for k, v in data.items() if isinstance(v, list)},
                    sort_keys=True,
                )
                fingerprint += sizes
            except Exception:
                pass
        return hashlib.md5(fingerprint.encode()).hexdigest()

    def is_query_processed(self, question: str, data: dict = None) -> bool:
        return self._hash_query(question, data) in self.query_cache

    def mark_query_processed(self, question: str, data: dict = None) -> None:
        self.query_cache[self._hash_query(question, data)] = True
        self._save_cache()

    # ============================================================
    # NODE / EDGE HELPERS
    # ============================================================

    def add_node_safe(self, node, node_type, **attrs):
        if not node:
            return

        node = str(node).strip().upper()
        if not node or node in ("NAN", "NONE", ""):
            return

        if node not in self.G:
            self.G.add_node(node, type=node_type, **attrs)
        else:
            for k, v in attrs.items():
                if k == "targets":
                    existing = self.G.nodes[node].get("targets", set())
                    if not isinstance(existing, set):
                        existing = set(existing) if existing else set()
                    if isinstance(v, (set, list)):
                        existing.update(v)
                    self.G.nodes[node]["targets"] = existing
                else:
                    if v not in (None, "", "nan"):
                        self.G.nodes[node][k] = v

    def add_edge_safe(self, source, target, edge_type, weight=1.0, **attrs):
        if not source or not target:
            return

        source = str(source).strip().upper()
        target = str(target).strip().upper()

        if not source or not target or source == target:
            return

        if self.G.has_edge(source, target):
            self.G[source][target]["weight"] += weight
        else:
            self.G.add_edge(source, target, type=edge_type, weight=weight, **attrs)

    # ============================================================
    # INGEST
    # ============================================================

    def ingest(self, data):
        if not data:
            return

        for g in data.get("aging_genes", []):
            gene = g.get("gene")
            self.add_node_safe(gene, "gene",
                source=g.get("source"),
                longevity_influence=g.get("longevity_influence", ""),
                name=g.get("name", ""),
            )

        for d in data.get("longevity_drugs", []):
            drug     = d.get("drug")
            organism = d.get("organism")
            lifespan = d.get("lifespan_change")
            self.add_node_safe(drug, "drug")
            if organism:
                self.add_node_safe(organism, "organism")
            if drug and organism:
                self.add_edge_safe(drug, organism, "extends_lifespan",
                                   weight=1.0, effect=lifespan)

        for g in data.get("cell_senescence", []):
            gene   = g.get("gene")
            effect = g.get("senescence_effect")
            self.add_node_safe(gene, "gene",
                source=g.get("source", "CellAge"),
                senescence_role=effect or "",
            )
            if effect:
                self.add_edge_safe(gene, effect, "senescence_effect")

        for v in data.get("longevity_variants", []):
            gene    = v.get("gene")
            variant = v.get("variant")
            self.add_node_safe(gene, "gene")
            self.add_node_safe(variant, "variant")
            if gene and variant:
                self.add_edge_safe(gene, variant, "has_variant")

        for s in data.get("species_aging", []):
            species   = s.get("species")
            longevity = s.get("max_longevity_yrs")
            self.add_node_safe(species, "species", longevity=longevity)

        for g in data.get("dietary_restriction_genes", []):
            gene     = g.get("gene")
            organism = g.get("organism")
            self.add_node_safe(gene, "gene",
                source=g.get("source", "GenDR"),
                dr_effect=g.get("lifespan_effect", ""),
            )
            if organism:
                self.add_node_safe(organism, "organism")
                self.add_edge_safe(gene, organism, "dietary_restriction_effect")

        for p in data.get("protein_interactions", []):
            a     = p.get("proteinA")
            b     = p.get("proteinB")
            score = p.get("score", 0.7)
            self.add_node_safe(a, "protein")
            self.add_node_safe(b, "protein")
            self.add_edge_safe(a, b, "ppi", weight=score)

        for p in data.get("pathways", []):
            gene    = p.get("gene")
            pathway = p.get("name")
            self.add_node_safe(gene, "gene")
            self.add_node_safe(pathway, "pathway")
            self.add_edge_safe(gene, pathway, "involved_in")

        for d in data.get("drug_targets", []):
            drug = d.get("drug")
            gene = d.get("target")
            if not drug or not gene:
                continue

            drug = str(drug).strip().upper()
            gene = str(gene).strip().upper()

            if drug not in self.G:
                self.G.add_node(drug, type="drug", targets=set())
            else:
                self.G.nodes[drug]["type"] = "drug"
                existing = self.G.nodes[drug].get("targets", set())
                if not isinstance(existing, set):
                    existing = set(existing) if existing else set()
                self.G.nodes[drug]["targets"] = existing

            self.G.nodes[drug]["targets"].add(gene)
            self.add_node_safe(gene, "gene")
            self.add_edge_safe(drug, gene, "targets")

        self._save_graph()
        logger.info(
            f"Graph updated — "
            f"Nodes: {self.G.number_of_nodes()}, "
            f"Edges: {self.G.number_of_edges()}"
        )

    # ============================================================
    # PROCESS QUERY (entry from /ask)
    # ============================================================

    def process_query(self, question: str, data: dict) -> None:
        """
        Entry point from /ask endpoint.
        Dedup fingerprint now includes data sizes (Change 1) so the same
        question re-ingests when new drug-target data is available.
        """
        if self.is_query_processed(question, data):
            logger.info("Query already processed with same data — skipping ingest.")
            return

        self.ingest(data)
        self.mark_query_processed(question, data)
        logger.info(f"Graph updated from query: '{question[:80]}'")

    # ============================================================
    # ANALYTICS
    # ============================================================

    def get_top_nodes(self, node_type="gene", top_k=10):
        nodes = [n for n, attr in self.G.nodes(data=True)
                 if attr.get("type") == node_type]
        degree_scores = [(n, self.G.degree(n)) for n in nodes]
        degree_scores.sort(key=lambda x: x[1], reverse=True)
        return degree_scores[:top_k]

    def get_neighbors(self, node):
        node = node.upper()
        if node not in self.G:
            return []
        return list(self.G.neighbors(node))

    def subgraph(self, node):
        node = node.upper()
        if node not in self.G:
            return {}
        neighbors = list(self.G.neighbors(node))
        sub = self.G.subgraph([node] + neighbors)
        return {
            "nodes": [
                (n, _serialize_node_attrs(dict(attr)))
                for n, attr in sub.nodes(data=True)
            ],
            "edges": list(sub.edges(data=True)),
        }

    # ============================================================
    # GENE SCORING ENGINE
    # ============================================================

    def _compute_all_gene_scores(self) -> dict:
        """
        Score EVERY gene — no top-k cutoff.
        Used internally by compute_drug_scores() so every target gene
        across all queries contributes to drug repurposing calculations.
        """
        scores = {}

        total_nodes = self.G.number_of_nodes()
        if total_nodes == 0:
            return scores

        degree_dict = dict(self.G.degree())
        max_degree  = max(degree_dict.values()) if degree_dict else 1

        for node, attr in self.G.nodes(data=True):
            if attr.get("type") != "gene":
                continue

            gene   = node
            degree = degree_dict.get(gene, 0)

            centrality_score = math.log1p(degree) / math.log1p(max_degree)

            source = str(attr.get("source", ""))
            if "GenAge" in source:
                evidence_score = 1.0
            elif "CellAge" in source:
                evidence_score = 0.75
            elif "GenDR" in source:
                evidence_score = 0.65
            else:
                evidence_score = 0.4

            li = str(attr.get("longevity_influence", "")).lower()
            if li in ("pro-longevity", "anti-longevity"):
                evidence_score = min(evidence_score + 0.10, 1.0)

            drug_edges = sum(
                1 for n in self.G.neighbors(gene)
                if self.G.nodes[n].get("type") == "drug"
            )
            drug_score = 0.1 + 0.9 * min(drug_edges / 10, 1.0)

            pathway_edges = sum(
                1 for n in self.G.neighbors(gene)
                if self.G.nodes[n].get("type") == "pathway"
            )
            pathway_score = 0.1 + 0.9 * min(pathway_edges / 10, 1.0)

            ppi_edges = sum(
                1 for n in self.G.neighbors(gene)
                if self.G.nodes[n].get("type") == "protein"
                and self.G[gene][n].get("type") == "ppi"
            )
            ppi_score = min(ppi_edges / 10, 1.0)

            coverage   = degree / total_nodes if total_nodes else 0
            confidence = 0.7 + 0.3 * coverage

            raw_score = (
                0.40 * centrality_score
                + 0.25 * evidence_score
                + 0.15 * drug_score
                + 0.12 * pathway_score
                + 0.08 * ppi_score
            )
            final_score = raw_score * confidence

            scores[gene] = {
                "score": round(final_score, 4),
                "components": {
                    "centrality":   round(centrality_score, 4),
                    "evidence":     round(evidence_score, 4),
                    "druggability": round(drug_score, 4),
                    "pathway":      round(pathway_score, 4),
                    "ppi":          round(ppi_score, 4),
                    "coverage":     round(coverage, 4),
                    "confidence":   round(confidence, 4),
                },
                "explanation": {
                    "degree":              degree,
                    "drug_connections":    drug_edges,
                    "pathway_connections": pathway_edges,
                    "ppi_connections":     ppi_edges,
                    "source":              source,
                },
            }

        return scores

    def compute_gene_scores(self):
        all_scores = self._compute_all_gene_scores()
        sorted_scores = sorted(
            all_scores.items(), key=lambda x: x[1]["score"], reverse=True
        )
        return sorted_scores[:20]

    # ============================================================
    # DRUG REPURPOSING ENGINE
    # ============================================================

    def compute_drug_scores(self):
        all_gene_scores = self._compute_all_gene_scores()

        if not all_gene_scores:
            logger.warning("[drug_scores] No gene scores — ask some questions first.")
            return []

        max_gene_score = max(
            (v["score"] for v in all_gene_scores.values()), default=1.0
        ) or 1.0

        HIGH_VALUE_THRESHOLD = 0.25

        drug_results = {}

        for node, attr in self.G.nodes(data=True):
            if attr.get("type") != "drug":
                continue

            drug = node

            stored_targets = attr.get("targets", set())
            if not isinstance(stored_targets, set):
                stored_targets = set(stored_targets) if stored_targets else set()

            edge_targets = {
                nb for nb in self.G.neighbors(drug)
                if self.G.nodes[nb].get("type") == "gene"
                and self.G[drug][nb].get("type") == "targets"
            }
            all_targets = stored_targets | edge_targets

            if not all_targets:
                continue

            scored_targets = sorted(
                [
                    (g, all_gene_scores[g]["score"] if g in all_gene_scores else 0.05)
                    for g in all_targets
                ],
                key=lambda x: x[1],
                reverse=True,
            )

            scores_only = [s for _, s in scored_targets]
            n_targets   = len(scored_targets)

            weighted_sum = sum(s * (0.80 ** i) for i, (_, s) in enumerate(scored_targets))
            max_possible = max_gene_score * sum(0.80 ** i for i in range(n_targets))
            direct_score = weighted_sum / max_possible if max_possible else 0.0

            high_value = [(g, s) for g, s in scored_targets if s >= HIGH_VALUE_THRESHOLD]
            n_hv = len(high_value)
            if n_hv == 0:
                multi_bonus = 0.0
            elif n_hv == 1:
                multi_bonus = 0.08
            elif n_hv == 2:
                multi_bonus = 0.22
            else:
                multi_bonus = min(0.22 + 0.18 * math.log(n_hv - 1), 0.65)

            indirect_scores = []
            for gene, g_score in scored_targets:
                if gene not in self.G:
                    continue
                for ppi_nb in self.G.neighbors(gene):
                    edge = self.G[gene].get(ppi_nb, {})
                    if (
                        self.G.nodes[ppi_nb].get("type") == "protein"
                        and edge.get("type") == "ppi"
                        and ppi_nb in all_gene_scores
                    ):
                        ppi_w = min(float(edge.get("weight", 0.7)), 1.0)
                        indirect_scores.append(
                            all_gene_scores[ppi_nb]["score"] * ppi_w * 0.35
                        )
            indirect_part = (
                min(sum(indirect_scores) / len(indirect_scores), 0.50)
                if indirect_scores else 0.0
            )

            avg_target_score = sum(scores_only) / n_targets
            low_ratio        = sum(1 for s in scores_only if s < 0.10) / n_targets
            selectivity      = max(
                (1.0 - 0.35 * low_ratio) * min(avg_target_score / 0.25, 1.0), 0.05
            )

            raw_score = (
                0.40 * direct_score
                + 0.30 * multi_bonus
                + 0.20 * indirect_part
                + 0.10 * selectivity
            )
            final_score = round(raw_score * selectivity, 4)

            drug_results[drug] = {
                "score":              final_score,
                "tier":               self._score_tier(final_score),
                "targets":            sorted(list(all_targets)),
                "num_targets":        n_targets,
                "high_value_targets": [g for g, s in high_value],
                "num_high_value":     n_hv,
                "score_breakdown": {
                    "direct_target_score":   round(direct_score, 4),
                    "multi_target_bonus":    round(multi_bonus, 4),
                    "indirect_ppi_effect":   round(indirect_part, 4),
                    "selectivity_factor":    round(selectivity, 4),
                    "avg_target_gene_score": round(avg_target_score, 4),
                },
                "top_3_targets": [
                    {"gene": g, "gene_score": round(s, 4)}
                    for g, s in scored_targets[:3]
                ],
                "explanation": (
                    f"Targets {n_targets} aging gene(s), "
                    f"{n_hv} high-value (score≥{HIGH_VALUE_THRESHOLD}): "
                    f"{', '.join(g for g, _ in scored_targets[:3])}. "
                    f"Repurposing score: {final_score:.4f}."
                ),
            }

        sorted_drugs = sorted(
            drug_results.items(), key=lambda x: x[1]["score"], reverse=True
        )   

        # ============================================================
        # PHARMA-GRADE DIVERSITY FILTER
        # ============================================================
        seen_signatures = set()
        diverse_drugs = []

        for drug, data in sorted_drugs:

            # signature = unique biological effect (sorted targets)
            signature = tuple(sorted(data["targets"]))

            if signature in seen_signatures:
                continue

            seen_signatures.add(signature)
            diverse_drugs.append((drug, data))

            if len(diverse_drugs) >= 20:
                break

        logger.info(
            f"[drug_scores] Scored {len(drug_results)} drugs → "
            f"{len(diverse_drugs)} after diversity filter"
        )

        return diverse_drugs

    @staticmethod
    def _score_tier(score: float) -> str:
        if score >= 0.55: return "Tier 1 — Strong repurposing candidate"
        if score >= 0.35: return "Tier 2 — Moderate repurposing candidate"
        if score >= 0.15: return "Tier 3 — Weak / needs validation"
        return                   "Tier 4 — Low confidence"