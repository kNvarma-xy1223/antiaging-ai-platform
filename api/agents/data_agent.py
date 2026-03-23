import requests
import zipfile
import io
import pandas as pd
import logging
import os
import json
import re
import math
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# HAGR DOWNLOAD URLS — All active databases
# ============================================================
HAGR_URLS = {
    "genage_human":  "https://genomics.senescence.info/genes/human_genes.zip",
    "drugage":       "https://genomics.senescence.info/drugs/dataset.zip",
    "cellage":       "https://genomics.senescence.info/cells/cellAge.zip",
    "longevitymap":  "https://genomics.senescence.info/longevity/longevity_genes.zip",
    "anage":         "https://genomics.senescence.info/species/dataset.zip",
    "gendr":         "https://genomics.senescence.info/diet/dataset.zip",
}

# ============================================================
# SAFE SCALAR — converts any value (including NaN/Inf) to JSON-safe type
# ============================================================
def safe_scalar(val, default=""):
    if val is None:
        return default
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return default
    if isinstance(val, (int, float)):
        return val
    s = str(val).strip()
    if s.lower() in ("nan", "none", "nat", ""):
        return default
    return s


def safe_str(val, default=""):
    result = safe_scalar(val, default)
    return str(result) if result != default else default


# ============================================================
# COLUMN FINDER — fuzzy column lookup by keyword fragment
# ============================================================
def find_col(df, *keywords):
    cols_lower = {c: c.lower() for c in df.columns}
    for col, col_l in cols_lower.items():
        if all(kw.lower() in col_l for kw in keywords):
            return col
    return None


def get_col(row, df, *keywords, default=""):
    col = find_col(df, *keywords)
    if col is None:
        return default
    return safe_str(row.get(col, default), default)


class DataAgent:

    def __init__(self):
        logger.info("DataAgent initialised")
        self.cache_dir = "data_cache"
        os.makedirs(self.cache_dir, exist_ok=True)

    # ============================================================
    # CACHE HELPERS
    # ============================================================
    def _cache_path(self, name):
        return os.path.join(self.cache_dir, f"{name}.json")

    def cache_data(self, name, data):
        if not data:
            logger.warning(f"[{name}] Skipping cache write — empty result.")
            return data
        try:
            with open(self._cache_path(name), "w") as f:
                json.dump(data, f)
            logger.info(f"[{name}] Cached {len(data)} records.")
        except Exception as e:
            logger.error(f"[{name}] Cache write failed: {e}")
        return data

    def load_cache(self, name):
        path = self._cache_path(name)
        if os.path.exists(path):
            try:
                with open(path) as f:
                    data = json.load(f)
                if data:
                    logger.info(f"[{name}] FALLBACK: loaded {len(data)} records from cache.")
                    return data
            except Exception as e:
                logger.error(f"[{name}] Cache read failed: {e}")
        return None

    # ============================================================
    # SAFE HTTP REQUEST
    # ============================================================
    def safe_request(self, url, retries=3, backoff=2.0):
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (compatible; AntiAgingResearchBot/1.0; "
                "+https://your-platform.com)"
            )
        }
        for attempt in range(1, retries + 1):
            try:
                r = requests.get(url, timeout=45, headers=headers)
                if r.status_code == 200:
                    try:
                        return r.json()
                    except Exception:
                        return []
                logger.warning(
                    f"[safe_request] HTTP {r.status_code} on attempt {attempt}: {url}"
                )
            except requests.RequestException as e:
                logger.error(f"[safe_request] Attempt {attempt} failed for {url}: {e}")
            if attempt < retries:
                time.sleep(backoff * attempt)
        return {}

    # ============================================================
    # COLUMN NORMALISER
    # ============================================================
    @staticmethod
    def _normalize_columns(df):
        df.columns = [
            c.strip().lower().replace(" ", "_").replace("-", "_")
            for c in df.columns
        ]
        return df

    # ============================================================
    # HAGR ZIP DOWNLOADER
    # ============================================================
    def _download_hagr_zip(self, db_name):
        url = HAGR_URLS.get(db_name)
        if not url:
            logger.error(f"[{db_name}] No URL configured.")
            return None

        logger.info(f"[{db_name}] Downloading from {url}")

        try:
            response = requests.get(
                url,
                timeout=90,
                headers={"User-Agent": "Mozilla/5.0 (compatible; HAGR-Client/2.0)"},
            )
        except requests.RequestException as e:
            logger.error(f"[{db_name}] Network error: {e}")
            return None

        if response.status_code != 200:
            logger.error(f"[{db_name}] HTTP {response.status_code} — download aborted.")
            return None

        content = response.content
        logger.info(f"[{db_name}] Downloaded {len(content):,} bytes")

        if not zipfile.is_zipfile(io.BytesIO(content)):
            text = content.decode("utf-8", errors="ignore")
            if "<html" in text.lower():
                logger.warning(f"[{db_name}] Received HTML — attempting table parse")
                try:
                    tables = pd.read_html(text)
                    df = self._normalize_columns(tables[0])
                    logger.info(f"[{db_name}] HTML parse OK — {len(df)} rows")
                    return df
                except Exception as e:
                    logger.error(f"[{db_name}] HTML parse failed: {e}")
                    return None
            logger.error(f"[{db_name}] Content is neither ZIP nor HTML — aborting.")
            return None

        try:
            with zipfile.ZipFile(io.BytesIO(content)) as z:
                files = z.namelist()
                logger.info(f"[{db_name}] ZIP contents: {files}")

                if db_name == "anage":
                    txt_files = [f for f in files if f.lower().endswith(".txt")]
                    target = txt_files[0] if txt_files else files[0]
                    with z.open(target) as fh:
                        raw_text = fh.read().decode("utf-8", errors="ignore")
                    df = pd.read_csv(io.StringIO(raw_text), sep="\t", on_bad_lines="skip")
                    df = self._normalize_columns(df)
                    logger.info(f"[{db_name}] Parsed {len(df)} rows — columns: {list(df.columns)}")
                    return df

                target = None
                for fname in files:
                    if fname.lower().endswith((".csv", ".tsv")):
                        target = fname
                        break
                if target is None:
                    target = files[0]

                with z.open(target) as fh:
                    raw_text = fh.read().decode("utf-8", errors="ignore")

                if "<html" in raw_text.lower():
                    tables = pd.read_html(raw_text)
                    df = self._normalize_columns(tables[0])
                elif target.lower().endswith(".tsv"):
                    df = pd.read_csv(io.StringIO(raw_text), sep="\t", on_bad_lines="skip")
                    df = self._normalize_columns(df)
                else:
                    df = pd.read_csv(io.StringIO(raw_text), on_bad_lines="skip")
                    df = self._normalize_columns(df)

                logger.info(f"[{db_name}] Parsed '{target}': {len(df)} rows — columns: {list(df.columns)}")
                return df

        except zipfile.BadZipFile as e:
            logger.error(f"[{db_name}] Bad zip file: {e}")
            return None
        except Exception as e:
            logger.error(f"[{db_name}] Extraction error: {e}")
            return None

    # ============================================================
    # 1. GENAGE HUMAN
    # ============================================================
    def get_aging_genes(self):
        df = self._download_hagr_zip("genage_human")
        if df is None:
            logger.warning("[genage_human] Download failed — trying cache.")
            return self.load_cache("genage_human") or []

        sym_col = find_col(df, "gene", "symbol") or find_col(df, "symbol") or find_col(df, "gene")
        if sym_col is None:
            logger.error(f"[genage_human] Cannot locate gene symbol column. Columns: {list(df.columns)}")
            return self.load_cache("genage_human") or []

        genes = []
        for _, row in df.iterrows():
            symbol = row.get(sym_col)
            if pd.isna(symbol) or str(symbol).strip() in ("", "nan"):
                continue
            genes.append({
                "gene":                safe_str(symbol),
                "name":                safe_str(row.get("name", "")),
                "entrez_id":           safe_str(row.get("entrez_gene_id") or row.get("entrez_id", "")),
                "aliases":             safe_str(row.get("aliases", "")),
                "longevity_influence": safe_str(row.get("longevity_influence", "")),
                "why_selected":        safe_str(row.get("why_selected", "")),
                "source":              "GenAge Human",
            })

        logger.info(f"[genage_human] {len(genes)} genes loaded.")
        if genes:
            self.cache_data("genage_human", genes)
        else:
            return self.load_cache("genage_human") or []
        return genes

    # ============================================================
    # 2. DRUGAGE
    # ============================================================
    def get_longevity_drugs(self):
        df = self._download_hagr_zip("drugage")
        if df is None:
            logger.warning("[drugage] Download failed — trying cache.")
            return self.load_cache("drugage") or []

        name_col    = find_col(df, "compound") or find_col(df, "drug") or find_col(df, "name")
        avg_col     = find_col(df, "avg", "lifespan") or find_col(df, "avg_lifespan")
        max_col     = find_col(df, "max", "lifespan") or find_col(df, "max_lifespan")
        species_col = find_col(df, "species") or find_col(df, "organism")

        logger.info(f"[drugage] Resolved columns — name: {name_col}, avg_ls: {avg_col}, max_ls: {max_col}, species: {species_col}")

        if name_col is None:
            logger.error(f"[drugage] Cannot locate compound name column. Columns: {list(df.columns)}")
            return self.load_cache("drugage") or []

        drugs = []
        for _, row in df.iterrows():
            name_val = row.get(name_col)
            if pd.isna(name_val) or str(name_val).strip() in ("", "nan"):
                continue
            drugs.append({
                "drug":            safe_str(name_val),
                "organism":        safe_str(row.get(species_col, "")) if species_col else "",
                "strain":          safe_str(row.get("strain", "")),
                "dosage":          safe_str(row.get("dosage", "")),
                "lifespan_change": safe_str(row.get(avg_col, "")) if avg_col else "",
                "max_lifespan":    safe_str(row.get(max_col, "")) if max_col else "",
                "gender":          safe_str(row.get("gender", "")),
                "significance":    safe_str(row.get("significance", "")),
                "pubmed_id":       safe_str(row.get("pubmed_id", "")),
                "source":          "DrugAge",
            })

        logger.info(f"[drugage] {len(drugs)} entries loaded.")
        if drugs:
            self.cache_data("drugage", drugs)
        else:
            return self.load_cache("drugage") or []
        return drugs

    # ============================================================
    # 3. CELLAGE
    # ============================================================
    def get_cell_senescence_genes(self):
        df = self._download_hagr_zip("cellage")
        if df is None:
            logger.warning("[cellage] Download failed — trying cache.")
            return self.load_cache("cellage") or []

        sym_col    = find_col(df, "gene", "symbol") or find_col(df, "gene")
        name_col   = find_col(df, "gene", "name") or find_col(df, "name")
        cell_col   = find_col(df, "cell", "line") or find_col(df, "cell")
        effect_col = find_col(df, "senescence", "effect") or find_col(df, "effect")

        logger.info(f"[cellage] Resolved columns — symbol: {sym_col}, name: {name_col}, cell_line: {cell_col}, effect: {effect_col}")

        if sym_col is None:
            logger.error(f"[cellage] Cannot locate gene symbol column. Columns: {list(df.columns)}")
            return self.load_cache("cellage") or []

        genes = []
        for _, row in df.iterrows():
            symbol = row.get(sym_col)
            if pd.isna(symbol) or str(symbol).strip() in ("", "nan"):
                continue
            genes.append({
                "gene":              safe_str(symbol),
                "gene_name":         safe_str(row.get(name_col, "")) if name_col else "",
                "cell_line":         safe_str(row.get(cell_col, "")) if cell_col else "",
                "senescence_effect": safe_str(row.get(effect_col, "")) if effect_col else "",
                "type":              safe_str(row.get("type", "")),
                "entrez_gene_id":    safe_str(row.get("entrez_gene_id", "")),
                "source":            "CellAge",
            })

        logger.info(f"[cellage] {len(genes)} genes loaded.")
        if genes:
            self.cache_data("cellage", genes)
        else:
            return self.load_cache("cellage") or []
        return genes

    # ============================================================
    # 4. LONGEVITYMAP
    # ============================================================
    def get_longevity_variants(self):
        df = self._download_hagr_zip("longevitymap")
        if df is None:
            logger.warning("[longevitymap] Download failed — trying cache.")
            return self.load_cache("longevitymap") or []

        gene_col  = find_col(df, "gene") or find_col(df, "symbol")
        var_col   = find_col(df, "variant") or find_col(df, "snp") or find_col(df, "rsid")
        assoc_col = find_col(df, "association") or find_col(df, "result")
        pop_col   = find_col(df, "population") or find_col(df, "country")

        logger.info(f"[longevitymap] Resolved columns — gene: {gene_col}, variant: {var_col}, assoc: {assoc_col}, pop: {pop_col}")

        if gene_col is None:
            logger.error(f"[longevitymap] Cannot locate gene column. Columns: {list(df.columns)}")
            return self.load_cache("longevitymap") or []

        variants = []
        for _, row in df.iterrows():
            gene = row.get(gene_col)
            if pd.isna(gene) or str(gene).strip() in ("", "nan"):
                continue
            variants.append({
                "gene":        safe_str(gene),
                "variant":     safe_str(row.get(var_col, "")) if var_col else "",
                "association": safe_str(row.get(assoc_col, "")) if assoc_col else "",
                "population":  safe_str(row.get(pop_col, "")) if pop_col else "",
                "pubmed_id":   safe_str(row.get("pubmed_id", "")),
                "source":      "LongevityMap",
            })

        logger.info(f"[longevitymap] {len(variants)} variants loaded.")
        if variants:
            self.cache_data("longevitymap", variants)
        else:
            return self.load_cache("longevitymap") or []
        return variants

    # ============================================================
    # 5. ANAGE
    # ============================================================
    def get_species_aging_data(self):
        df = self._download_hagr_zip("anage")
        if df is None:
            logger.warning("[anage] Download failed — trying cache.")
            return self.load_cache("anage") or []

        common_col    = find_col(df, "common", "name") or find_col(df, "common")
        species_col   = find_col(df, "species")
        genus_col     = find_col(df, "genus")
        longevity_col = find_col(df, "longevity") or find_col(df, "maximum")
        mass_col      = find_col(df, "body", "mass") or find_col(df, "mass")
        metab_col     = find_col(df, "metabolic")
        quality_col   = find_col(df, "data", "quality") or find_col(df, "quality")

        logger.info(f"[anage] Resolved columns — common: {common_col}, species: {species_col}, longevity: {longevity_col}, mass: {mass_col}")

        species_list = []
        for _, row in df.iterrows():
            name = row.get(common_col) if common_col else None
            if name is None or pd.isna(name) or str(name).strip() in ("", "nan"):
                genus   = safe_str(row.get(genus_col, "")) if genus_col else ""
                species = safe_str(row.get(species_col, "")) if species_col else ""
                name = f"{genus} {species}".strip() if genus or species else None
            if not name:
                continue

            species_list.append({
                "species":           safe_str(name),
                "kingdom":           safe_str(row.get("kingdom", "")),
                "phylum":            safe_str(row.get("phylum", "")),
                "class":             safe_str(row.get("class", "")),
                "order":             safe_str(row.get("order", "")),
                "family":            safe_str(row.get("family", "")),
                "max_longevity_yrs": safe_scalar(row.get(longevity_col) if longevity_col else None),
                "body_mass_g":       safe_scalar(row.get(mass_col) if mass_col else None),
                "metabolic_rate_w":  safe_scalar(row.get(metab_col) if metab_col else None),
                "data_quality":      safe_str(row.get(quality_col, "")) if quality_col else "",
                "source":            "AnAge",
            })

        logger.info(f"[anage] {len(species_list)} species loaded.")
        if species_list:
            self.cache_data("anage", species_list)
        else:
            return self.load_cache("anage") or []
        return species_list

    # ============================================================
    # 6. GENDR
    # ============================================================
    def get_dietary_restriction_genes(self):
        df = self._download_hagr_zip("gendr")
        if df is None:
            logger.warning("[gendr] Download failed — trying cache.")
            return self.load_cache("gendr") or []

        df = df.loc[:, ~df.columns.str.contains("unnamed")]

        sym_col    = find_col(df, "gene", "symbol") or find_col(df, "gene")
        org_col    = find_col(df, "organism") or find_col(df, "species")
        effect_col = find_col(df, "lifespan", "effect") or find_col(df, "effect")
        dr_col     = find_col(df, "dr", "type") or find_col(df, "dr_type") or find_col(df, "type")

        logger.info(f"[gendr] Resolved columns — symbol: {sym_col}, organism: {org_col}, effect: {effect_col}, dr_type: {dr_col}")

        if sym_col is None:
            logger.error(f"[gendr] Cannot locate gene symbol column. Columns: {list(df.columns)}")
            return self.load_cache("gendr") or []

        genes = []
        for _, row in df.iterrows():
            symbol = row.get(sym_col)
            if pd.isna(symbol) or str(symbol).strip() in ("", "nan"):
                continue
            genes.append({
                "gene":            safe_str(symbol),
                "organism":        safe_str(row.get(org_col, "")) if org_col else "",
                "lifespan_effect": safe_str(row.get(effect_col, "")) if effect_col else "",
                "dr_type":         safe_str(row.get(dr_col, "")) if dr_col else "",
                "source":          "GenDR",
            })

        logger.info(f"[gendr] {len(genes)} genes loaded.")
        if genes:
            self.cache_data("gendr", genes)
        else:
            return self.load_cache("gendr") or []
        return genes

    # ============================================================
    # STRING DB
    # ============================================================
    def get_protein_interactions(self, gene):
        url = (
            f"https://string-db.org/api/json/network"
            f"?identifiers={gene}&species=9606&required_score=700"
        )
        data = self.safe_request(url)
        if not isinstance(data, list):
            return []
        edges = []
        for i in data:
            score = i.get("score", 0)
            if score > 0.7:
                edges.append({
                    "proteinA": i.get("preferredName_A", ""),
                    "proteinB": i.get("preferredName_B", ""),
                    "score":    round(float(score), 4),
                })
        return edges

    # ============================================================
    # UNIPROT RESOLVER
    # ============================================================
    def resolve_uniprot(self, gene):
        url = (
            f"https://rest.uniprot.org/uniprotkb/search"
            f"?query=gene:{gene}+AND+organism_id:9606+AND+reviewed:true"
            f"&fields=accession&format=json&size=1"
        )
        data = self.safe_request(url)
        if isinstance(data, dict):
            results = data.get("results", [])
            if results:
                return results[0].get("primaryAccession")
        return None

    # ============================================================
    # REACTOME PATHWAYS
    # ============================================================
    def get_gene_pathways(self, gene):
        uniprot = self.resolve_uniprot(gene)
        if not uniprot:
            return []
        url = (
            f"https://reactome.org/ContentService/data/mapping/UniProt"
            f"/{uniprot}/pathways"
        )
        data = self.safe_request(url)
        if not isinstance(data, list):
            return []
        return [
            {"gene": gene, "name": p.get("displayName", ""), "stId": p.get("stId", "")}
            for p in data[:20]
        ]

    # ============================================================
    # CHEMBL DRUG TARGETS — with activity-based cross-target enrichment
    # ============================================================

    # Class-level cache: ChEMBL target_id → gene_symbol
    # Shared across all queries, survives the full server session
    _chembl_target_gene_cache: dict = {}

    def resolve_chembl_target(self, gene: str) -> str:
        """Resolve a gene symbol to its primary ChEMBL target ID."""
        url = f"https://www.ebi.ac.uk/chembl/api/data/target/search.json?q={gene}"
        data = self.safe_request(url)
        if not isinstance(data, dict):
            return None
        targets = data.get("targets", [])
        return targets[0]["target_chembl_id"] if targets else None

    def _resolve_chembl_target_id_to_gene(self, target_chembl_id: str) -> str:
        """
        Resolve a ChEMBL target ID → gene symbol.
        Cached in-memory — each unique target ID is fetched only once per
        server session, making repeated cross-target lookups free.
        """
        if target_chembl_id in DataAgent._chembl_target_gene_cache:
            return DataAgent._chembl_target_gene_cache[target_chembl_id]

        url = f"https://www.ebi.ac.uk/chembl/api/data/target/{target_chembl_id}.json"
        data = self.safe_request(url)
        gene_sym = ""
        if isinstance(data, dict):
            for comp in data.get("target_components", []):
                for syn in comp.get("target_component_synonyms", []):
                    if syn.get("syn_type") == "GENE_SYMBOL":
                        gene_sym = syn.get("component_synonym", "").strip().upper()
                        break
                if gene_sym:
                    break

        DataAgent._chembl_target_gene_cache[target_chembl_id] = gene_sym
        return gene_sym

    def _get_cross_targets(self, molecule_chembl_id: str, exclude_gene: str) -> list:
        """
        Reverse-query ChEMBL: given a drug molecule, find ALL genes it targets.

        WHY ACTIVITY ENDPOINT (not mechanism endpoint):
        ------------------------------------------------
        The mechanism endpoint (/mechanism.json) only contains drugs with a
        formally DEFINED mechanism of action — mostly approved drugs.
        Most ChEMBL entries are assay hits without defined MOA → mechanism
        endpoint returns empty for ~95% of molecules → cross-targets = 0.

        The activity endpoint (/activity.json?molecule_chembl_id=X) returns
        EVERY assay that molecule was ever tested in, across ALL targets.
        This is the correct reverse-query: "what genes has this drug touched?"

        Example: CHEMBL84047 queried by molecule returns activities against
        MTOR, AKT1, PIK3CA, RPTOR, RPS6KB1 — all genuine biological targets.
        Without this, the drug shows only MTOR (the gene we queried under).

        We deduplicate by target_chembl_id and resolve each to gene symbol
        using the cached _resolve_chembl_target_id_to_gene().
        """
        url = (
            f"https://www.ebi.ac.uk/chembl/api/data/activity.json"
            f"?molecule_chembl_id={molecule_chembl_id}&limit=50"
        )
        data = self.safe_request(url)
        if not isinstance(data, dict):
            return []

        pairs = []
        seen_targets: set = set()

        for activity in data.get("activities", []):
            tid = activity.get("target_chembl_id", "")
            if not tid or tid in seen_targets:
                continue
            seen_targets.add(tid)

            gene_sym = self._resolve_chembl_target_id_to_gene(tid)
            if not gene_sym or gene_sym == exclude_gene.strip().upper():
                continue

            pairs.append({
                "drug":            molecule_chembl_id,
                "target":          gene_sym,
                "activity_type":   activity.get("standard_type", ""),
                "activity_value":  str(activity.get("standard_value", "")),
                "activity_units":  activity.get("standard_units", ""),
                "assay_chembl_id": activity.get("assay_chembl_id", ""),
            })
            logger.debug(f"[cross_target] {molecule_chembl_id} → {gene_sym}")

        return pairs

    def get_drug_targets(self, gene: str, fast: bool = False) -> list:
        """
        ChEMBL — drug molecules targeting a gene, with activity-based cross-target enrichment.

        TWO-PHASE APPROACH:

        Phase 1 — Primary (target → drugs):
          Query ChEMBL activities for gene's target ID.
          Returns up to 40 drug molecules all tagged target=gene.

        Phase 2 — Reverse enrich (drug → all targets):
          For each unique molecule (top 12), call activity endpoint BY MOLECULE.
          This reverse query returns ALL genes that drug has ever been tested
          against — the correct way to discover polypharmacology.

          Using activity-by-molecule (not mechanism) because:
          - mechanism endpoint: only ~5% of drugs have defined MOA entries
          - activity-by-molecule: 100% of drugs have activity records

        Result: each drug accumulates ALL its known gene targets, not just the
        one gene it was queried under. Drug repurposing scores correctly reflect
        multi-target biology.
        """
        target_id = self.resolve_chembl_target(gene)
        if not target_id:
            logger.warning(f"[chembl] Could not resolve target for: {gene}")
            return []

        url = (
            f"https://www.ebi.ac.uk/chembl/api/data/activity.json"
            f"?target_chembl_id={target_id}&limit=40"
        )
        data = self.safe_request(url)
        if not isinstance(data, dict):
            return []

        # ── Phase 1: Primary drug-gene pairs (gene → drugs) ──────────────
        primary_pairs = []
        unique_mols: list = []
        seen_mols: set = set()

        for a in data.get("activities", []):
            mol = a.get("molecule_chembl_id")
            if not mol:
                continue
            primary_pairs.append({
                "drug":            mol,
                "target":          gene,
                "activity_type":   a.get("standard_type", ""),
                "activity_value":  safe_str(a.get("standard_value", "")),
                "activity_units":  a.get("standard_units", ""),
                "assay_chembl_id": a.get("assay_chembl_id", ""),
            })
            if mol not in seen_mols:
                seen_mols.add(mol)
                unique_mols.append(mol)

        logger.info(
            f"[chembl] {gene}: {len(primary_pairs)} activity records, "
            f"{len(unique_mols)} unique molecules"
        )

        # ── Phase 2: Reverse enrich — activity by molecule (drug → targets) ──
        # Top 12 molecules only: sweet spot between coverage and response time.
        # Each call: 1 activity query + up to 5 cached target resolutions.
        # Target resolution is cached → repeated genes cost 0 extra calls.
        cross_pairs = []
        if not fast:
            for mol in unique_mols[:12]:
                extra = self._get_cross_targets(mol, exclude_gene=gene)
                cross_pairs.extend(extra)
                if extra:
                    gene_names = [p["target"] for p in extra]
                    logger.info(
                        f"[cross_target] {mol}: +{len(extra)} targets → {gene_names}"
                    )
        else:
            logger.info(f"[chembl] {gene}: fast mode — skipping cross-targets")

        all_pairs = primary_pairs + cross_pairs
        logger.info(
            f"[chembl] {gene}: total {len(all_pairs)} drug-gene pairs "
            f"({len(primary_pairs)} primary + {len(cross_pairs)} cross-targets)"
        )
        return all_pairs
    
    # ============================================================
    # CHEMBL COMPOUND → GENE TARGETS (correct direction)
    # ============================================================
    def get_compound_gene_targets(self, compound_name: str) -> list:
        """
        Correct lookup: compound name → gene targets.
        compound_name → ChEMBL molecule ID → mechanism/activity → genes

        Different from get_drug_targets() which treats input as a GENE name.
        Use this for DrugAge compounds: RAPAMYCIN, METFORMIN, RESVERATROL etc.
        """
        # Step 1: find ChEMBL molecule ID for this compound name
        url = (f"https://www.ebi.ac.uk/chembl/api/data/molecule/search.json"
               f"?q={compound_name}&limit=5")
        data = self.safe_request(url)
        if not isinstance(data, dict):
            return []

        molecules = data.get("molecules", [])
        if not molecules:
            logger.warning(f"[compound_targets] No ChEMBL molecule for: {compound_name}")
            return []

        mol_id = molecules[0].get("molecule_chembl_id", "")
        if not mol_id:
            return []

        logger.info(f"[compound_targets] {compound_name} → {mol_id}")

        # Step 2: get mechanism of action (gene targets)
        url2 = (f"https://www.ebi.ac.uk/chembl/api/data/mechanism.json"
                f"?molecule_chembl_id={mol_id}&limit=20")
        data2 = self.safe_request(url2)
        pairs = []
        seen = set()

        if isinstance(data2, dict):
            for mech in data2.get("mechanisms", []):
                tid = mech.get("target_chembl_id", "")
                if not tid or tid in seen:
                    continue
                seen.add(tid)
                gene_sym = self._resolve_chembl_target_id_to_gene(tid)
                if not gene_sym:
                    continue
                pairs.append({
                    "drug":           compound_name.upper(),
                    "target":         gene_sym,
                    "activity_type":  mech.get("mechanism_of_action", ""),
                    "activity_value": "",
                    "activity_units": "",
                    "assay_chembl_id": mol_id,
                })
                logger.info(f"[compound_targets] {compound_name} → {gene_sym}")

        # Step 3: fallback to activity endpoint if no mechanisms found
        if not pairs:
            url3 = (f"https://www.ebi.ac.uk/chembl/api/data/activity.json"
                    f"?molecule_chembl_id={mol_id}&limit=30")
            data3 = self.safe_request(url3)
            if isinstance(data3, dict):
                seen2 = set()
                for act in data3.get("activities", []):
                    tid = act.get("target_chembl_id", "")
                    if not tid or tid in seen2:
                        continue
                    seen2.add(tid)
                    gene_sym = self._resolve_chembl_target_id_to_gene(tid)
                    if not gene_sym:
                        continue
                    pairs.append({
                        "drug":           compound_name.upper(),
                        "target":         gene_sym,
                        "activity_type":  act.get("standard_type", ""),
                        "activity_value": str(act.get("standard_value", "")),
                        "activity_units": act.get("standard_units", ""),
                        "assay_chembl_id": mol_id,
                    })

        logger.info(f"[compound_targets] {compound_name}: {len(pairs)} gene targets")
        return pairs


    # ============================================================
    # CLINICAL TRIALS
    # ============================================================
    def get_clinical_trials(self, query):
        url = (
            f"https://clinicaltrials.gov/api/query/study_fields"
            f"?expr={query}&fields=NCTId,Phase,BriefTitle&min_rnk=1&max_rnk=20&fmt=json"
        )
        data = self.safe_request(url)
        if not isinstance(data, dict):
            return []
        studies = data.get("StudyFieldsResponse", {}).get("StudyFields", [])
        trials = []
        for s in studies:
            trials.append({
                "trial_id": s["NCTId"][0] if s.get("NCTId") else None,
                "phase":    s["Phase"][0] if s.get("Phase") else None,
                "title":    s["BriefTitle"][0] if s.get("BriefTitle") else None,
            })
        return trials

    # ============================================================
    # GENE EXTRACTOR
    # ============================================================
    def extract_gene(self, question):
        IGNORE = {
            "PROTEIN", "INTERACTION", "INTERACTIONS", "DRUG", "DRUGS",
            "TARGET", "TARGETING", "GENE", "GENES", "AGING", "AGE",
            "PATHWAY", "PATHWAYS", "TRIAL", "TRIALS", "PPI",
            "WHAT", "WHICH", "LIST", "SHOW", "FOR", "THE", "AND",
            "WITH", "FROM", "ABOUT", "DATA", "ALL", "HUMAN", "SPECIES",
        }
        candidates = []
        for w in re.findall(r"[A-Za-z0-9]+", question):
            g = w.upper()
            if g in IGNORE:
                continue
            if re.match(r"^[A-Z][A-Z0-9]{2,7}$", g):
                candidates.append(g)
        return candidates[-1] if candidates else None

    # ============================================================
    # QUESTION ROUTER
    # ============================================================
    def route_question(self, q):
        ql = q.lower()
        tasks = []
        if any(x in ql for x in ["interaction", "ppi", "protein-protein"]):
            tasks.append("ppi")
        if "pathway" in ql:
            tasks.append("pathways")
        if "drug" in ql and any(x in ql for x in ["target", "targeting"]):
            tasks.append("drug_targets")
        if any(x in ql for x in ["aging gene", "longevity gene", "genes for aging"]):
            tasks.append("aging_genes")
        if any(x in ql for x in ["longevity", "lifespan", "drugage", "compound"]):
            tasks.append("longevity_drugs")
        if any(x in ql for x in ["senescence", "cell age", "cellage"]):
            tasks.append("cellage")
        if any(x in ql for x in ["variant", "longevitymap", "snp", "gwas"]):
            tasks.append("longevitymap")
        if any(x in ql for x in ["species", "anage", "animal", "comparative"]):
            tasks.append("anage")
        if any(x in ql for x in ["diet", "restriction", "gendr", "caloric"]):
            tasks.append("gendr")
        if "trial" in ql:
            tasks.append("clinical_trials")
        return tasks

    # ============================================================
    # MAIN HANDLER
    # ============================================================
    def handle_question(self, question):
        gene  = self.extract_gene(question)
        tasks = self.route_question(question)
        data  = {}

        if gene:
            if "ppi" in tasks:
                data["protein_interactions"] = self.get_protein_interactions(gene)
            if "pathways" in tasks:
                data["pathways"] = self.get_gene_pathways(gene)
            if "drug_targets" in tasks:
                data["drug_targets"] = self.get_drug_targets(gene)

        if "aging_genes"     in tasks:
            data["aging_genes"]              = self.get_aging_genes()
        if "longevity_drugs" in tasks:
            data["longevity_drugs"]           = self.get_longevity_drugs()
        if "cellage"         in tasks:
            data["cell_senescence"]           = self.get_cell_senescence_genes()
        if "longevitymap"    in tasks:
            data["longevity_variants"]        = self.get_longevity_variants()
        if "anage"           in tasks:
            data["species_aging"]             = self.get_species_aging_data()
        if "gendr"           in tasks:
            data["dietary_restriction_genes"] = self.get_dietary_restriction_genes()
        if "clinical_trials" in tasks:
            data["clinical_trials"]           = self.get_clinical_trials("aging")

        return {
            "question": question,
            "gene":     gene,
            "tasks":    tasks,
            "data":     data,
        }