from __future__ import annotations

import json
from pathlib import Path


CASES = {
    "fresh_issue": {
        "gold": [
            "ellenbarrie_rhp_chunk_0067",
            "ellenbarrie_rhp_chunk_0214",
            "ellenbarrie_rhp_chunk_0279",
            "ellenbarrie_rhp_chunk_0969",
        ],
        "queries": [
            "What is the size of the Fresh Issue?",
            "What is the Fresh Issue amount?",
            "What amount does the Fresh Issue aggregate to?",
            "What is the amount of the Fresh Issue?",
        ],
    },
    "total_assets": {
        "gold": [
            "ellenbarrie_rhp_chunk_0218",
            "ellenbarrie_rhp_chunk_0636",
        ],
        "queries": [
            "What were the company's total assets as at March 31, 2025?",
            "What is the company's total assets as at March 31, 2025?",
            "What is the total assets value for March 31, 2025?",
            "What is the total assets amount for March 31, 2025?",
        ],
    },
    "manufacturing_operations": {
        "gold": [
            "ellenbarrie_rhp_chunk_0527",
            "ellenbarrie_rhp_chunk_0528",
            "ellenbarrie_rhp_chunk_0529",
            "ellenbarrie_rhp_chunk_0530",
            "ellenbarrie_rhp_chunk_0534",
            "ellenbarrie_rhp_chunk_0547",
            "ellenbarrie_rhp_chunk_0813",
        ],
        "queries": [
            "What does the prospectus say about the company's manufacturing operations?",
            "Which manufacturing facilities does the company operate?",
            "What are the company's manufacturing facilities and capacities?",
            "What are the company's manufacturing operations and capacity utilisation?",
        ],
    },
}


def token_count(model, text: str) -> int | None:
    tokenizer = getattr(model.model, "tokenizer", None)
    if tokenizer is None:
        try:
            tokenizer = model.model[0].tokenizer
        except Exception:
            return None

    encoded = tokenizer(
        text,
        add_special_tokens=True,
        truncation=False,
        return_attention_mask=False,
    )
    return len(encoded["input_ids"])


def main() -> None:
    print("Starting retrieval diagnostics...", flush=True)

    root = Path(__file__).resolve().parents[1]
    document_path = root / "data" / "processed" / "ellenbarrie_rhp.json"
    index_path = root / "vector_store" / "integration_test"

    print(f"Project root: {root}", flush=True)
    print(f"Document: {document_path}", flush=True)
    print(f"Index: {index_path}", flush=True)

    if not document_path.exists():
        raise FileNotFoundError(f"Missing document: {document_path}")
    if not index_path.exists():
        raise FileNotFoundError(f"Missing index directory: {index_path}")

    print("Importing project modules...", flush=True)

    from ipo_analyzer.chunking.semantic_chunker import chunk_document
    from ipo_analyzer.embeddings.embedding_model import EmbeddingModel
    from ipo_analyzer.retrieval.vector_store import FAISSVectorStore
    from ipo_analyzer.schemas.document import Document

    print("Loading document...", flush=True)
    data = json.loads(document_path.read_text(encoding="utf-8"))
    document = Document.model_validate(data)

    print("Chunking document...", flush=True)
    chunks = chunk_document(document)
    lookup = {c.chunk_id: c for c in chunks}

    print(f"Document: {document.company_name}", flush=True)
    print(f"Pages: {len(document.pages)}", flush=True)
    print(f"Chunks: {len(chunks)}", flush=True)

    print("Loading embedding model...", flush=True)
    model = EmbeddingModel()

    max_seq = getattr(model.model, "max_seq_length", None)
    print(f"Model: {model.model_name}", flush=True)
    print(f"Dimension: {model.dimension}", flush=True)
    print(f"max_seq_length: {max_seq}", flush=True)

    print("Loading FAISS index...", flush=True)
    store = FAISSVectorStore.load(index_path)
    print(f"Vectors: {store.size}", flush=True)

    # ---------------------------------------------------------------
    # A. Truncation audit
    # ---------------------------------------------------------------
    audit_ids = sorted({
        cid
        for case in CASES.values()
        for cid in case["gold"]
    } | {
        "ellenbarrie_rhp_chunk_0217",
        "ellenbarrie_rhp_chunk_0537",
        "ellenbarrie_rhp_chunk_0511",
    })

    print("\n" + "=" * 90, flush=True)
    print("A. CHUNK LENGTH / TRUNCATION AUDIT", flush=True)
    print("=" * 90, flush=True)

    for cid in audit_ids:
        c = lookup[cid]
        lower = c.content.lower()

        positions = {}
        for term in (
            "fresh issue",
            "total assets",
            "manufacturing",
            "capacity utilisation",
        ):
            p = lower.find(term)
            if p >= 0:
                positions[term] = p

        print(
            f"\n{cid}\n"
            f"  chars={len(c.content)}\n"
            f"  tokens={token_count(model, c.content)}\n"
            f"  pages={c.pages}\n"
            f"  section={c.section}\n"
            f"  subsection={c.subsection}\n"
            f"  keyword_positions={positions}",
            flush=True,
        )

    # ---------------------------------------------------------------
    # B. Local evidence experiment
    # ---------------------------------------------------------------
    local_targets = {
        "ellenbarrie_rhp_chunk_0218": (
            "total assets",
            "What were the company's total assets as at March 31, 2025?",
        ),
        "ellenbarrie_rhp_chunk_0636": (
            "total assets",
            "What were the company's total assets as at March 31, 2025?",
        ),
        "ellenbarrie_rhp_chunk_0067": (
            "fresh issue",
            "What is the size of the Fresh Issue?",
        ),
        "ellenbarrie_rhp_chunk_0214": (
            "fresh issue",
            "What is the size of the Fresh Issue?",
        ),
        "ellenbarrie_rhp_chunk_0279": (
            "fresh issue",
            "What is the size of the Fresh Issue?",
        ),
        "ellenbarrie_rhp_chunk_0969": (
            "fresh issue",
            "What is the size of the Fresh Issue?",
        ),
    }

    print("\n" + "=" * 90, flush=True)
    print("B. EVIDENCE-LOCALITY EXPERIMENT", flush=True)
    print("=" * 90, flush=True)

    for cid, (term, query) in local_targets.items():
        c = lookup[cid]
        lower = c.content.lower()
        p = lower.find(term)

        if p < 0:
            print(f"\n{cid}: '{term}' not found", flush=True)
            continue

        start = max(0, p - 350)
        end = min(len(c.content), p + 700)
        snippet = c.content[start:end]

        qv = model.encode_single(query)
        full_score = float(qv @ model.encode_single(c.content))
        local_score = float(qv @ model.encode_single(snippet))

        print(
            f"\n{cid}\n"
            f"  term='{term}' at char={p}\n"
            f"  full_chunk_similarity={full_score:.4f}\n"
            f"  local_snippet_similarity={local_score:.4f}",
            flush=True,
        )

    # ---------------------------------------------------------------
    # C. Query wording experiment
    # ---------------------------------------------------------------
    print("\n" + "=" * 90, flush=True)
    print("C. QUERY-WORDING EXPERIMENT", flush=True)
    print("=" * 90, flush=True)

    for case_name, case in CASES.items():
        gold = set(case["gold"])
        print(f"\n{'-' * 90}\nCASE: {case_name}", flush=True)

        for query in case["queries"]:
            qv = model.encode_single(query)
            scores, ids = store.search(qv, top_k=10)

            rows = list(zip(scores[0].tolist(), ids[0]))
            ranks = [
                rank
                for rank, (_, cid) in enumerate(rows, start=1)
                if cid in gold
            ]

            print(f"\nQUERY: {query}", flush=True)
            print(
                f"  First gold rank: {min(ranks) if ranks else 'None'}",
                flush=True,
            )

            for rank, (score, cid) in enumerate(rows, start=1):
                c = lookup[cid]
                marker = " <-- GOLD" if cid in gold else ""
                print(
                    f"  {rank:>2}. score={float(score):.4f}"
                    f" | {cid}{marker}"
                    f" | pages={c.pages}"
                    f" | subsection={c.subsection}",
                    flush=True,
                )

    print("\n" + "=" * 90, flush=True)
    print("DIAGNOSTICS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
