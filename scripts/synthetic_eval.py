#!/usr/bin/env python3
"""
Synthetic query evaluation pipeline for GAMI.

Generates queries from known segments, runs them through recall,
evaluates if the source segment is found, and tunes scoring weights.

Three phases:
1. Generate synthetic queries from high-value segments using vLLM
2. Run each query through recall and measure retrieval quality
3. Analyze results and output tuning recommendations
"""
import os, sys, json, time, logging, hashlib, requests, random
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api.config import settings
from api.llm.embeddings import embed_text_sync
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("synthetic_eval")

VLLM_URL = settings.VLLM_URL
OLLAMA_URL = settings.OLLAMA_URL
RESULTS_FILE = "/opt/gami/data/synthetic_eval_results.jsonl"
SUMMARY_FILE = "/opt/gami/data/synthetic_eval_summary.json"

os.makedirs("/opt/gami/data", exist_ok=True)

def call_vllm(prompt, max_tokens=500, temperature=0.3):
    """Call vLLM for query generation."""
    try:
        r = requests.post(f"{VLLM_URL}/chat/completions", json={
            "model": "qwen35-27b-unredacted",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }, timeout=60)
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        log.warning(f"vLLM error: {e}")
    return None

def generate_queries_for_segment(segment_id, segment_text, source_title):
    """Generate 3-5 synthetic queries that should retrieve this segment."""
    prompt = f"""Given this text from "{source_title}", generate 5 different questions that a user might ask where this text would be the correct answer. 

The questions should be natural — like how someone would actually ask about this information. Mix short factual questions with broader ones.

Text:
{segment_text[:2000]}

Return ONLY a JSON array of strings (the questions), nothing else:"""
    
    response = call_vllm(prompt)
    if not response:
        return []
    
    try:
        import re
        m = re.search(r'\[.*\]', response, re.DOTALL)
        if m:
            queries = json.loads(m.group())
            return [q.strip() for q in queries if isinstance(q, str) and len(q) > 10]
    except:
        pass
    return []

def run_vector_search(query_text, tenant_ids, limit=20):
    """Run vector search directly against the database."""
    try:
        engine = create_engine(settings.DATABASE_URL_SYNC)
        query_emb = embed_text_sync(query_text)
        vec = "[" + ",".join(str(v) for v in query_emb) + "]"
        
        with engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT segment_id, 
                       1 - (embedding <=> CAST(:vec AS vector)) as similarity,
                       left(text, 200) as preview,
                       source_id,
                       owner_tenant_id
                FROM segments 
                WHERE embedding IS NOT NULL
                AND owner_tenant_id = ANY(:tids)
                ORDER BY embedding <=> CAST(:vec AS vector) 
                LIMIT :lim
            """), {"vec": vec, "tids": tenant_ids, "lim": limit}).fetchall()
            
            return [{"segment_id": r[0], "similarity": float(r[1]), "preview": r[2], 
                     "source_id": r[3], "tenant_id": r[4]} for r in rows]
    except Exception as e:
        log.warning(f"Search error: {e}")
        return []

def run_lexical_search(query_text, tenant_ids, limit=20):
    """Run lexical search directly against the database."""
    try:
        engine = create_engine(settings.DATABASE_URL_SYNC)
        with engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT segment_id,
                       ts_rank(lexical_tsv, plainto_tsquery('english', :q)) as rank,
                       left(text, 200) as preview,
                       source_id
                FROM segments
                WHERE lexical_tsv @@ plainto_tsquery('english', :q)
                AND owner_tenant_id = ANY(:tids)
                ORDER BY rank DESC
                LIMIT :lim
            """), {"q": query_text, "tids": tenant_ids, "lim": limit}).fetchall()
            
            return [{"segment_id": r[0], "rank": float(r[1]), "preview": r[2], 
                     "source_id": r[3]} for r in rows]
    except Exception as e:
        log.warning(f"Lexical search error: {e}")
        return []

def evaluate_query(query_text, expected_segment_id, expected_source_id, tenant_ids):
    """Run a query and evaluate if the expected segment is found."""
    start = time.time()
    
    # Vector search
    vector_results = run_vector_search(query_text, tenant_ids)
    vector_time = time.time() - start
    
    # Lexical search  
    lex_start = time.time()
    lexical_results = run_lexical_search(query_text, tenant_ids)
    lexical_time = time.time() - lex_start
    
    # Check where the expected segment appears
    vector_rank = -1
    for i, r in enumerate(vector_results):
        if r["segment_id"] == expected_segment_id or r["source_id"] == expected_source_id:
            vector_rank = i + 1
            break
    
    lexical_rank = -1
    for i, r in enumerate(lexical_results):
        if r["segment_id"] == expected_segment_id or r["source_id"] == expected_source_id:
            lexical_rank = i + 1
            break
    
    # Combined rank (best of either)
    combined_rank = -1
    if vector_rank > 0 and lexical_rank > 0:
        combined_rank = min(vector_rank, lexical_rank)
    elif vector_rank > 0:
        combined_rank = vector_rank
    elif lexical_rank > 0:
        combined_rank = lexical_rank
    
    # Top similarity scores
    top_vector_sim = vector_results[0]["similarity"] if vector_results else 0
    expected_sim = 0
    for r in vector_results:
        if r["segment_id"] == expected_segment_id:
            expected_sim = r["similarity"]
            break
    
    return {
        "query": query_text,
        "expected_segment_id": expected_segment_id,
        "expected_source_id": expected_source_id,
        "vector_rank": vector_rank,
        "lexical_rank": lexical_rank,
        "combined_rank": combined_rank,
        "vector_time_ms": round(vector_time * 1000),
        "lexical_time_ms": round(lexical_time * 1000),
        "top_vector_similarity": round(top_vector_sim, 4),
        "expected_similarity": round(expected_sim, 4),
        "vector_results_count": len(vector_results),
        "lexical_results_count": len(lexical_results),
        "found_in_top_1": combined_rank == 1,
        "found_in_top_5": 0 < combined_rank <= 5,
        "found_in_top_10": 0 < combined_rank <= 10,
        "found_in_top_20": 0 < combined_rank <= 20,
        "not_found": combined_rank == -1,
    }

def main():
    engine = create_engine(settings.DATABASE_URL_SYNC)
    
    log.info("=" * 60)
    log.info("GAMI Synthetic Evaluation Pipeline")
    log.info("=" * 60)
    
    # Phase 1: Select high-value segments for query generation
    log.info("\n[Phase 1] Selecting high-value segments...")
    
    with engine.connect() as conn:
        # Get segments from memory files, docs, and key conversations
        segments = conn.execute(text("""
            SELECT s.segment_id, s.text, s.source_id, s.owner_tenant_id, 
                   src.title as source_title, length(s.text) as text_len
            FROM segments s
            JOIN sources src ON s.source_id = src.source_id
            WHERE s.embedding IS NOT NULL
            AND length(s.text) BETWEEN 100 AND 3000
            AND s.segment_type NOT IN ('tool_call', 'tool_result')
            AND (
                src.source_type = 'markdown'
                OR (src.source_type = 'conversation_session' AND s.text ILIKE '%CT%' AND length(s.text) > 300)
            )
            ORDER BY 
                CASE WHEN src.title ILIKE '%MEMORY%' THEN 0
                     WHEN src.title ILIKE '%network%' THEN 1
                     WHEN src.title ILIKE '%AI-STACK%' THEN 2
                     WHEN src.title ILIKE '%CLAUDE%' THEN 3
                     ELSE 4 END,
                length(s.text) DESC
            LIMIT 100
        """)).fetchall()
        
        log.info(f"Selected {len(segments)} segments for evaluation")
    
    # Phase 2: Generate synthetic queries
    log.info("\n[Phase 2] Generating synthetic queries via vLLM...")
    
    all_queries = []
    segments_processed = 0
    
    for seg_id, seg_text, source_id, tenant_id, source_title, text_len in segments:
        queries = generate_queries_for_segment(seg_id, seg_text, source_title or "unknown")
        
        for q in queries[:3]:  # Max 3 queries per segment
            all_queries.append({
                "query": q,
                "expected_segment_id": seg_id,
                "expected_source_id": source_id,
                "tenant_id": tenant_id,
                "source_title": source_title,
            })
        
        segments_processed += 1
        if segments_processed % 10 == 0:
            log.info(f"  Generated queries for {segments_processed}/{len(segments)} segments ({len(all_queries)} queries total)")
        
        time.sleep(0.5)  # Rate limit vLLM
        
        # Stop after we have enough queries
        if len(all_queries) >= 200:
            log.info(f"  Reached 200 queries, stopping generation")
            break
    
    log.info(f"Generated {len(all_queries)} synthetic queries from {segments_processed} segments")
    
    # Phase 3: Evaluate each query
    log.info("\n[Phase 3] Evaluating queries against recall...")
    
    results = []
    with open(RESULTS_FILE, "w") as f:
        for i, qinfo in enumerate(all_queries):
            tenant_ids = [qinfo["tenant_id"], "shared"]
            
            result = evaluate_query(
                qinfo["query"],
                qinfo["expected_segment_id"],
                qinfo["expected_source_id"],
                tenant_ids
            )
            result["source_title"] = qinfo["source_title"]
            results.append(result)
            
            f.write(json.dumps(result) + "\n")
            f.flush()
            
            if (i + 1) % 20 == 0:
                found_top5 = sum(1 for r in results if r["found_in_top_5"])
                found_top20 = sum(1 for r in results if r["found_in_top_20"])
                not_found = sum(1 for r in results if r["not_found"])
                log.info(f"  Evaluated {i+1}/{len(all_queries)}: "
                        f"top5={found_top5}/{i+1} ({100*found_top5/(i+1):.0f}%), "
                        f"top20={found_top20}/{i+1} ({100*found_top20/(i+1):.0f}%), "
                        f"missed={not_found}/{i+1}")
    
    # Phase 4: Analyze results
    log.info("\n[Phase 4] Analyzing results...")
    
    total = len(results)
    if total == 0:
        log.error("No results to analyze!")
        return
    
    top1 = sum(1 for r in results if r["found_in_top_1"])
    top5 = sum(1 for r in results if r["found_in_top_5"])
    top10 = sum(1 for r in results if r["found_in_top_10"])
    top20 = sum(1 for r in results if r["found_in_top_20"])
    missed = sum(1 for r in results if r["not_found"])
    
    avg_vector_time = sum(r["vector_time_ms"] for r in results) / total
    avg_lexical_time = sum(r["lexical_time_ms"] for r in results) / total
    avg_top_sim = sum(r["top_vector_similarity"] for r in results) / total
    avg_expected_sim = sum(r["expected_similarity"] for r in results if r["expected_similarity"] > 0) / max(1, sum(1 for r in results if r["expected_similarity"] > 0))
    
    # Find which queries failed
    failed_queries = [r for r in results if r["not_found"]]
    
    # Find queries where lexical beats vector
    lexical_wins = sum(1 for r in results if r["lexical_rank"] > 0 and (r["vector_rank"] == -1 or r["lexical_rank"] < r["vector_rank"]))
    vector_wins = sum(1 for r in results if r["vector_rank"] > 0 and (r["lexical_rank"] == -1 or r["vector_rank"] < r["lexical_rank"]))
    
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_queries": total,
        "segments_evaluated": segments_processed,
        "precision_at_1": round(top1 / total, 4),
        "precision_at_5": round(top5 / total, 4),
        "precision_at_10": round(top10 / total, 4),
        "precision_at_20": round(top20 / total, 4),
        "miss_rate": round(missed / total, 4),
        "avg_vector_latency_ms": round(avg_vector_time),
        "avg_lexical_latency_ms": round(avg_lexical_time),
        "avg_top_similarity": round(avg_top_sim, 4),
        "avg_expected_similarity": round(avg_expected_sim, 4),
        "vector_wins": vector_wins,
        "lexical_wins": lexical_wins,
        "recommendations": [],
        "failed_query_samples": [{"query": r["query"], "source": r["source_title"]} for r in failed_queries[:10]],
    }
    
    # Generate recommendations
    if summary["miss_rate"] > 0.3:
        summary["recommendations"].append(
            f"HIGH: {missed}/{total} queries missed entirely. Check embedding alignment — "
            f"GPU sentence-transformers vs Ollama embeddings may be in different vector spaces. "
            f"Consider re-embedding all segments with the same model."
        )
    
    if summary["precision_at_1"] < 0.3:
        summary["recommendations"].append(
            f"MEDIUM: Only {top1}/{total} found in top-1. Reranking weights need tuning. "
            f"Current avg similarity to expected: {avg_expected_sim:.4f} vs top result: {avg_top_sim:.4f}"
        )
    
    if lexical_wins > vector_wins * 1.5:
        summary["recommendations"].append(
            f"INFO: Lexical search outperforms vector ({lexical_wins} vs {vector_wins} wins). "
            f"Consider increasing lexical_weight in hybrid search from 0.3 to 0.5"
        )
    
    if avg_vector_time > 500:
        summary["recommendations"].append(
            f"PERF: Avg vector search latency {avg_vector_time:.0f}ms. Consider building HNSW index "
            f"instead of IVFFlat for better query performance."
        )
    
    if summary["precision_at_5"] > 0.6 and summary["precision_at_1"] < 0.4:
        summary["recommendations"].append(
            f"TUNING: Results are in top-5 ({top5}/{total}) but not top-1 ({top1}/{total}). "
            f"The right content is retrieved but poorly ranked. Focus on reranking weight tuning."
        )
    
    with open(SUMMARY_FILE, "w") as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    log.info("\n" + "=" * 60)
    log.info("EVALUATION RESULTS")
    log.info("=" * 60)
    log.info(f"Total queries evaluated: {total}")
    log.info(f"Precision@1:  {top1}/{total} ({100*top1/total:.1f}%)")
    log.info(f"Precision@5:  {top5}/{total} ({100*top5/total:.1f}%)")
    log.info(f"Precision@10: {top10}/{total} ({100*top10/total:.1f}%)")
    log.info(f"Precision@20: {top20}/{total} ({100*top20/total:.1f}%)")
    log.info(f"Missed:       {missed}/{total} ({100*missed/total:.1f}%)")
    log.info(f"Avg vector latency:  {avg_vector_time:.0f}ms")
    log.info(f"Avg lexical latency: {avg_lexical_time:.0f}ms")
    log.info(f"Vector wins: {vector_wins}, Lexical wins: {lexical_wins}")
    log.info(f"\nRecommendations:")
    for rec in summary["recommendations"]:
        log.info(f"  → {rec}")
    log.info(f"\nResults saved to: {RESULTS_FILE}")
    log.info(f"Summary saved to: {SUMMARY_FILE}")

if __name__ == "__main__":
    main()
