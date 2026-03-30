#!/usr/bin/env python3
"""Synthetic eval v2 - uses Ollama for query gen, direct DB search for eval."""
import os, sys, json, time, logging, requests, re, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api.config import settings
from api.llm.embeddings import embed_text_sync
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("eval")

RESULTS_FILE = "/opt/gami/data/synthetic_eval_results.jsonl"
SUMMARY_FILE = "/opt/gami/data/synthetic_eval_summary.json"
os.makedirs("/opt/gami/data", exist_ok=True)

def call_ollama(prompt):
    try:
        r = requests.post(f"{settings.OLLAMA_URL}/api/generate", json={
            "model": "qwen3:8b", "prompt": prompt, "stream": False,
            "options": {"temperature": 0.3, "num_predict": 500, "num_ctx": 4096}
        }, timeout=60)
        if r.status_code == 200:
            resp = r.json().get("response", "")
            # Strip thinking blocks
            resp = re.sub(r'<think>.*?</think>', '', resp, flags=re.DOTALL).strip()
            return resp
    except Exception as e:
        log.warning(f"Ollama error: {e}")
    return None

def gen_queries(seg_text, title):
    prompt = f"""/no_think
Given this text from "{title}", generate 3 different questions a user would ask where this text is the answer. Return ONLY a JSON array of strings.

Text: {seg_text[:1500]}

JSON array:"""
    resp = call_ollama(prompt)
    if not resp:
        return []
    try:
        m = re.search(r'\[.*?\]', resp, re.DOTALL)
        if m:
            return [q.strip() for q in json.loads(m.group()) if isinstance(q, str) and len(q) > 10][:3]
    except:
        pass
    return []

def vector_search(query_emb, tenant_ids, limit=20):
    engine = create_engine(settings.DATABASE_URL_SYNC)
    vec = "[" + ",".join(str(v) for v in query_emb) + "]"
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT segment_id, 1 - (embedding <=> CAST(:v AS vector)) as sim, source_id
            FROM segments WHERE embedding IS NOT NULL AND owner_tenant_id = ANY(:t)
            ORDER BY embedding <=> CAST(:v AS vector) LIMIT :l
        """), {"v": vec, "t": tenant_ids, "l": limit}).fetchall()
        return [{"segment_id": r[0], "similarity": float(r[1]), "source_id": r[2]} for r in rows]

def lexical_search(query, tenant_ids, limit=20):
    engine = create_engine(settings.DATABASE_URL_SYNC)
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT segment_id, ts_rank(lexical_tsv, plainto_tsquery('english', :q)) as rank, source_id
            FROM segments WHERE lexical_tsv @@ plainto_tsquery('english', :q) AND owner_tenant_id = ANY(:t)
            ORDER BY rank DESC LIMIT :l
        """), {"q": query, "t": tenant_ids, "l": limit}).fetchall()
        return [{"segment_id": r[0], "rank": float(r[1]), "source_id": r[2]} for r in rows]

def evaluate(query, expected_seg_id, expected_src_id, tenants):
    t0 = time.time()
    qemb = embed_text_sync(query)
    embed_ms = int((time.time() - t0) * 1000)
    
    t0 = time.time()
    vr = vector_search(qemb, tenants)
    vec_ms = int((time.time() - t0) * 1000)
    
    t0 = time.time()
    lr = lexical_search(query, tenants)
    lex_ms = int((time.time() - t0) * 1000)
    
    # Find rank of expected in vector results
    v_rank = next((i+1 for i,r in enumerate(vr) if r["segment_id"]==expected_seg_id or r["source_id"]==expected_src_id), -1)
    l_rank = next((i+1 for i,r in enumerate(lr) if r["segment_id"]==expected_seg_id or r["source_id"]==expected_src_id), -1)
    best = min(r for r in [v_rank, l_rank] if r > 0) if any(r > 0 for r in [v_rank, l_rank]) else -1
    
    return {
        "query": query, "expected_segment_id": expected_seg_id,
        "vector_rank": v_rank, "lexical_rank": l_rank, "best_rank": best,
        "embed_ms": embed_ms, "vec_ms": vec_ms, "lex_ms": lex_ms,
        "top_sim": round(vr[0]["similarity"], 4) if vr else 0,
        "expected_sim": round(next((r["similarity"] for r in vr if r["segment_id"]==expected_seg_id), 0), 4),
        "in_top1": best == 1, "in_top5": 0 < best <= 5,
        "in_top10": 0 < best <= 10, "in_top20": 0 < best <= 20,
        "missed": best == -1,
    }

def main():
    engine = create_engine(settings.DATABASE_URL_SYNC)
    
    log.info("=" * 50)
    log.info("GAMI Synthetic Evaluation v2")
    log.info("=" * 50)
    
    # Phase 1: Get segments
    with engine.connect() as conn:
        segs = conn.execute(text("""
            SELECT s.segment_id, s.text, s.source_id, s.owner_tenant_id, src.title
            FROM segments s JOIN sources src ON s.source_id = src.source_id
            WHERE s.embedding IS NOT NULL AND length(s.text) BETWEEN 100 AND 3000
            AND s.segment_type NOT IN ('tool_call','tool_result','chunk')
            ORDER BY CASE WHEN src.title ILIKE '%MEMORY%' THEN 0 WHEN src.title ILIKE '%network%' THEN 1
                WHEN src.title ILIKE '%AI-STACK%' THEN 2 ELSE 3 END, length(s.text) DESC
            LIMIT 80
        """)).fetchall()
    log.info(f"Selected {len(segs)} segments")
    
    # Phase 2: Generate queries
    log.info("Generating queries via Ollama qwen3:8b...")
    all_q = []
    for i, (sid, stxt, srcid, tid, title) in enumerate(segs):
        queries = gen_queries(stxt, title or "doc")
        for q in queries:
            all_q.append({"query": q, "seg_id": sid, "src_id": srcid, "tenant": tid, "title": title})
        if (i+1) % 10 == 0:
            log.info(f"  {i+1}/{len(segs)} segs, {len(all_q)} queries")
        if len(all_q) >= 150:
            break
    log.info(f"Generated {len(all_q)} queries")
    
    # Phase 3: Evaluate
    log.info("Evaluating...")
    results = []
    with open(RESULTS_FILE, "w") as f:
        for i, qi in enumerate(all_q):
            r = evaluate(qi["query"], qi["seg_id"], qi["src_id"], [qi["tenant"], "shared"])
            r["source_title"] = qi["title"]
            results.append(r)
            f.write(json.dumps(r) + "\n")
            if (i+1) % 25 == 0:
                t1 = sum(1 for x in results if x["in_top1"])
                t5 = sum(1 for x in results if x["in_top5"])
                miss = sum(1 for x in results if x["missed"])
                log.info(f"  {i+1}/{len(all_q)}: P@1={t1}/{i+1} P@5={t5}/{i+1} miss={miss}/{i+1}")
    
    # Phase 4: Analyze
    N = len(results)
    if N == 0:
        log.error("No results!")
        return
    
    t1 = sum(1 for r in results if r["in_top1"])
    t5 = sum(1 for r in results if r["in_top5"])
    t10 = sum(1 for r in results if r["in_top10"])
    t20 = sum(1 for r in results if r["in_top20"])
    miss = sum(1 for r in results if r["missed"])
    vw = sum(1 for r in results if r["vector_rank"]>0 and (r["lexical_rank"]==-1 or r["vector_rank"]<r["lexical_rank"]))
    lw = sum(1 for r in results if r["lexical_rank"]>0 and (r["vector_rank"]==-1 or r["lexical_rank"]<r["vector_rank"]))
    avg_embed = sum(r["embed_ms"] for r in results) / N
    avg_vec = sum(r["vec_ms"] for r in results) / N
    avg_lex = sum(r["lex_ms"] for r in results) / N
    avg_top_sim = sum(r["top_sim"] for r in results) / N
    avg_exp_sim = sum(r["expected_sim"] for r in results if r["expected_sim"]>0) / max(1, sum(1 for r in results if r["expected_sim"]>0))
    
    recs = []
    if miss/N > 0.3: recs.append(f"HIGH: {miss}/{N} missed. Re-embed all with consistent model.")
    if t1/N < 0.3: recs.append(f"MED: P@1={t1}/{N}. Tune reranking weights.")
    if lw > vw*1.5: recs.append(f"INFO: Lexical wins {lw} vs vector {vw}. Increase lexical weight to 0.5")
    if t5/N > 0.6 and t1/N < 0.4: recs.append(f"TUNE: Content in top-5 but not top-1. Focus on reranking.")
    
    summary = {
        "total": N, "P@1": round(t1/N,3), "P@5": round(t5/N,3), "P@10": round(t10/N,3),
        "P@20": round(t20/N,3), "miss_rate": round(miss/N,3),
        "vector_wins": vw, "lexical_wins": lw,
        "avg_embed_ms": round(avg_embed), "avg_vec_ms": round(avg_vec), "avg_lex_ms": round(avg_lex),
        "avg_top_sim": round(avg_top_sim,4), "avg_expected_sim": round(avg_exp_sim,4),
        "recommendations": recs,
        "failed_samples": [{"q":r["query"],"title":r["source_title"]} for r in results if r["missed"]][:10],
    }
    
    with open(SUMMARY_FILE, "w") as f:
        json.dump(summary, f, indent=2)
    
    log.info("\n" + "="*50)
    log.info("RESULTS")
    log.info("="*50)
    log.info(f"Queries: {N}")
    log.info(f"P@1:  {t1}/{N} ({100*t1/N:.1f}%)")
    log.info(f"P@5:  {t5}/{N} ({100*t5/N:.1f}%)")
    log.info(f"P@10: {t10}/{N} ({100*t10/N:.1f}%)")
    log.info(f"P@20: {t20}/{N} ({100*t20/N:.1f}%)")
    log.info(f"Miss: {miss}/{N} ({100*miss/N:.1f}%)")
    log.info(f"Vector wins: {vw}, Lexical wins: {lw}")
    log.info(f"Latency — embed: {avg_embed:.0f}ms, vector: {avg_vec:.0f}ms, lexical: {avg_lex:.0f}ms")
    log.info(f"Similarity — top: {avg_top_sim:.4f}, expected: {avg_exp_sim:.4f}")
    for rec in recs: log.info(f"  → {rec}")

if __name__ == "__main__":
    main()
