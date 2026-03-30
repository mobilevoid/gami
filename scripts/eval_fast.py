#!/usr/bin/env python3
"""Fast synthetic eval — generates queries from segment text directly, no LLM needed."""
import os, sys, json, time, logging, re, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api.config import settings
from api.llm.embeddings import embed_text_sync
from sqlalchemy import create_engine, text as sql_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("eval")

def extract_query_from_text(text, title):
    """Generate queries from text using keyword extraction — no LLM needed."""
    queries = []
    
    # Strategy 1: Use the title/heading as a query
    if title and len(title) > 5:
        queries.append(f"What is {title}?")
    
    # Strategy 2: Extract key phrases (IPs, hostnames, CTs)
    ips = re.findall(r'\d+\.\d+\.\d+\.\d+', text)
    cts = re.findall(r'CT\d{3}', text)
    hosts = re.findall(r'(?:Walter|Stargate|Proxmox|pfSense|GitLab|Authentik|Wazuh|Grafana|Loki|NPM|Ollama|vLLM)', text, re.IGNORECASE)
    creds = re.findall(r'(?:password|pass|token|key|secret)\s*[:=]\s*\S+', text, re.IGNORECASE)
    
    if ips:
        queries.append(f"What is at {ips[0]}?")
    if cts:
        queries.append(f"What runs on {cts[0]}?")
    if hosts:
        h = hosts[0]
        queries.append(f"Tell me about {h}")
        queries.append(f"What is the IP of {h}?")
    if creds:
        queries.append("What are the credentials?")
    
    # Strategy 3: First sentence as query
    first_line = text.strip().split('\n')[0][:200]
    if len(first_line) > 20:
        queries.append(first_line)
    
    # Strategy 4: Random 3-5 word phrase from the text
    words = [w for w in text.split() if len(w) > 3 and not w.startswith('http')]
    if len(words) > 10:
        start = random.randint(0, len(words)-5)
        queries.append(' '.join(words[start:start+4]))
    
    return list(set(queries))[:4]

def main():
    engine = create_engine(settings.DATABASE_URL_SYNC)
    
    log.info("GAMI Fast Evaluation (no LLM needed)")
    
    # Get segments
    with engine.connect() as conn:
        segs = conn.execute(sql_text("""
            SELECT s.segment_id, s.text, s.source_id, s.owner_tenant_id, src.title
            FROM segments s JOIN sources src ON s.source_id = src.source_id
            WHERE s.embedding IS NOT NULL AND length(s.text) BETWEEN 100 AND 3000
            AND s.segment_type NOT IN ('tool_call','tool_result','chunk')
            ORDER BY CASE WHEN src.title ILIKE '%%MEMORY%%' THEN 0 WHEN src.title ILIKE '%%network%%' THEN 1
                WHEN src.title ILIKE '%%AI-STACK%%' THEN 2 ELSE 3 END, length(s.text) DESC
            LIMIT 60
        """)).fetchall()
    log.info(f"Selected {len(segs)} segments")
    
    # Generate queries
    all_q = []
    for sid, stxt, srcid, tid, title in segs:
        for q in extract_query_from_text(stxt, title):
            all_q.append({"query": q, "seg_id": sid, "src_id": srcid, "tenant": tid, "title": title})
        if len(all_q) >= 200:
            break
    log.info(f"Generated {len(all_q)} queries (no LLM)")
    
    # Evaluate
    results = []
    with open("/opt/gami/data/eval_fast_results.jsonl", "w") as f:
        for i, qi in enumerate(all_q):
            t0 = time.time()
            qemb = embed_text_sync(qi["query"])
            embed_ms = int((time.time()-t0)*1000)
            vec = "[" + ",".join(str(v) for v in qemb) + "]"
            
            with engine.connect() as conn:
                # Vector search
                t0 = time.time()
                vrows = conn.execute(sql_text("""
                    SELECT segment_id, 1-(embedding <=> CAST(:v AS vector)) as sim, source_id
                    FROM segments WHERE embedding IS NOT NULL AND owner_tenant_id = ANY(:t)
                    ORDER BY embedding <=> CAST(:v AS vector) LIMIT 20
                """), {"v": vec, "t": [qi["tenant"],"shared"], "l": 20}).fetchall()
                vec_ms = int((time.time()-t0)*1000)
                
                # Lexical
                t0 = time.time()
                lrows = conn.execute(sql_text("""
                    SELECT segment_id, ts_rank(lexical_tsv, plainto_tsquery('english', :q)) as rank, source_id
                    FROM segments WHERE lexical_tsv @@ plainto_tsquery('english', :q) AND owner_tenant_id = ANY(:t)
                    ORDER BY rank DESC LIMIT 20
                """), {"q": qi["query"], "t": [qi["tenant"],"shared"]}).fetchall()
                lex_ms = int((time.time()-t0)*1000)
            
            v_rank = next((i+1 for i,r in enumerate(vrows) if r[0]==qi["seg_id"] or r[2]==qi["src_id"]), -1)
            l_rank = next((i+1 for i,r in enumerate(lrows) if r[0]==qi["seg_id"] or r[2]==qi["src_id"]), -1)
            best = min(r for r in [v_rank, l_rank] if r>0) if any(r>0 for r in [v_rank,l_rank]) else -1
            
            r = {
                "query": qi["query"][:100], "seg_id": qi["seg_id"], "title": qi["title"],
                "v_rank": v_rank, "l_rank": l_rank, "best": best,
                "embed_ms": embed_ms, "vec_ms": vec_ms, "lex_ms": lex_ms,
                "top_sim": round(float(vrows[0][1]),4) if vrows else 0,
                "exp_sim": round(float(next((r[1] for r in vrows if r[0]==qi["seg_id"]),0)),4),
                "top1": best==1, "top5": 0<best<=5, "top10": 0<best<=10, "top20": 0<best<=20, "miss": best==-1,
            }
            results.append(r)
            f.write(json.dumps(r)+"\n")
            
            if (i+1) % 25 == 0:
                t1=sum(1 for x in results if x["top1"]); t5=sum(1 for x in results if x["top5"])
                miss=sum(1 for x in results if x["miss"])
                log.info(f"  {i+1}/{len(all_q)}: P@1={t1}/{i+1}({100*t1/(i+1):.0f}%) P@5={t5}/{i+1}({100*t5/(i+1):.0f}%) miss={miss}/{i+1}({100*miss/(i+1):.0f}%)")
    
    # Summary
    N = len(results)
    t1=sum(1 for r in results if r["top1"]); t5=sum(1 for r in results if r["top5"])
    t10=sum(1 for r in results if r["top10"]); t20=sum(1 for r in results if r["top20"])
    miss=sum(1 for r in results if r["miss"])
    vw=sum(1 for r in results if r["v_rank"]>0 and (r["l_rank"]==-1 or r["v_rank"]<r["l_rank"]))
    lw=sum(1 for r in results if r["l_rank"]>0 and (r["v_rank"]==-1 or r["l_rank"]<r["v_rank"]))
    
    summary = {"total":N, "P@1":round(t1/N,3), "P@5":round(t5/N,3), "P@10":round(t10/N,3),
        "P@20":round(t20/N,3), "miss":round(miss/N,3), "vec_wins":vw, "lex_wins":lw,
        "avg_embed_ms":round(sum(r["embed_ms"] for r in results)/N),
        "avg_vec_ms":round(sum(r["vec_ms"] for r in results)/N),
        "avg_lex_ms":round(sum(r["lex_ms"] for r in results)/N),
        "avg_top_sim":round(sum(r["top_sim"] for r in results)/N,4),
        "avg_exp_sim":round(sum(r["exp_sim"] for r in results if r["exp_sim"]>0)/max(1,sum(1 for r in results if r["exp_sim"]>0)),4),
    }
    recs = []
    if miss/N>0.3: recs.append(f"HIGH: {miss}/{N} missed. Re-embed with consistent model.")
    if t1/N<0.3: recs.append(f"MED: P@1 low ({t1}/{N}). Tune reranking.")
    if lw>vw*1.5: recs.append(f"INFO: Lexical wins {lw} vs vector {vw}. Increase lexical_weight.")
    if t5/N>0.5 and t1/N<0.3: recs.append(f"TUNE: In top-5 but not top-1. Reranking issue.")
    summary["recommendations"] = recs
    summary["failed"] = [{"q":r["query"],"t":r["title"]} for r in results if r["miss"]][:10]
    
    with open("/opt/gami/data/eval_fast_summary.json","w") as f:
        json.dump(summary, f, indent=2)
    
    log.info("\n"+"="*50)
    log.info(f"RESULTS ({N} queries)")
    log.info(f"P@1:  {t1}/{N} ({100*t1/N:.1f}%)")
    log.info(f"P@5:  {t5}/{N} ({100*t5/N:.1f}%)")
    log.info(f"P@10: {t10}/{N} ({100*t10/N:.1f}%)")
    log.info(f"P@20: {t20}/{N} ({100*t20/N:.1f}%)")
    log.info(f"Miss: {miss}/{N} ({100*miss/N:.1f}%)")
    log.info(f"Vec wins: {vw}, Lex wins: {lw}")
    log.info(f"Avg latency — embed:{summary['avg_embed_ms']}ms vec:{summary['avg_vec_ms']}ms lex:{summary['avg_lex_ms']}ms")
    for rec in recs: log.info(f"  → {rec}")

if __name__ == "__main__":
    main()
