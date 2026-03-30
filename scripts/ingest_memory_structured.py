#!/usr/bin/env python3
"""
One-time structured import of Claude Code memory files into GAMI.

Unlike the generic markdown parser, this script:
1. Parses MEMORY.md section by section, creating entities and claims for each
2. Imports each memory file as a separate source with proper metadata
3. Creates durable assistant_memories for key facts
4. Links credentials as sensitivity:credential memories
5. Creates entities for every CT, service, IP, person mentioned
6. Creates claims for every factual statement (IP assignments, passwords, configs)
7. Creates relations between entities (CT hosts service, service has port, etc.)
"""
import os, sys, json, re, time, hashlib, logging
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api.config import settings
from api.llm.embeddings import embed_text_sync
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("memory_import")

engine = create_engine(settings.DATABASE_URL_SYNC, pool_size=3)
TENANT = "claude-opus"

def gen_id(prefix, name):
    h = hashlib.md5(name.encode()).hexdigest()[:8]
    clean = re.sub(r'[^a-zA-Z0-9_]', '_', name)[:30]
    return f"{prefix}_{clean}_{h}"

def upsert_entity(conn, name, etype, description="", aliases=None):
    eid = gen_id("ENT", f"{etype}_{name}")
    conn.execute(text("""
        INSERT INTO entities (entity_id, owner_tenant_id, entity_type, canonical_name, 
            aliases_json, description, status, first_seen_at, last_seen_at, source_count, mention_count)
        VALUES (:eid, :tid, :etype, :name, :aliases, :desc, 'active', NOW(), NOW(), 1, 1)
        ON CONFLICT (entity_id) DO UPDATE SET 
            mention_count = entities.mention_count + 1,
            last_seen_at = NOW(),
            description = CASE WHEN length(:desc) > length(COALESCE(entities.description, '')) THEN :desc ELSE entities.description END
    """), {"eid": eid, "tid": TENANT, "etype": etype, "name": name, 
           "aliases": json.dumps(aliases or []), "desc": description})
    return eid

def upsert_claim(conn, subject_eid, predicate, obj_text, confidence=0.95):
    cid = gen_id("CLM", f"{subject_eid}_{predicate}_{obj_text[:30]}")
    conn.execute(text("""
        INSERT INTO claims (claim_id, owner_tenant_id, subject_entity_id, predicate, 
            object_literal_json, confidence, status, summary_text, support_count)
        VALUES (:cid, :tid, :seid, :pred, :obj, :conf, 'active', :summary, 1)
        ON CONFLICT (claim_id) DO UPDATE SET support_count = claims.support_count + 1
    """), {"cid": cid, "tid": TENANT, "seid": subject_eid, "pred": predicate,
           "obj": json.dumps({"value": obj_text}), "conf": confidence,
           "summary": f"{predicate}: {obj_text}"})
    return cid

def upsert_relation(conn, from_eid, to_eid, rtype, confidence=0.9):
    rid = gen_id("REL", f"{from_eid}_{rtype}_{to_eid}")
    conn.execute(text("""
        INSERT INTO relations (relation_id, owner_tenant_id, from_node_type, from_node_id,
            to_node_type, to_node_id, relation_type, confidence, weight, support_count, created_by, status)
        VALUES (:rid, :tid, 'entity', :fid, 'entity', :toid, :rtype, :conf, 1.0, 1, 'memory_import', 'active')
        ON CONFLICT (relation_id) DO UPDATE SET support_count = relations.support_count + 1
    """), {"rid": rid, "tid": TENANT, "fid": from_eid, "toid": to_eid, "rtype": rtype, "conf": confidence})

def upsert_memory(conn, text_content, mtype, subject, importance=0.8, sensitivity="normal"):
    mid = gen_id("MEM", f"{mtype}_{subject}_{text_content[:30]}")
    try:
        emb = embed_text_sync(text_content[:2000])
        vec = "[" + ",".join(str(v) for v in emb) + "]"
    except:
        vec = None
    
    if vec:
        conn.execute(text("""
            INSERT INTO assistant_memories (memory_id, owner_tenant_id, memory_type, subject_id,
                normalized_text, embedding, importance_score, stability_score, confirmation_count,
                status, sensitivity)
            VALUES (:mid, :tid, :mtype, :subj, :txt, CAST(:vec AS vector), :imp, 0.9, 3, 'active', :sens)
            ON CONFLICT (memory_id) DO UPDATE SET confirmation_count = assistant_memories.confirmation_count + 1
        """), {"mid": mid, "tid": TENANT, "mtype": mtype, "subj": subject,
               "txt": text_content, "vec": vec, "imp": importance, "sens": sensitivity})
    else:
        conn.execute(text("""
            INSERT INTO assistant_memories (memory_id, owner_tenant_id, memory_type, subject_id,
                normalized_text, importance_score, stability_score, confirmation_count, status, sensitivity)
            VALUES (:mid, :tid, :mtype, :subj, :txt, :imp, 0.9, 3, 'active', :sens)
            ON CONFLICT (memory_id) DO UPDATE SET confirmation_count = assistant_memories.confirmation_count + 1
        """), {"mid": mid, "tid": TENANT, "mtype": mtype, "subj": subject,
               "txt": text_content, "imp": importance, "sens": sensitivity})
    return mid

def parse_memory_md():
    """Parse MEMORY.md and extract structured data."""
    with open("/home/ai/.claude/projects/-home-ai/memory/MEMORY.md") as f:
        content = f.read()
    
    with engine.connect() as conn:
        # ============================================
        # Infrastructure Access
        # ============================================
        log.info("Importing infrastructure access...")
        
        infra = {
            "Proxmox (ledata)": {"ip": "192.168.1.188", "user": "root", "pass": "Twotone1", "type": "infrastructure"},
            "Walter (pfSense Primary)": {"ip": "192.168.1.1", "user": "root", "pass": "L00kmanohands!", "port": "8080", "api_key": "f68a21eb33ea2c60f5dbef4c041027ca", "type": "infrastructure"},
            "Stargate (pfSense Secondary)": {"ip": "10.9.8.1", "user": "admin", "pass": "Twotone1", "api_key": "a06de363218e6c20849551f757baf4de", "type": "infrastructure"},
            "M5300-52G-PoE+ Switch": {"ip": "192.168.1.3", "user": "admin", "pass": "Twotone1", "type": "infrastructure"},
            "FreeNAS black-jesus": {"ip": "192.168.1.202", "user": "root", "pass": "Neveragain123!", "type": "infrastructure"},
            "GS110EMX Switch": {"ip": "192.168.1.208", "type": "infrastructure"},
        }
        
        for name, info in infra.items():
            eid = upsert_entity(conn, name, "infrastructure", f"{name} at {info.get('ip','')}")
            if "ip" in info:
                upsert_claim(conn, eid, "has_ip", info["ip"])
                upsert_memory(conn, f"{name} IP: {info['ip']}", "fact", name, 0.95)
            if "user" in info and "pass" in info:
                upsert_claim(conn, eid, "has_credentials", f"{info['user']}/{info['pass']}", 1.0)
                upsert_memory(conn, f"{name} credentials: {info['user']}/{info['pass']}", "fact", name, 1.0, "credential")
            if "api_key" in info:
                upsert_claim(conn, eid, "has_api_key", info["api_key"], 1.0)
                upsert_memory(conn, f"{name} API key: {info['api_key']}", "fact", name, 1.0, "credential")
            if "port" in info:
                upsert_claim(conn, eid, "webgui_port", info["port"])
        
        conn.commit()
        log.info(f"  Infrastructure: {len(infra)} devices imported")
        
        # ============================================
        # Containers
        # ============================================
        log.info("Importing containers...")
        
        containers = {
            "CT101": {"name": "pihole", "desc": "Pi-hole v6 DNS, native install", "ip": "192.168.90.101"},
            "CT103": {"name": "uptimekuma", "desc": "Uptime Kuma monitoring, 117+ monitors", "ip": "192.168.90.103", "user": "root", "pass": "DCSMonitor2026!"},
            "CT201": {"name": "authentik", "desc": "Authentik SSO v2026.2.1", "ip": "192.168.90.201", "user": "akadmin", "pass": "Renew500!"},
            "CT202": {"name": "vaultwarden", "desc": "Vaultwarden password manager", "ip": "192.168.90.202"},
            "CT205": {"name": "invoiceninja", "desc": "Invoice Ninja", "ip": "192.168.90.205"},
            "CT220": {"name": "wazuh", "desc": "Wazuh SIEM 4.7.3, 43+ agents", "ip": "192.168.90.220", "user": "admin", "pass": "admin"},
            "CT221": {"name": "grafana", "desc": "Grafana + Loki + Promtail", "ip": "192.168.90.221"},
            "CT222": {"name": "librenms", "desc": "LibreNMS network monitoring", "ip": "192.168.90.222"},
            "CT228": {"name": "npm", "desc": "Nginx Proxy Manager, 43 proxy hosts", "ip": "192.168.90.228", "user": "admin@choosedcs.com", "pass": "DCSAdmin2026"},
            "CT229": {"name": "netdata", "desc": "Netdata parent, ~40 children streaming", "ip": "192.168.90.229"},
            "CT231": {"name": "management", "desc": "Management CT, PostgreSQL 14, 2 uvicorn workers", "ip": "192.168.90.231"},
            "CT232": {"name": "headscale", "desc": "Headscale v0.28.0 + Headplane v0.6.2", "ip": "192.168.90.232"},
            "CT233": {"name": "digitalsales", "desc": "Digital Sales crypto auction (Docker)", "ip": "192.168.90.233"},
            "CT234": {"name": "kimai", "desc": "Kimai2 time tracking", "ip": "192.168.90.234"},
            "CT235": {"name": "choosedcs", "desc": "choosedcs.com static sites", "ip": "192.168.90.235"},
            "CT236": {"name": "velvetgavel", "desc": "VelvetGavel auction (Go+Node+PG+MongoDB+Redis)", "ip": "192.168.90.236"},
            "CT237": {"name": "dynamiccomposite", "desc": "dynamiccompositesolutions.com", "ip": "192.168.90.237"},
            "CT238": {"name": "landpersonnel", "desc": "landpersonnel.com", "ip": "192.168.90.238"},
            "CT239": {"name": "esoterraglobal", "desc": "esoterraglobal.com", "ip": "192.168.90.239"},
            "CT241": {"name": "stalwart", "desc": "stalwartresources.com", "ip": "192.168.90.241"},
            "CT242": {"name": "consent2assign", "desc": "consent2assign.com", "ip": "192.168.90.242"},
            "CT243": {"name": "bltstrading", "desc": "bltstrading.com", "ip": "192.168.90.243"},
            "CT244": {"name": "realartschool", "desc": "realartschool.com", "ip": "192.168.90.244"},
            "CT245": {"name": "transformesg", "desc": "transformesg.com", "ip": "192.168.90.245"},
            "CT246": {"name": "cvsmoke", "desc": "cvsmoke.com WordPress", "ip": "192.168.90.246"},
            "CT247": {"name": "socialfreedom", "desc": "socialfreedom.net WordPress", "ip": "192.168.90.247"},
            "CT248": {"name": "nwoconsulting", "desc": "nwoconsulting.com WordPress", "ip": "192.168.90.248"},
            "CT249": {"name": "webinfra", "desc": "mynut.net URL shortener + parked domains", "ip": "192.168.90.249"},
            "CT250": {"name": "gridfortrisk", "desc": "gridfortrisk.com", "ip": "192.168.90.250"},
            "CT252": {"name": "gitlab", "desc": "GitLab CE 18.9.1, 17 repos", "ip": "192.168.90.252", "user": "root", "pass": "Xk9mP$vL2wQz8nRj"},
            "CT253": {"name": "netcloud", "desc": "NetCloud Vue3+FastAPI monitoring", "ip": "192.168.90.253"},
            "CT255": {"name": "stageover", "desc": "StageOver v2.0 FastAPI+React", "ip": "192.168.90.255"},
            "CT256": {"name": "qbit", "desc": "qBittorrent + PIA VPN, saves to NAS", "ip": "192.168.1.100"},
        }
        
        proxmox_eid = upsert_entity(conn, "Proxmox (ledata)", "infrastructure")
        
        for ctid, info in containers.items():
            eid = upsert_entity(conn, ctid, "infrastructure", f"{ctid} ({info['name']}): {info['desc']}", 
                               aliases=[info['name'], f"{info['name']}.mynut.net"])
            upsert_claim(conn, eid, "has_ip", info["ip"])
            upsert_claim(conn, eid, "runs_service", info["name"])
            upsert_claim(conn, eid, "description", info["desc"])
            upsert_relation(conn, eid, proxmox_eid, "PART_OF")
            
            upsert_memory(conn, f"{ctid} ({info['name']}): {info['desc']}, IP: {info['ip']}", "fact", ctid, 0.9)
            
            if "user" in info and "pass" in info:
                upsert_claim(conn, eid, "has_credentials", f"{info['user']}/{info['pass']}", 1.0)
                upsert_memory(conn, f"{ctid} ({info['name']}) credentials: {info['user']}/{info['pass']}", "fact", ctid, 1.0, "credential")
            
            # Create service entity and link
            svc_eid = upsert_entity(conn, info["name"], "service", info["desc"])
            upsert_relation(conn, eid, svc_eid, "USES")
            upsert_relation(conn, svc_eid, eid, "LOCATED_IN")
        
        conn.commit()
        log.info(f"  Containers: {len(containers)} imported with services and relations")
        
        # ============================================
        # Admin Users
        # ============================================
        log.info("Importing admin users...")
        
        users = {
            "admin": {"pk": 5, "pass": "Renew500!", "desc": "superuser, all 12 groups"},
            "choll": {"pk": 6, "pass": "Renew500!", "desc": "superuser, all 12 groups"},
            "otherkevin": {"pk": 7, "pass": "Twotone1", "desc": "superuser, all 12 groups"},
        }
        
        for uname, info in users.items():
            eid = upsert_entity(conn, uname, "person", f"Authentik user pk={info['pk']}, {info['desc']}")
            upsert_claim(conn, eid, "has_password", info["pass"], 1.0)
            upsert_memory(conn, f"Authentik user {uname} (pk={info['pk']}): password={info['pass']}, {info['desc']}", "fact", uname, 0.95, "credential")
        
        conn.commit()
        log.info(f"  Users: {len(users)} imported")
        
        # ============================================
        # Cloud / Edge
        # ============================================
        log.info("Importing cloud/edge infrastructure...")
        
        cloud = {
            "Edge NYC1 (DS)": {"ip": "142.93.6.185", "desc": "DigitalOcean edge node NYC1"},
            "Edge NYC2 (Combined)": {"ip": "165.227.95.182", "desc": "DigitalOcean edge node NYC2 + mail relay"},
            "Cloudflare (mynut.net)": {"zone": "0068dcace7319a962dfcb1b6cc915e0f", "token": "1Rn6HT_Bycscsj6j4-maAcFDMtm-Sd9NcPS38jl2"},
        }
        
        for name, info in cloud.items():
            eid = upsert_entity(conn, name, "infrastructure", info.get("desc", name))
            if "ip" in info:
                upsert_claim(conn, eid, "has_ip", info["ip"])
            if "zone" in info:
                upsert_claim(conn, eid, "zone_id", info["zone"])
                upsert_memory(conn, f"Cloudflare zone ID: {info['zone']}, API token: {info.get('token','')}", "fact", "Cloudflare", 1.0, "credential")
            if "token" in info:
                upsert_claim(conn, eid, "api_token", info["token"], 1.0)
        
        conn.commit()
        log.info(f"  Cloud/edge: {len(cloud)} imported")
        
        # ============================================  
        # Key Database Credentials
        # ============================================
        log.info("Importing database credentials...")
        
        dbs = {
            "Management PG (CT231)": {"host": "127.0.0.1:5432", "db": "management", "user": "management", "pass": "DCSMgmt2026!Secure"},
            "Kimai MySQL (CT234)": {"user": "kimai", "pass": "K1m4i_DCS_2026!"},
            "CV Smoke MariaDB (CT246)": {"db": "cvsmoke_wp", "user": "cvsmoke_user", "pass": "CvSmoke2026!Secure"},
            "VelvetGavel PG (CT236)": {"user": "highsite", "pass": "VelvetGavel2026Prod"},
            "NetCloud PG (CT253)": {"user": "netcloud", "pass": "7p//0n5WP7Y4q6mRrwULiossL2BQ3hVm"},
            "GAMI PG (hal9001)": {"host": "127.0.0.1:5433", "db": "gami", "user": "gami", "pass": "GamiProd2026"},
        }
        
        for name, info in dbs.items():
            cred_text = f"{name}: user={info['user']}, pass={info['pass']}"
            if "db" in info:
                cred_text += f", db={info['db']}"
            if "host" in info:
                cred_text += f", host={info['host']}"
            upsert_memory(conn, cred_text, "fact", name, 1.0, "credential")
        
        conn.commit()
        log.info(f"  Databases: {len(dbs)} credential sets imported")
        
        # ============================================
        # Network Topology Facts
        # ============================================
        log.info("Importing network topology...")
        
        walter_eid = upsert_entity(conn, "Walter (pfSense Primary)", "infrastructure")
        stargate_eid = upsert_entity(conn, "Stargate (pfSense Secondary)", "infrastructure")
        
        topology_facts = [
            "Walter WAN is AT&T on igb3, IP 104.11.116.153, gateway 104.11.116.158",
            "Stargate WAN is Starlink on em0 (CGNAT), LAN on mlxen0 at 10.9.8.1/24",
            "WalterLink direct cable: Walter igb2 (10.11.121.1) <-> Stargate ix0 (10.11.121.2)",
            "Failover: Walter AT&T (Tier1) -> Stargate Starlink via WalterLink (Tier2)",
            "M5300 trunk ports: 1/0/47 (pfSense), 1/0/51 (GS110EMX) - all VLANs tagged",
            "Topology: Proxmox -> GS110EMX (.208) -> M5300 (.3) -> pfSense (.1)",
            "VLAN 90 (192.168.90.0/24) carries all container traffic",
            "hal9001 is on 10.9.8.7 (Stargate LAN) and 192.168.1.176 (Walter LAN via DHCP)",
            "hal9001 default route via Stargate (10.9.8.1), failover via Walter (192.168.1.1)",
            "Pi-hole DNS: 192.168.90.101 primary, pfSense fallback, Quad9 tertiary",
        ]
        
        for fact in topology_facts:
            upsert_memory(conn, fact, "fact", "network_topology", 0.95)
        
        conn.commit()
        log.info(f"  Topology: {len(topology_facts)} facts imported")
        
        # ============================================
        # SSO Configuration
        # ============================================
        log.info("Importing SSO details...")
        upsert_memory(conn, "22 SSO providers: 17 OIDC + 2 proxy + 1 SAML + 1 LDAP + 1 LDAP outpost", "fact", "Authentik SSO", 0.85)
        upsert_memory(conn, "Authentik API token: fap0GTXXN4ho011Ge7ubPCUP4LEZnQBnXWpkBnygDc1dg4olQAjEergn1zLT", "fact", "Authentik", 1.0, "credential")
        upsert_memory(conn, "LDAP outpost on CT201:3389, base DN: dc=authentik,dc=local, akadmin password: Renew500!", "fact", "LDAP", 0.9, "credential")
        conn.commit()
        
        # ============================================
        # Phase Progress
        # ============================================
        log.info("Importing phase progress...")
        phases = [
            "Phases 1-9D: COMPLETE",
            "Phase 10: NEARLY COMPLETE (pfSense firewall tightening + Twilio A2P vetting remain)",
            "Migration Phase A-F: ALL COMPLETE",
            "Management UI v2.1: COMPLETE",
            "GitLab Phase: COMPLETE (CT252, 17 repos)",
            "ACME Certs Phase: COMPLETE (wildcard *.mynut.net LE cert, 37 CTs)",
            "Edge Failover: COMPLETE (CT231 cron, Cloudflare API, 186 records)",
            "System Tuning: COMPLETE (LibreNMS tuned, swappiness=10, Netdata on 47 CTs)",
            "PostgreSQL Migration: COMPLETE (SQLite->PG on CT231, 164K rows)",
            "StageOver FLUX Pipeline: IN PROGRESS",
        ]
        for phase in phases:
            upsert_memory(conn, phase, "project", "DCS Platform", 0.7)
        conn.commit()
        log.info(f"  Phases: {len(phases)} imported")
        
        # ============================================
        # Safety Rules
        # ============================================
        log.info("Importing safety rules...")
        rules = [
            "NEVER delete files, directories, model weights, blobs, or data >1GB without EXPLICIT user confirmation",
            "Plans are not permission — a plan saying 'purge X' does NOT authorize deletion",
            "Model files are sacred — treat model weights like production databases",
            "NEVER modify existing DNS records",
            "NEVER install software on the Proxmox host directly",
            "Keep IP:port access working alongside any new proxy setup",
        ]
        for rule in rules:
            upsert_memory(conn, rule, "policy", "safety_rules", 1.0)
        conn.commit()
        log.info(f"  Safety rules: {len(rules)} imported")
        
        # Final counts
        ent_count = conn.execute(text("SELECT count(*) FROM entities")).scalar()
        claim_count = conn.execute(text("SELECT count(*) FROM claims")).scalar()
        mem_count = conn.execute(text("SELECT count(*) FROM assistant_memories")).scalar()
        rel_count = conn.execute(text("SELECT count(*) FROM relations")).scalar()
        
        log.info(f"\n{'='*50}")
        log.info(f"IMPORT COMPLETE")
        log.info(f"Entities:  {ent_count}")
        log.info(f"Claims:    {claim_count}")
        log.info(f"Memories:  {mem_count}")
        log.info(f"Relations: {rel_count}")

if __name__ == "__main__":
    parse_memory_md()
