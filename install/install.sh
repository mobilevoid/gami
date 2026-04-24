#!/usr/bin/env bash
# GAMI Installer — sets up the full memory system on a fresh machine.
# Supports: Ubuntu/Debian, macOS (with Homebrew)
# Run as: bash /opt/gami/install/install.sh
set -euo pipefail

GAMI_DIR="${GAMI_DIR:-/opt/gami}"
GAMI_PORT="${GAMI_PORT:-9090}"
PG_PORT="${PG_PORT:-5433}"
REDIS_PORT="${REDIS_PORT:-6380}"
OLLAMA_PORT="${OLLAMA_PORT:-11434}"
DB_NAME="gami"
DB_USER="gami"
DB_PASS="${GAMI_DB_PASS:-$(openssl rand -hex 16 2>/dev/null || echo 'gami_default_pass_change_me')}"

log() { echo -e "\033[1;34m[GAMI]\033[0m $*"; }
err() { echo -e "\033[1;31m[ERROR]\033[0m $*" >&2; }
ok()  { echo -e "\033[1;32m[OK]\033[0m $*"; }

# ---------------------------------------------------------------
# Detect OS
# ---------------------------------------------------------------
OS="unknown"
if [ -f /etc/debian_version ]; then
    OS="debian"
elif [ -f /etc/redhat-release ]; then
    OS="rhel"
elif [[ "$(uname)" == "Darwin" ]]; then
    OS="macos"
fi
log "Detected OS: $OS"

# ---------------------------------------------------------------
# Step 1: Install PostgreSQL 16 + pgvector
# ---------------------------------------------------------------
install_postgres() {
    log "Installing PostgreSQL 16..."
    if command -v psql &>/dev/null && psql --version | grep -q "16"; then
        ok "PostgreSQL 16 already installed"
    else
        case $OS in
            debian)
                sudo apt-get update -qq
                sudo apt-get install -y -qq postgresql-16 postgresql-16-pgvector postgresql-client-16 2>/dev/null || {
                    # Add PG repo if not available
                    sudo sh -c 'echo "deb http://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" > /etc/apt/sources.list.d/pgdg.list'
                    wget -qO- https://www.postgresql.org/media/keys/ACCC4CF8.asc | sudo apt-key add -
                    sudo apt-get update -qq
                    sudo apt-get install -y -qq postgresql-16 postgresql-16-pgvector postgresql-client-16
                }
                ;;
            macos)
                brew install postgresql@16 pgvector
                brew services start postgresql@16
                ;;
            *)
                err "Unsupported OS for auto-install. Install PostgreSQL 16 + pgvector manually."
                return 1
                ;;
        esac
    fi

    # Configure to run on custom port
    local PG_CONF
    PG_CONF=$(find /etc/postgresql/16 -name postgresql.conf 2>/dev/null | head -1)
    if [ -n "$PG_CONF" ] && ! grep -q "port = $PG_PORT" "$PG_CONF"; then
        log "Configuring PostgreSQL on port $PG_PORT..."
        sudo sed -i "s/^port = .*/port = $PG_PORT/" "$PG_CONF" 2>/dev/null || true
        sudo systemctl restart postgresql@16-main 2>/dev/null || sudo systemctl restart postgresql 2>/dev/null || true
    fi

    # Create database and user
    log "Creating database and user..."
    sudo -u postgres psql -p "$PG_PORT" -c "CREATE USER $DB_USER WITH PASSWORD '$DB_PASS';" 2>/dev/null || true
    sudo -u postgres psql -p "$PG_PORT" -c "CREATE DATABASE $DB_NAME OWNER $DB_USER;" 2>/dev/null || true
    sudo -u postgres psql -p "$PG_PORT" -d "$DB_NAME" -c "CREATE EXTENSION IF NOT EXISTS vector;" 2>/dev/null || true
    sudo -u postgres psql -p "$PG_PORT" -d "$DB_NAME" -c "CREATE EXTENSION IF NOT EXISTS pg_trgm;" 2>/dev/null || true
    sudo -u postgres psql -p "$PG_PORT" -d "$DB_NAME" -c "GRANT ALL ON SCHEMA public TO $DB_USER;" 2>/dev/null || true

    ok "PostgreSQL 16 ready on port $PG_PORT"
}

# ---------------------------------------------------------------
# Step 2: Install Redis
# ---------------------------------------------------------------
install_redis() {
    log "Installing Redis..."
    if command -v redis-server &>/dev/null; then
        ok "Redis already installed"
    else
        case $OS in
            debian) sudo apt-get install -y -qq redis-server ;;
            macos) brew install redis && brew services start redis ;;
            *) err "Install Redis manually"; return 1 ;;
        esac
    fi

    # Start on custom port if not already running
    if ! redis-cli -p "$REDIS_PORT" ping &>/dev/null; then
        log "Starting Redis on port $REDIS_PORT..."
        redis-server --port "$REDIS_PORT" --daemonize yes --maxmemory 512mb --maxmemory-policy allkeys-lru 2>/dev/null || true
    fi
    ok "Redis ready on port $REDIS_PORT"
}

# ---------------------------------------------------------------
# Step 3: Install Ollama + embedding model
# ---------------------------------------------------------------
install_ollama() {
    log "Installing Ollama..."
    if command -v ollama &>/dev/null; then
        ok "Ollama already installed"
    else
        curl -fsSL https://ollama.com/install.sh | sh
    fi

    # Start Ollama if not running
    if ! curl -sf "http://localhost:$OLLAMA_PORT/api/tags" &>/dev/null; then
        log "Starting Ollama..."
        nohup ollama serve > /tmp/ollama.log 2>&1 &
        sleep 5
    fi

    # Pull embedding model
    if ! ollama list 2>/dev/null | grep -q nomic-embed-text; then
        log "Pulling nomic-embed-text (274MB, CPU)..."
        ollama pull nomic-embed-text
    fi
    ok "Ollama ready with nomic-embed-text"
}

# ---------------------------------------------------------------
# Step 4: Install Python dependencies
# ---------------------------------------------------------------
install_python() {
    log "Installing Python dependencies..."
    if [ -f "$GAMI_DIR/pyproject.toml" ]; then
        cd "$GAMI_DIR"
        pip install -q -e ".[all]" 2>/dev/null || pip install -q -r requirements.txt 2>/dev/null || {
            # Minimal install
            pip install -q fastapi uvicorn sqlalchemy asyncpg psycopg2-binary pgvector \
                redis httpx tiktoken jinja2 python-multipart alembic mcp requests
        }
    else
        pip install -q fastapi uvicorn sqlalchemy asyncpg psycopg2-binary pgvector \
            redis httpx tiktoken jinja2 python-multipart alembic mcp requests
    fi
    ok "Python dependencies installed"
}

# ---------------------------------------------------------------
# Step 5: Create .env file
# ---------------------------------------------------------------
create_env() {
    local ENV_FILE="$GAMI_DIR/.env"
    if [ -f "$ENV_FILE" ]; then
        ok ".env already exists"
        return
    fi

    log "Creating .env file..."
    cat > "$ENV_FILE" << ENVEOF
DATABASE_URL=postgresql+asyncpg://${DB_USER}:${DB_PASS}@localhost:${PG_PORT}/${DB_NAME}
DATABASE_URL_SYNC=postgresql://${DB_USER}:${DB_PASS}@localhost:${PG_PORT}/${DB_NAME}
REDIS_URL=redis://localhost:${REDIS_PORT}/0
OLLAMA_URL=http://localhost:${OLLAMA_PORT}
VLLM_URL=http://localhost:8000/v1
EMBEDDING_MODEL=nomic-embed-text
GAMI_API_PORT=${GAMI_PORT}
ENVEOF
    chmod 600 "$ENV_FILE"
    ok ".env created"
}

# ---------------------------------------------------------------
# Step 6: Run database migrations
# ---------------------------------------------------------------
run_migrations() {
    log "Running database migrations..."
    cd "$GAMI_DIR"
    if [ -d "storage/sql/migrations" ]; then
        PYTHONPATH="$GAMI_DIR" python3 -m alembic upgrade head 2>/dev/null || {
            log "Alembic migration failed, trying direct schema creation..."
            PGPASSWORD="$DB_PASS" psql -h localhost -p "$PG_PORT" -U "$DB_USER" -d "$DB_NAME" \
                -f "$GAMI_DIR/install/schema.sql" 2>/dev/null || true
        }
    fi

    # Ensure core tables exist (idempotent)
    PYTHONPATH="$GAMI_DIR" python3 -c "
from api.services.db import engine_sync
from sqlalchemy import text
with engine_sync.connect() as conn:
    # Check if segments table exists
    result = conn.execute(text(\"SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name='segments')\"))
    if result.scalar():
        print('Schema OK — tables exist')
    else:
        print('WARNING: Tables not found. Run migrations manually.')
" 2>/dev/null || log "Could not verify schema — check database connection"

    # Seed default tenants
    PGPASSWORD="$DB_PASS" psql -h localhost -p "$PG_PORT" -U "$DB_USER" -d "$DB_NAME" -c "
        INSERT INTO tenants (tenant_id, display_name, status) VALUES
            ('default', 'Claude Opus', 'active'),
            ('shared', 'Shared', 'active'),
            ('background-worker', 'Background Worker', 'active')
        ON CONFLICT (tenant_id) DO NOTHING;
    " 2>/dev/null || true

    ok "Database ready"
}

# ---------------------------------------------------------------
# Step 7: Start GAMI API
# ---------------------------------------------------------------
start_api() {
    if curl -sf "http://localhost:$GAMI_PORT/health" &>/dev/null; then
        ok "GAMI API already running on port $GAMI_PORT"
        return
    fi

    log "Starting GAMI API on port $GAMI_PORT..."
    cd "$GAMI_DIR"
    PYTHONPATH="$GAMI_DIR" nohup python3 -m uvicorn api.main:app \
        --host 0.0.0.0 --port "$GAMI_PORT" \
        > /tmp/gami-api.log 2>&1 &

    # Wait for startup
    for i in $(seq 1 15); do
        if curl -sf "http://localhost:$GAMI_PORT/health" &>/dev/null; then
            ok "GAMI API running on port $GAMI_PORT"
            return
        fi
        sleep 2
    done
    err "GAMI API failed to start. Check /tmp/gami-api.log"
}

# ---------------------------------------------------------------
# Step 8: Configure Claude Code
# ---------------------------------------------------------------
configure_claude() {
    log "Configuring Claude Code MCP + hooks..."

    local PYTHON_PATH
    PYTHON_PATH=$(which python3)

    # Add MCP server config
    local SETTINGS="$HOME/.claude/settings.json"
    mkdir -p "$(dirname "$SETTINGS")"

    if [ -f "$SETTINGS" ]; then
        python3 -c "
import json
with open('$SETTINGS', 'r') as f:
    d = json.load(f)

# Add MCP server
d.setdefault('mcpServers', {})
d['mcpServers']['gami'] = {
    'command': '$PYTHON_PATH',
    'args': ['-m', 'mcp_tools.server'],
    'cwd': '$GAMI_DIR',
    'env': {'PYTHONPATH': '$GAMI_DIR'}
}

# Add hooks
d.setdefault('hooks', {})
hook_entry = [{'hooks': [{'type': 'command', 'command': '$GAMI_DIR/cli/journal_hook.sh', 'timeout': 30, 'async': True}]}]
for event in ['PreCompact', 'SessionEnd', 'PreClear']:
    if event not in d['hooks']:
        d['hooks'][event] = hook_entry

with open('$SETTINGS', 'w') as f:
    json.dump(d, f, indent=2)
print('Updated $SETTINGS')
" 2>/dev/null || log "Could not update settings.json — configure manually"
    else
        cat > "$SETTINGS" << SETEOF
{
  "mcpServers": {
    "gami": {
      "command": "$PYTHON_PATH",
      "args": ["-m", "mcp_tools.server"],
      "cwd": "$GAMI_DIR",
      "env": {"PYTHONPATH": "$GAMI_DIR"}
    }
  },
  "hooks": {
    "PreCompact": [{"hooks": [{"type": "command", "command": "$GAMI_DIR/cli/journal_hook.sh", "timeout": 30, "async": true}]}],
    "SessionEnd": [{"hooks": [{"type": "command", "command": "$GAMI_DIR/cli/journal_hook.sh", "timeout": 30, "async": true}]}],
    "PreClear": [{"hooks": [{"type": "command", "command": "$GAMI_DIR/cli/journal_hook.sh", "timeout": 30, "async": true}]}]
  }
}
SETEOF
    fi

    ok "Claude Code configured with GAMI MCP + session hooks"
}

# ---------------------------------------------------------------
# Health check
# ---------------------------------------------------------------
health_check() {
    echo ""
    log "Running health checks..."
    local PASS=0 FAIL=0

    # PostgreSQL
    if PGPASSWORD="$DB_PASS" psql -h localhost -p "$PG_PORT" -U "$DB_USER" -d "$DB_NAME" -c "SELECT 1" &>/dev/null; then
        ok "PostgreSQL: connected"; ((PASS++))
    else
        err "PostgreSQL: connection failed"; ((FAIL++))
    fi

    # Redis
    if redis-cli -p "$REDIS_PORT" ping 2>/dev/null | grep -q PONG; then
        ok "Redis: connected"; ((PASS++))
    else
        err "Redis: connection failed"; ((FAIL++))
    fi

    # Ollama
    if curl -sf "http://localhost:$OLLAMA_PORT/api/tags" &>/dev/null; then
        ok "Ollama: running"; ((PASS++))
    else
        err "Ollama: not responding"; ((FAIL++))
    fi

    # Embedding test
    if curl -sf "http://localhost:$OLLAMA_PORT/api/embeddings" -d '{"model":"nomic-embed-text","prompt":"test"}' 2>/dev/null | grep -q embedding; then
        ok "Embeddings: working"; ((PASS++))
    else
        err "Embeddings: nomic-embed-text not available"; ((FAIL++))
    fi

    # GAMI API
    if curl -sf "http://localhost:$GAMI_PORT/health" &>/dev/null; then
        ok "GAMI API: healthy"; ((PASS++))
    else
        err "GAMI API: not responding"; ((FAIL++))
    fi

    echo ""
    if [ $FAIL -eq 0 ]; then
        ok "All $PASS checks passed! GAMI is ready."
        echo ""
        echo "  Next steps:"
        echo "  1. Restart Claude Code to load MCP tools"
        echo "  2. Try: \"Recall what you know about this system\""
        echo "  3. Try: \"Ingest file /path/to/some/document.md\""
        echo ""
        echo "  Credentials saved to: $GAMI_DIR/.env"
        echo "  Logs: /tmp/gami-api.log"
    else
        err "$FAIL checks failed. Review errors above."
    fi
}

# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
main() {
    echo ""
    echo "╔═══════════════════════════════════════════════════╗"
    echo "║  GAMI — Graph-Augmented Memory Interface         ║"
    echo "║  Persistent memory system for Claude Code        ║"
    echo "╚═══════════════════════════════════════════════════╝"
    echo ""

    install_postgres
    install_redis
    install_ollama
    install_python
    create_env
    run_migrations
    start_api
    configure_claude
    health_check
}

main "$@"
