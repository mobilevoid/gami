#!/usr/bin/env python3
"""vLLM Conversation Watcher — SSE server that streams conversations to browsers.

Tails /tmp/vllm_conversations.jsonl and pushes events via Server-Sent Events.
All data lives in the browser — server stores nothing beyond the rolling JSONL.

Usage: python3 scripts/vllm_watcher.py [--port 9091]
"""
import asyncio, json, os, argparse

MONITOR_FILE = "/tmp/vllm_conversations.jsonl"
PORT = 9091

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>vLLM Watcher</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #0d1117; color: #c9d1d9; font-family: 'Consolas','Monaco',monospace; font-size: 13px; }
#header { position: fixed; top: 0; left: 0; right: 0; z-index: 10; background: #161b22; border-bottom: 1px solid #30363d; padding: 8px 16px; display: flex; align-items: center; gap: 16px; }
#header h1 { font-size: 15px; color: #58a6ff; font-weight: 600; }
.stat { color: #8b949e; }
.stat b { color: #c9d1d9; }
.filters { display: flex; gap: 6px; }
.filters button { background: #21262d; border: 1px solid #30363d; color: #8b949e; padding: 3px 10px; border-radius: 4px; cursor: pointer; font-size: 12px; }
.filters button.active { background: #1f6feb; border-color: #1f6feb; color: #fff; }
.filters button:hover { border-color: #58a6ff; }
#toggle-scroll { margin-left: auto; }
#toggle-persist { }
#status-dot { width: 8px; height: 8px; border-radius: 50%; background: #f85149; display: inline-block; }
#status-dot.connected { background: #3fb950; }
#cards { padding: 56px 16px 16px; display: flex; flex-direction: column; gap: 8px; }
.card { background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 10px 14px; border-left: 3px solid #3fb950; }
.card.fail { border-left-color: #f85149; }
.card.slow { border-left-color: #d29922; }
.card-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; }
.card-script { color: #58a6ff; font-weight: 600; font-size: 12px; }
.card-meta { color: #8b949e; font-size: 11px; }
.card-prompt { color: #8b949e; margin-bottom: 4px; white-space: pre-wrap; word-break: break-word; max-height: 60px; overflow: hidden; cursor: pointer; transition: max-height 0.2s; }
.card-prompt.expanded { max-height: none; }
.card-response { color: #c9d1d9; white-space: pre-wrap; word-break: break-word; max-height: 200px; overflow: hidden; cursor: pointer; transition: max-height 0.2s; }
.card-response.expanded { max-height: none; }
.card-error { color: #f85149; margin-top: 4px; }
.hidden { display: none; }
</style>
</head>
<body>
<div id="header">
  <span id="status-dot"></span>
  <h1>vLLM Watcher</h1>
  <span class="stat">Total: <b id="stat-total">0</b></span>
  <span class="stat">Rate: <b id="stat-rate">0</b>/min</span>
  <span class="stat">Avg: <b id="stat-latency">0</b>ms</span>
  <div class="filters" id="filters">
    <button class="active" data-filter="all">All</button>
  </div>
  <button class="filters" id="toggle-scroll" onclick="toggleScroll()">Auto-scroll: ON</button>
  <button class="filters" id="toggle-persist" onclick="togglePersist()">Persist: OFF</button>
</div>
<div id="cards"></div>
<script>
let events = [];
let autoScroll = true;
let persist = false;
let activeFilter = 'all';
let scripts = new Set();
let rateBuf = [];

const cards = document.getElementById('cards');
const dot = document.getElementById('status-dot');

// Restore from localStorage if available
try {
  const saved = localStorage.getItem('vllm_watcher_events');
  if (saved) { events = JSON.parse(saved); persist = true; document.getElementById('toggle-persist').textContent = 'Persist: ON'; renderAll(); }
} catch(e) {}

function connect() {
  const es = new EventSource('/stream');
  es.onopen = () => { dot.classList.add('connected'); };
  es.onerror = () => { dot.classList.remove('connected'); setTimeout(connect, 3000); };
  es.onmessage = (e) => {
    try {
      const ev = JSON.parse(e.data);
      events.push(ev);
      rateBuf.push(ev.ts);
      if (!scripts.has(ev.script)) { scripts.add(ev.script); rebuildFilters(); }
      renderCard(ev, events.length - 1);
      updateStats();
      if (persist) { try { localStorage.setItem('vllm_watcher_events', JSON.stringify(events.slice(-500))); } catch(e){} }
      if (autoScroll) window.scrollTo(0, document.body.scrollHeight);
    } catch(err) {}
  };
}

function renderCard(ev, idx) {
  const div = document.createElement('div');
  const latency = ev.latency_ms || 0;
  let cls = 'card';
  if (ev.status !== 'ok') cls += ' fail';
  else if (latency > 120000) cls += ' slow';
  if (activeFilter !== 'all' && ev.script !== activeFilter) cls += ' hidden';
  div.className = cls;
  div.dataset.script = ev.script;
  div.dataset.idx = idx;
  const time = new Date(ev.ts * 1000).toLocaleTimeString();
  const promptPreview = (ev.prompt || '').replace(/</g,'&lt;');
  const responsePreview = (ev.response || '').replace(/</g,'&lt;');
  div.innerHTML = `
    <div class="card-header">
      <span class="card-script">${ev.script}</span>
      <span class="card-meta">${time} · ${(latency/1000).toFixed(1)}s · ${ev.tokens||'?'} tok · ${ev.status}</span>
    </div>
    <div class="card-prompt" onclick="this.classList.toggle('expanded')">▸ ${promptPreview}</div>
    <div class="card-response" onclick="this.classList.toggle('expanded')">${responsePreview}</div>
    ${ev.error ? '<div class="card-error">' + ev.error.replace(/</g,'&lt;') + '</div>' : ''}
  `;
  cards.appendChild(div);
}

function renderAll() {
  cards.innerHTML = '';
  events.forEach((ev, i) => { if (!scripts.has(ev.script)) scripts.add(ev.script); renderCard(ev, i); });
  rebuildFilters();
  updateStats();
}

function updateStats() {
  document.getElementById('stat-total').textContent = events.length;
  const now = Date.now() / 1000;
  rateBuf = rateBuf.filter(t => now - t < 60);
  document.getElementById('stat-rate').textContent = rateBuf.length;
  if (events.length > 0) {
    const recent = events.slice(-20);
    const avg = recent.reduce((s, e) => s + (e.latency_ms || 0), 0) / recent.length;
    document.getElementById('stat-latency').textContent = (avg / 1000).toFixed(1) + 's';
  }
}

function rebuildFilters() {
  const f = document.getElementById('filters');
  f.innerHTML = '<button class="' + (activeFilter === 'all' ? 'active' : '') + '" data-filter="all" onclick="setFilter(\'all\')">All</button>';
  scripts.forEach(s => {
    f.innerHTML += '<button class="' + (activeFilter === s ? 'active' : '') + '" data-filter="' + s + '" onclick="setFilter(\'' + s + '\')">' + s + '</button>';
  });
}

function setFilter(f) {
  activeFilter = f;
  document.querySelectorAll('.card').forEach(c => {
    c.classList.toggle('hidden', f !== 'all' && c.dataset.script !== f);
  });
  rebuildFilters();
}

function toggleScroll() {
  autoScroll = !autoScroll;
  document.getElementById('toggle-scroll').textContent = 'Auto-scroll: ' + (autoScroll ? 'ON' : 'OFF');
}

function togglePersist() {
  persist = !persist;
  document.getElementById('toggle-persist').textContent = 'Persist: ' + (persist ? 'ON' : 'OFF');
  if (!persist) localStorage.removeItem('vllm_watcher_events');
}

connect();
setInterval(updateStats, 5000);
</script>
</body>
</html>"""


async def handle_request(reader, writer):
    """Minimal HTTP handler — serves HTML or SSE stream."""
    data = await reader.read(4096)
    request_line = data.split(b'\r\n')[0].decode()
    path = request_line.split(' ')[1] if ' ' in request_line else '/'

    if path == '/stream':
        writer.write(b'HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: keep-alive\r\nAccess-Control-Allow-Origin: *\r\n\r\n')
        await writer.drain()

        # Seek to end of file and tail
        try:
            f = open(MONITOR_FILE, 'r')
            f.seek(0, 2)  # Seek to end
        except FileNotFoundError:
            f = None

        try:
            while True:
                if f is None:
                    try:
                        f = open(MONITOR_FILE, 'r')
                        f.seek(0, 2)
                    except FileNotFoundError:
                        await asyncio.sleep(1)
                        continue

                line = f.readline()
                if line:
                    line = line.strip()
                    if line:
                        try:
                            json.loads(line)  # Validate JSON
                            writer.write(f"data: {line}\n\n".encode())
                            await writer.drain()
                        except (json.JSONDecodeError, ValueError):
                            pass
                else:
                    # Send keepalive every 15s
                    writer.write(b": keepalive\n\n")
                    try:
                        await writer.drain()
                    except (ConnectionResetError, BrokenPipeError):
                        break
                    await asyncio.sleep(0.5)
        except (ConnectionResetError, BrokenPipeError, asyncio.CancelledError):
            pass
        finally:
            if f:
                f.close()
    elif path == '/health':
        body = b'ok'
        writer.write(f'HTTP/1.1 200 OK\r\nContent-Length: {len(body)}\r\n\r\n'.encode() + body)
        await writer.drain()
    else:
        body = HTML_PAGE.encode()
        writer.write(f'HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: {len(body)}\r\n\r\n'.encode() + body)
        await writer.drain()

    writer.close()


async def main():
    server = await asyncio.start_server(handle_request, '0.0.0.0', PORT)
    print(f"vLLM Watcher running on http://0.0.0.0:{PORT}")
    async with server:
        await server.serve_forever()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=9091)
    args = parser.parse_args()
    PORT = args.port
    asyncio.run(main())
