// Runs the page's real JavaScript against a tiny fake DOM and fake API (no browser needed).
// Usage: node tests/ui_smoke.js pka/web/index.html
const fs = require("fs"), vm = require("vm"), assert = require("assert");
const html = fs.readFileSync(process.argv[2], "utf8");
const script = /<script>([\s\S]*?)<\/script>/.exec(html)[1];

class Node_ {}
class FakeText extends Node_ { constructor(t) { super(); this.tag = "#text"; this.text = String(t); } }
class FakeEl extends Node_ {
  constructor(tag) {
    super(); this.tag = tag; this.children = []; this.attrs = {}; this.listeners = {}; this.style = {};
    this.className = ""; this.value = ""; this.checked = false; this.disabled = false; this.scrollTop = 0; this.scrollHeight = 0;
    const self = this;
    this.classList = {
      _s() { return new Set(self.className.split(/\s+/).filter(Boolean)); },
      add(...c) { const s = this._s(); c.forEach(x => s.add(x)); self.className = [...s].join(" "); },
      remove(...c) { const s = this._s(); c.forEach(x => s.delete(x)); self.className = [...s].join(" "); },
      toggle(c, f) { const s = this._s(); const on = f === undefined ? !s.has(c) : f; on ? s.add(c) : s.delete(c); self.className = [...s].join(" "); },
      contains(c) { return this._s().has(c); },
    };
  }
  get childNodes() { return this.children; }
  get firstChild() { return this.children[0] || null; }
  append(...kids) { kids.forEach(k => this.children.push(typeof k === "string" ? new FakeText(k) : k)); }
  appendChild(k) { this.children.push(k); return k; }
  removeChild(k) { this.children.splice(this.children.indexOf(k), 1); return k; }
  setAttribute(k, v) { this.attrs[k] = v; }
  addEventListener(t, f) { (this.listeners[t] = this.listeners[t] || []).push(f); }
  scrollIntoView() {}
  set textContent(t) { this.children = [new FakeText(t)]; }
  get textContent() { return this.children.map(c => c.tag === "#text" ? c.text : c.textContent).join(""); }
}
const byId = {};
for (const m of html.matchAll(/<(\w+)[^>]*\sid="([^"]+)"/g)) byId[m[2]] = new FakeEl(m[1]);
byId.details.checked = false;
const document = {
  getElementById: id => byId[id] || null,
  createElement: t => new FakeEl(t),
  createTextNode: t => new FakeText(t),
  querySelectorAll: () => [],
};
for (const id of Object.keys(byId)) {          // elements created by h() register themselves by id
  const el = byId[id]; Object.defineProperty(el, "id", {get() { return el.attrs.id || id; }});
}
const origCreate = document.createElement;
document.createElement = t => { const e = origCreate(t); return e; };
// h() sets id via setAttribute; make getElementById find those too (cards: "src-1")
const registry = {};
const origSet = FakeEl.prototype.setAttribute;
FakeEl.prototype.setAttribute = function (k, v) { origSet.call(this, k, v); if (k === "id") registry[v] = this; };
document.getElementById = id => byId[id] || registry[id] || null;

function walk(el, fn) { fn(el); (el.children || []).forEach(c => walk(c, fn)); }
const textOf = el => el.tag === "#text" ? el.text : el.textContent;
const find = (el, pred) => { const out = []; walk(el, n => { if (pred(n)) out.push(n); }); return out; };

let routes = {};
const calls = [];
const fetch = async (path, opts) => {
  calls.push([path, opts && opts.method || "GET"]);
  const key = path.split("?")[0];
  const r = routes[key];
  if (!r) return {ok: false, status: 404, statusText: "Not Found", json: async () => ({detail: "not found: " + path})};
  if (r.error) return {ok: false, status: 500, statusText: "err", json: async () => ({detail: r.error})};
  return {ok: true, status: 200, json: async () => r.body};
};
const ctx = vm.createContext({document, fetch, Node: Node_, alert: () => {}, setTimeout: () => 0, clearTimeout: () => {},
  console, encodeURIComponent, Date, parseInt, Object, Set, JSON, Array, String, Math, isNaN});

const status = {ready: true, problem: "", system: "summary", documents: 26, versions: 30, chunks: 412, summaries: 150,
  pending: {docs_total: 19, docs_pending: 1, calls_needed: 164}, syncing: false, watching: true,
  last_sync: {at: "2026-11-13T10:00:00", summary: "1 added, 0 new version(s)", warnings: [], failed: []},
  last_sync_error: "", llm: "groq", corpus_dir: "/x", server_time: "2026-11-13T10:05:00"};
const reply = {answer: "Your rent is **₹30,240**[1] <img src=x onerror=alert(1)> see [2] and [7].", error: "", time_frame: "current (today, 2026-11-13)",
  sources: [
    {n: 1, doc_id: "rental_agreement", version: 2, section: "Doc > 4. Licence Fee", heading: "4. Licence Fee", level: 0, status: "current", in_force: "current version, dated 2026-11-01", score: 5.1, text: "The monthly licence fee is ₹30,240."},
    {n: 2, doc_id: "rental_agreement", version: 1, section: "Doc", heading: "", level: 1, status: "superseded", in_force: "", score: 1.2, text: "x".repeat(400)}]};
const detail = {doc_id: "rental_agreement", title: "Leave and Licence Agreement", versions: [
  {version: 1, from: "2025-12-15", until: "2026-11-01", status: "superseded", chunks: 11, auto: false, summary: {nodes: 3, has_overview: true}},
  {version: 2, from: "2026-11-01", until: null, status: "current", chunks: 11, auto: true, summary: null}],
  changes: [{from: 1, to: 2, kind: "modified", section: "4. Licence Fee", detail: "₹28,000 → ₹30,240"},
            {from: 1, to: 2, kind: "removed", section: "7. Parking", detail: "One covered parking slot"}]};

const test = `(async () => {
  routes["/api/status"] = {body: STATUS}; routes["/api/documents"] = {body: [{doc_id: "rental_agreement", title: "t", versions: 2, chunks: 11, changes: 8, summarised: true}]};
  routes["/api/documents/rental_agreement"] = {body: DETAIL};
  routes["/api/documents/rental_agreement/chunks"] = {body: {doc_id: "rental_agreement", version: 2, chunks: [{index: 0, section: "S", heading: "4. Licence Fee", tokens: 50, text: "Chunk text"}], summaries: [{level: 2, part: 1, overview: true, span: "all", text: "Overview text"}]}};
  routes["/api/ask"] = {body: REPLY}; routes["/api/sync"] = {body: {started: true}};

  await refreshStatus();
  const pills = T.textOf($("pills"));
  T.assert(pills.includes("26 documents"), "document count shown");
  T.assert(pills.includes("Summaries 18/19"), "summary progress shown");
  T.assert(pills.includes("Watching folder"), "watching shown");
  T.assert(pills.includes("5 min ago"), "sync age shown");

  renderMessages();
  T.assert(T.find($("messages"), n => n.tag === "button").length === 3, "three example chips");
  await send("What is my rent now?");
  const m = state.messages[0];
  T.assert(!m.loading && m.sources.length === 2, "reply stored");
  const msgs = $("messages");
  T.assert(T.find(msgs, n => n.className === "cite").length === 2, "only valid citations become buttons ([7] stays text)");
  T.assert(T.find(msgs, n => n.tag === "img").length === 0, "HTML in the answer is not interpreted");
  T.assert(T.textOf(msgs).includes("<img src=x"), "HTML is shown as plain text");
  T.assert(T.textOf(msgs).includes("Time frame: current"), "time frame shown");
  T.assert(T.textOf($("src-1")).includes("₹30,240"), "source 1 card");
  T.assert(T.textOf($("src-2")).includes("summary"), "summary badge");
  T.assert(T.textOf($("src-1")).includes("current version"), "status badge");
  T.assert(T.find($("src-2"), n => n.className === "more").length === 1, "long text gets a Show more button");
  T.assert(!T.textOf($("src-1")).includes("score"), "scores hidden by default");
  $("details").checked = true; renderSources();
  T.assert(T.textOf($("src-1")).includes("score 5.1"), "details toggle shows scores");
  T.assert(T.find($("src-2"), n => n.className === "more").length === 0, "details toggle expands text");

  const cites = T.find($("messages"), n => n.className === "cite");
  cites[1].listeners.click[0]();                 // click the [2] citation in the answer
  T.assert($("src-2").classList.contains("flash"), "clicking a citation highlights its source card");

  await loadDocs(); T.assert(T.textOf($("doclist")).includes("rental_agreement"), "document list");
  await openDocument("rental_agreement");
  const d = T.textOf($("docdetail"));
  for (const want of ["Versions", "v2 (from an edit)", "overview + parts", "What changed", "v1 → v2", "4. Licence Fee: ₹28,000 → ₹30,240", "removed", "Chunks", "Chunk text", "Document overview"])
    T.assert(d.includes(want), "document detail has: " + want);
  state.boundaries = true; await openDocument("rental_agreement", 2, true);
  T.assert(T.textOf($("docdetail")).includes("chunk 1 · 50 tokens · 4. Licence Fee"), "boundary labels");

  routes["/api/ask"] = {error: "boom"};
  await send("again");
  T.assert(state.messages[1].error === "boom", "API errors reach the message");
  T.assert(T.textOf($("messages")).includes("boom"), "error displayed");

  routes["/api/status"] = {body: Object.assign({}, STATUS, {ready: false, problem: "The index is empty."})};
  await refreshStatus();
  T.assert(T.textOf($("banner")).includes("The index is empty."), "problem banner");
  routes["/api/status"] = {body: Object.assign({}, STATUS, {syncing: true})};
  await refreshStatus();
  T.assert(T.textOf($("pills")).includes("Syncing"), "syncing pill"); T.assert($("syncBtn").disabled, "sync button disabled while syncing");
  return "ui smoke ok";
})()`;
ctx.routes = routes; ctx.STATUS = status; ctx.REPLY = reply; ctx.DETAIL = detail;
ctx.T = {assert: (c, msg) => { if (!c) throw new Error("FAILED: " + msg); }, find, textOf};
const result = vm.runInContext(script + "\n" + test, ctx);
Promise.resolve(result).then(r => { console.log(r); process.exit(0); },
  e => { console.error(e.message || e); process.exit(1); });
