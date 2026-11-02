// Drives the real website in headless Chromium via the DevTools protocol (Node >= 22: global WebSocket, fetch).
// Usage: node scripts/ui_e2e.mjs http://127.0.0.1:8765 /path/to/audio.wav /dir/for/screenshots [--separate]
import { spawn } from "node:child_process";
import { writeFileSync, mkdirSync } from "node:fs";

const [base, audio, shots, ...flags] = process.argv.slice(2);
const TEMPO = process.env.TEMPO ?? "100"; // TEMPO="" = let the site detect it
mkdirSync(shots, { recursive: true });
const port = 9300 + Math.floor(Math.random() * 500);
const chrome = spawn("chromium", ["--headless=new", "--disable-gpu", "--no-sandbox", `--remote-debugging-port=${port}`,
  "--window-size=1100,1000", `--user-data-dir=/tmp/ui-e2e-${port}`, "about:blank"], { stdio: "ignore" });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let ws, id = 0; const pending = new Map();
const send = (method, params = {}) => new Promise((res, rej) => { const i = ++id; pending.set(i, { res, rej }); ws.send(JSON.stringify({ id: i, method, params })); });
const ev = async (expr) => { const r = await send("Runtime.evaluate", { expression: expr, returnByValue: true, awaitPromise: true }); if (r.exceptionDetails) throw new Error(JSON.stringify(r.exceptionDetails.exception?.description || r.exceptionDetails)); return r.result.value; };
const until = async (expr, label, ms = 240000) => { const t = Date.now(); while (Date.now() - t < ms) { if (await ev(expr)) return; await sleep(400); } throw new Error("timeout waiting for " + label); };
const shot = async (name) => { const r = await send("Page.captureScreenshot", { format: "png" }); writeFileSync(`${shots}/${name}.png`, Buffer.from(r.data, "base64")); };
const results = [];
const check = (name, ok, extra = "") => { results.push(ok); console.log(`${ok ? "PASS" : "FAIL"}  ${name} ${extra}`); };

try {
  let targets;
  for (let i = 0; i < 50 && !targets; i++) { try { targets = await (await fetch(`http://127.0.0.1:${port}/json`)).json(); } catch { await sleep(200); } }
  const page = targets.find((t) => t.type === "page");
  ws = new WebSocket(page.webSocketDebuggerUrl);
  await new Promise((r) => (ws.onopen = r));
  ws.onmessage = (m) => { const d = JSON.parse(m.data); if (d.id && pending.has(d.id)) { const p = pending.get(d.id); pending.delete(d.id); d.error ? p.rej(new Error(d.error.message)) : p.res(d.result); } };
  await send("Page.enable"); await send("DOM.enable"); await send("Runtime.enable");
  await send("Emulation.setDeviceMetricsOverride", { width: 1100, height: 1000, deviceScaleFactor: 1, mobile: false });

  await send("Page.navigate", { url: base + "/" });
  await until(`document.querySelector('#limits') && document.querySelector('#limits').textContent.includes('minutes')`, "home page + limits");
  check("home page renders with limits", true, await ev(`document.querySelector('#limits').textContent`));
  check("only the home view is visible", await ev(`!document.querySelector('#view-home').hidden && document.querySelector('#view-setup').hidden`));
  await shot("1-home");

  const doc = await send("DOM.getDocument");
  const input = await send("DOM.querySelector", { nodeId: doc.root.nodeId, selector: "#file" });
  await send("DOM.setFileInputFiles", { nodeId: input.nodeId, files: [audio] });
  await until(`!document.querySelector('#view-setup').hidden`, "setup view after upload");
  check("upload leads to the setup view", true, await ev(`document.querySelector('#f-title').textContent + ' | ' + document.querySelector('#f-meta').textContent`));
  check("URL now holds the job id (bookmarkable)", await ev(`/^#\\/[0-9a-f]{32}$/.test(location.hash)`));
  check("create button is disabled until rights are confirmed", await ev(`document.querySelector('#create').disabled`));
  await shot("2-setup");
  check("drum separation is on by default", await ev(`document.querySelector('#separate').checked`));
  if (!flags.includes("--separate")) await ev(`document.querySelector('#separate').click()`);
  await ev(`document.querySelector('#tempo').value='${TEMPO}'; document.querySelector('#rights').click()`);
  check("ticking the rights box enables Create", await ev(`!document.querySelector('#create').disabled`));
  await ev(`document.querySelector('#create').click()`);
  await until(`!document.querySelector('#view-progress').hidden`, "progress view");
  await until(`document.querySelector('#steps li.active, #steps li.done')`, "a running step"); await sleep(1500); await shot("3-progress");
  check("progress view shows steps", await ev(`document.querySelectorAll('#steps li').length >= 7`), await ev(`document.querySelector('#p-title').textContent`));
  await until(`!document.querySelector('#view-result').hidden || !document.querySelector('#p-error').hidden`, "result or failure");
  const ok = await ev(`!document.querySelector('#view-result').hidden`);
  check("job finishes and the result view appears", ok, ok ? "" : await ev(`document.querySelector('#p-error').textContent`));
  if (ok) {
    await sleep(2500);
    const info = await ev(`JSON.stringify({title: document.querySelector('#r-title').textContent, chips: [...document.querySelectorAll('#chips li')].map(l=>l.textContent), pdf: document.querySelector('#dl-pdf').href, player: document.querySelector('#player').src, notes: document.querySelectorAll('#notes-list li').length})`);
    const o = JSON.parse(info); console.log("      ", info);
    check("result shows tempo, bars and hits", o.chips.some((c) => c.includes(TEMPO + " BPM")) && o.chips.some((c) => c.includes("bars")));
    check("PDF link points at the job's file", /\/files\/[0-9a-f]{32}\/pdf\/.*\.pdf\?download=true$/.test(o.pdf));
    const pdf = await ev(`fetch(document.querySelector('#dl-pdf').href).then(async r => ({s: r.status, t: r.headers.get('content-type'), d: r.headers.get('content-disposition'), h: new TextDecoder().decode((await r.arrayBuffer()).slice(0,4))}))`);
    check("PDF downloads as an attachment", pdf.s === 200 && pdf.h === "%PDF" && /attachment/.test(pdf.d), JSON.stringify(pdf));
    check("player is embedded", /player\/index\.html$/.test(o.player));
    const frameOk = await ev(`new Promise(r => { const f = document.querySelector('#player'); const t = setInterval(() => { try { const d = f.contentDocument; if (d && d.querySelector('.page img') && d.getElementById('bar').textContent) { clearInterval(t); r(d.getElementById('bar').textContent + ' | ' + d.getElementById('source').options.length + ' sources'); } } catch (e) {} }, 300); setTimeout(() => r(false), 15000); })`);
    check("embedded player loaded its score and audio sources", !!frameOk, String(frameOk));
    await shot("4-result");
    const alts = await ev(`[...document.querySelectorAll('#tempo-alt-btns .pill')].map(b => b.textContent)`);
    check("alternative tempos are offered", alts.length > 0 && await ev(`!document.querySelector('#tempo-alts').hidden`), JSON.stringify(alts));
    await ev(`document.querySelector('#tempo-alt-btns .pill').click()`);
    await until(`!document.querySelector('#view-progress').hidden`, "progress after choosing another tempo");
    await until(`!document.querySelector('#view-result').hidden`, "result after re-creating with another tempo");
    check("score re-created at the chosen tempo", await ev(`[...document.querySelectorAll('#chips li')].some(l => l.textContent.includes(${JSON.stringify(alts[0])}))`), alts[0]);
    await send("Page.navigate", { url: base + "/" }); await sleep(1200);
    check("finished score is listed under 'Your scores'", await ev(`!document.querySelector('#recent').hidden && document.querySelectorAll('#recent-list li').length >= 1`));
    await shot("5-home-with-recent");
    await ev(`document.querySelector('#recent-list a').click()`); await until(`!document.querySelector('#view-result').hidden`, "reopen from recent");
    check("clicking a recent score reopens it", true);
    await ev(`window.confirm = () => true; document.querySelector('#r-delete').click()`);
    await until(`!document.querySelector('#view-home').hidden`, "home after delete");
    check("deleting returns home and forgets the score", await ev(`document.querySelectorAll('#recent-list li').length === 0`));
  }
} catch (e) { console.log("ERROR", e.message); results.push(false); }
finally { chrome.kill(); }
console.log(results.every(Boolean) ? "ALL UI CHECKS PASSED" : "UI CHECKS FAILED");
process.exit(results.every(Boolean) ? 0 : 1);
