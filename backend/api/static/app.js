(() => {
  "use strict";
  const $ = (s) => document.querySelector(s);
  const views = { home: $("#view-home"), setup: $("#view-setup"), progress: $("#view-progress"), result: $("#view-result") };
  const STEPS = [
    "Reading the audio", "Separating the drums from the mix", "Finding the drum hits",
    "Telling ghost notes, open hi-hats and cymbals apart", "Working out tempo and bar lines",
    "Writing the notation", "Engraving the PDF", "Building the playback player",
  ];
  const ART_NAMES = { ghost_notes: "ghost notes", open_hihats: "open hi-hats", ride: "ride", crash: "crash", splash: "splash", china: "china", flams: "flams", tambourine: "tambourine / jingles" };
  let job = null, pollTimer = null, upXhr = null, routeToken = 0, limits = { max_minutes: 12, max_upload_mb: 200 };

  // ---------- helpers (all dynamic text goes through textContent, never innerHTML) ----------
  const say = (t) => { $("#live").textContent = t; };
  const fmtDur = (s) => `${Math.floor(s / 60)}:${String(Math.round(s % 60)).padStart(2, "0")}`;
  const fmtMin = (m) => (m < 1.5 ? "about a minute" : `about ${Math.round(m)} minutes`);
  function show(name) {
    for (const [k, v] of Object.entries(views)) v.hidden = k !== name;
    const h = views[name].querySelector("h1, h2");
    if (h) { h.tabIndex = -1; h.focus({ preventScroll: true }); }
    window.scrollTo(0, 0);
  }
  function showError(el, msg) { el.textContent = msg; el.hidden = !msg; }
  async function api(path, opts) {
    const r = await fetch(path, opts);
    let data = null;
    try { data = await r.json(); } catch (e) { /* not json */ }
    if (!r.ok) {
      const d = data && data.detail;
      throw new Error(typeof d === "string" ? d : "Something went wrong. Please check your settings and try again.");
    }
    return data;
  }
  const store = {
    get() { try { return JSON.parse(localStorage.getItem("drumscribe.recent") || "[]"); } catch (e) { return []; } },
    set(v) { try { localStorage.setItem("drumscribe.recent", JSON.stringify(v.slice(0, 8))); } catch (e) { /* storage blocked */ } },
    add(j) { this.set([{ id: j.id, title: j.title, ts: Date.now() }, ...this.get().filter((x) => x.id !== j.id)]); },
    drop(id) { this.set(this.get().filter((x) => x.id !== id)); },
  };

  // ---------- home ----------
  function renderRecent() {
    const items = store.get(), list = $("#recent-list");
    list.replaceChildren(...items.map((it) => {
      const li = document.createElement("li"), a = document.createElement("a"), t = document.createElement("span"), d = document.createElement("span");
      a.href = "#/" + it.id; t.textContent = it.title; d.textContent = new Date(it.ts).toLocaleDateString(undefined, { month: "short", day: "numeric" });
      a.append(t, d); li.append(a); return li;
    }));
    $("#recent").hidden = items.length === 0;
  }
  function dropState(busy, msg, frac) {
    $("#drop").classList.toggle("busy", busy);
    $(".drop-idle").hidden = busy; $(".drop-busy").hidden = !busy;
    if (msg) $("#drop-msg").textContent = msg;
    const bar = $("#up-bar");
    bar.parentElement.classList.toggle("indeterminate", frac === null);
    if (frac !== null && frac !== undefined) { bar.style.width = Math.round(frac * 100) + "%"; bar.parentElement.setAttribute("aria-valuenow", Math.round(frac * 100)); }
  }
  function upload(file) {
    showError($("#home-error"), "");
    if (file.size > limits.max_upload_mb * 1024 * 1024) { showError($("#home-error"), `That file is too large (limit ${limits.max_upload_mb} MB).`); return; }
    dropState(true, "Uploading…", 0);
    const xhr = (upXhr = new XMLHttpRequest());
    xhr.open("POST", "/api/upload");
    xhr.setRequestHeader("X-Filename", encodeURIComponent(file.name));
    xhr.upload.onprogress = (e) => { if (e.lengthComputable) dropState(true, e.loaded < e.total ? "Uploading…" : "Checking the file…", e.loaded < e.total ? e.loaded / e.total : null); };
    xhr.onload = () => {
      upXhr = null; let data = null;
      try { data = JSON.parse(xhr.responseText); } catch (e) { /* ignore */ }
      dropState(false);
      if (xhr.status >= 200 && xhr.status < 300 && data) { store.add(data); location.hash = "#/" + data.id; }
      else showError($("#home-error"), (data && typeof data.detail === "string" && data.detail) || "The upload failed. Please try again.");
    };
    xhr.onerror = () => { upXhr = null; dropState(false); showError($("#home-error"), "The connection was interrupted. Please try again."); };
    xhr.onabort = () => { upXhr = null; dropState(false); };
    xhr.send(file);
  }
  const drop = $("#drop"), fileInput = $("#file");
  fileInput.addEventListener("change", () => { if (fileInput.files[0]) upload(fileInput.files[0]); fileInput.value = ""; });
  drop.addEventListener("keydown", (e) => { if ((e.key === "Enter" || e.key === " ") && !upXhr) { e.preventDefault(); fileInput.click(); } });
  drop.addEventListener("click", (e) => { if (upXhr) e.preventDefault(); });
  ["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => { const f = e.dataTransfer.files[0]; if (f && !upXhr) upload(f); });
  $("#up-cancel").addEventListener("click", (e) => { e.preventDefault(); if (upXhr) upXhr.abort(); });

  // ---------- setup ----------
  function renderSetup(j) {
    $("#f-title").textContent = j.title;
    $("#f-meta").textContent = `${fmtDur(j.duration_s)} · ${j.size_mb} MB`;
    const p = j.params || {};
    $("#tempo").value = p.tempo || ""; $("#kit").value = p.kit || "standard"; $("#hats").value = p.hat_detail || "auto"; $("#meter").value = p.meter || "auto"; $("#separate").checked = !!p.separate;
    $("#rights").checked = false; $("#create").disabled = true; showError($("#setup-error"), "");
    const tooLong = j.duration_s > j.limits.max_separate_minutes * 60;
    $("#separate").disabled = tooLong;
    $("#sep-note").textContent = tooLong
      ? `Available for recordings up to ${j.limits.max_separate_minutes} minutes.`
      : `Slower (${fmtMin(j.estimate.separated)} for this file). Can help on dense mixes, but not always.`;
    updateEta(j);
    show("setup");
  }
  function updateEta(j) {
    $("#eta").textContent = `Usually ${fmtMin($("#separate").checked ? j.estimate.separated : j.estimate.standard)} for this file.`;
  }
  $("#separate").addEventListener("change", () => job && updateEta(job));
  $("#rights").addEventListener("change", () => { $("#create").disabled = !$("#rights").checked; });
  $("#create").addEventListener("click", async () => {
    const t = $("#tempo").value.trim();
    $("#create").disabled = true; showError($("#setup-error"), "");
    try {
      job = await api(`/api/jobs/${job.id}/start`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ rights: $("#rights").checked, tempo: t ? parseFloat(t) : null, separate: $("#separate").checked, kit: $("#kit").value, hat_detail: $("#hats").value, meter: $("#meter").value }),
      });
      route();
    } catch (e) { showError($("#setup-error"), e.message); $("#create").disabled = !$("#rights").checked; }
  });
  $("#f-change").addEventListener("click", async () => { if (job) { try { await api("/api/jobs/" + job.id, { method: "DELETE" }); } catch (e) { /* ignore */ } store.drop(job.id); } location.hash = "#/"; });

  // ---------- progress ----------
  function renderProgress(j) {
    const sep = j.params && j.params.separate, steps = STEPS.filter((s) => sep || s !== STEPS[1]);
    const idx = steps.indexOf(j.stage);
    $("#steps").replaceChildren(...steps.map((s, i) => {
      const li = document.createElement("li"); li.textContent = s;
      if (idx > i) li.className = "done"; else if (idx === i) li.className = "active"; return li;
    }));
    const failed = j.state === "failed", waiting = j.state === "queued";
    const bar = $("#p-bar"); bar.style.width = Math.round((j.progress || 0) * 100) + "%";
    const track = bar.parentElement; track.setAttribute("aria-valuenow", Math.round((j.progress || 0) * 100));
    track.classList.toggle("indeterminate", j.state === "running" && sep && j.stage === STEPS[1]);
    $("#p-title").textContent = failed ? "This one didn't work" : waiting ? "Waiting for a turn" : "Creating your score";
    $("#p-sub").textContent = failed ? "" : waiting
      ? (j.queue_position > 1 ? `${j.queue_position - 1} other score${j.queue_position > 2 ? "s are" : " is"} ahead of yours.` : "You're next.")
      : `${j.title} · usually ${fmtMin(sep ? j.estimate.separated : j.estimate.standard)}.`;
    showError($("#p-error"), failed ? j.error || "Something went wrong." : "");
    if (failed && j.detail) $("#p-error").textContent += ` (${j.detail})`;  // only present when the operator set DRUMSCRIBE_DEBUG=1
    $("#p-retry").hidden = !failed; $("#p-cancel").textContent = failed ? "Delete" : "Cancel";
    $("#p-hint").hidden = failed;
    $("#steps").hidden = failed;
    say(failed ? "Failed" : j.stage || "Working");
  }
  $("#p-retry").addEventListener("click", () => { if (job) { job.state = "uploaded"; renderSetup(job); } });
  $("#p-cancel").addEventListener("click", async () => {
    if (!job) return;
    try { await api("/api/jobs/" + job.id, { method: "DELETE" }); } catch (e) { /* ignore */ }
    store.drop(job.id); location.hash = "#/";
  });

  // ---------- result ----------
  function chip(parts, soft) {
    const li = document.createElement("li"); if (soft) li.className = "soft";
    parts.forEach((part) => {
      const [txt, bold] = Array.isArray(part) ? part : [part, false];  // a bare string is plain text
      if (bold) { const b = document.createElement("b"); b.textContent = txt; li.append(b); } else li.append(txt);
    });
    return li;
  }
  function renderResult(j) {
    const r = j.result, s = r.summary, base = `/files/${j.id}/`;
    $("#r-title").textContent = s.title;
    const total = Object.values(s.hits).reduce((a, b) => a + b, 0);
    const chips = [chip([[`${s.tempo_bpm} BPM`, true], s.tempo_detected ? " detected" : " set by you"]), chip([[`${s.bars}`, true], " bars"]), ...(s.time_signature ? [chip([[s.time_signature, true], ` · ${s.marking}`])] : []),
      chip([[`${total}`, true], " drum hits"]), chip([[fmtDur(s.duration_s), true], " long"])];
    Object.entries(s.articulations).forEach(([k, v]) => chips.push(chip([[`${v}`, true], ` ${ART_NAMES[k] || k}`], true)));
    $("#chips").replaceChildren(...chips);
    const alts = (s.tempo_alternatives || []).filter((t) => Math.abs(t - s.tempo_bpm) > 1);
    $("#tempo-alts").hidden = alts.length === 0;
    $("#tempo-alt-btns").replaceChildren(...alts.map((t) => {
      const b = document.createElement("button"); b.type = "button"; b.className = "pill"; b.textContent = `${t} BPM`;
      b.addEventListener("click", () => recreate(t)); return b;
    }));
    $("#dl-pdf").href = base + r.files.pdf + "?download=true"; $("#dl-xml").href = base + r.files.musicxml + "?download=true";
    $("#dl-mid").href = base + r.files.midi + "?download=true"; $("#dl-json").href = base + r.files.json + "?download=true";
    $("#open-player").href = base + r.files.player;
    const frame = $("#player"); if (frame.dataset.job !== j.id) { frame.src = base + r.files.player; frame.dataset.job = j.id; }
    $("#notes-list").replaceChildren(...s.notes.map((n) => { const li = document.createElement("li"); li.textContent = n; return li; }));
    $("#r-keep").textContent = `Kept for ${j.limits.retention_hours} hours, then deleted.`;
    store.add({ id: j.id, title: s.title });
    show("result"); say("Your score is ready");
  }
  async function recreate(tempo) {  // same upload and options, different tempo
    const p = job.params || {};
    try {
      job = await api(`/api/jobs/${job.id}/start`, { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ rights: true, tempo, separate: !!p.separate, kit: p.kit || "standard", hat_detail: p.hat_detail || "auto", meter: p.meter || "auto" }) });
      route();
    } catch (e) { say(e.message); }
  }
  $("#r-again").addEventListener("click", () => { if (job) renderSetup(job); });
  $("#r-delete").addEventListener("click", async () => {
    if (!job || !confirm("Delete this score and your uploaded audio?")) return;
    try { await api("/api/jobs/" + job.id, { method: "DELETE" }); } catch (e) { /* ignore */ }
    store.drop(job.id); $("#player").removeAttribute("src"); delete $("#player").dataset.job; location.hash = "#/";
  });

  // ---------- routing ----------
  async function route() {
    clearTimeout(pollTimer);
    const token = ++routeToken, m = location.hash.match(/^#\/([0-9a-f]{32})$/);
    if (!m) { job = null; renderRecent(); show("home"); return; }
    try { job = await api("/api/jobs/" + m[1]); }
    catch (e) { store.drop(m[1]); job = null; location.hash = "#/"; showError($("#home-error"), e.message); return; }
    if (token !== routeToken) return;
    if (job.state === "uploaded") renderSetup(job);
    else if (job.state === "done") renderResult(job);
    else { renderProgress(job); show("progress"); if (job.state !== "failed") poll(token); }
  }
  function poll(token) {
    pollTimer = setTimeout(async () => {
      if (token !== routeToken) return;
      try { job = await api("/api/jobs/" + job.id); } catch (e) { return poll(token); }
      if (token !== routeToken) return;
      if (job.state === "done") renderResult(job);
      else { renderProgress(job); if (job.state !== "failed") poll(token); }
    }, 1500);
  }
  window.addEventListener("hashchange", route);
  fetch("/api/health").then((r) => r.json()).then((h) => {
    limits = h.limits;
    $("#limits").textContent = `Up to ${limits.max_minutes} minutes · ${limits.max_upload_mb} MB`;
  }).catch(() => {});
  route();
})();
