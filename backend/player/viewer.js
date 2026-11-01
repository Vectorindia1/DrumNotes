// Score viewer with a blue playhead. Pure position maths first (tested under node), DOM below.
(function (root) {
  // last index i with arr[i] <= v (0 if v is before the first element)
  function lastLE(arr, v) {
    let lo = 0, hi = arr.length - 1, ans = 0;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      if (arr[mid] <= v) { ans = mid; lo = mid + 1; } else hi = mid - 1;
    }
    return ans;
  }
  // piecewise-linear map xs -> ys (xs ascending), clamped at both ends
  function interp(xs, ys, v) {
    if (v <= xs[0]) return ys[0];
    if (v >= xs[xs.length - 1]) return ys[ys.length - 1];
    const i = lastLE(xs, v);
    const f = (v - xs[i]) / (xs[i + 1] - xs[i] || 1);
    return ys[i] + f * (ys[i + 1] - ys[i]);
  }
  // time in a source recording -> time on the score's own (constant-tempo) timeline, and back
  // The synthesised score audio is heard `latency` s after its notated time (synth attack + mp3 delay,
  // measured by cross-correlation); the playhead compensates for it.
  function toScoreTime(data, src, t) {
    if (src === "score") return t - (data.sources.score.latency || 0);
    return interp(data.sources[src].beats, data.beatsScore, t);
  }
  function fromScoreTime(data, src, t) {
    if (src === "score") return t + (data.sources.score.latency || 0);
    return interp(data.beatsScore, data.sources[src].beats, t);
  }
  // where the playhead sits for a score time: {bar, page, x, y, h} in page pixels
  function positionAt(data, tScore) {
    const m = data.measures;
    const bar = lastLE(m.map((b) => b.t), tScore);
    const b = m[bar];
    return { bar, page: b.page, x: interp(b.stops.t, b.stops.x, tScore), y: b.y, h: b.h };
  }
  // Smoothed media clock: audio.currentTime advances in coarse steps (and lags the sound that has
  // actually left the speakers), so between updates it is extrapolated with the wall clock.
  // st = {anchor:null}; returns the estimated playback position in seconds.
  function smoothTime(st, a, now, paused, rate) {
    if (paused) { st.anchor = null; return a; }
    if (!st.anchor || a !== st.anchor.a) st.anchor = { a, t: now };
    return Math.min(a + (now - st.anchor.t) * rate, a + 0.08);
  }
  const api = { lastLE, interp, toScoreTime, fromScoreTime, positionAt, smoothTime };
  if (typeof module !== "undefined" && module.exports) { module.exports = api; return; }
  root.ScoreMath = api;

  // ---------------- DOM ----------------
  const D = root.SCORE;
  const $ = (id) => document.getElementById(id);
  const audio = $("audio");
  const pagesEl = $("pages");
  const head = document.createElement("div");
  head.className = "playhead";
  const box = document.createElement("div");
  box.className = "barbox";
  const pageEls = D.pages.map((p, i) => {
    const wrap = document.createElement("div");
    wrap.className = "page";
    wrap.style.aspectRatio = p.w + " / " + p.h;
    wrap.innerHTML = '<img src="' + p.src + '" alt="page ' + (i + 1) + '" draggable="false">';
    wrap.addEventListener("click", (e) => onPageClick(e, i, wrap));
    pagesEl.appendChild(wrap);
    return wrap;
  });
  const clock = { anchor: null };
  let syncS = 0;  // seconds the sound is heard AFTER audio.currentTime (device/browser output latency)
  let src = "score", curBar = -1, curPage = -1, hold = false;  // hold: test hook keeps a parked position

  function place(pos) {
    const p = D.pages[pos.page];
    if (curPage !== pos.page) { pageEls[pos.page].append(box, head); curPage = pos.page; }
    head.style.left = (pos.x / p.w) * 100 + "%";
    head.style.top = (pos.y / p.h) * 100 + "%";
    head.style.height = (pos.h / p.h) * 100 + "%";
    if (pos.bar !== curBar) {
      const b = D.measures[pos.bar];
      box.style.left = (b.x / p.w) * 100 + "%"; box.style.top = (b.y / p.h) * 100 + "%";
      box.style.width = (b.w / p.w) * 100 + "%"; box.style.height = (b.h / p.h) * 100 + "%";
      curBar = pos.bar;
      $("bar").textContent = "bar " + (pos.bar + 1) + " / " + D.measures.length;
      const r = head.getBoundingClientRect();
      if (r.top < 90 || r.bottom > innerHeight - 40) window.scrollBy({ top: r.top - innerHeight / 3, behavior: "smooth" });
    }
  }
  function tick() {
    if (!(hold && audio.paused)) {
      const t = smoothTime(clock, audio.currentTime, performance.now() / 1000, audio.paused, audio.playbackRate);
      place(positionAt(D, toScoreTime(D, src, t - syncS)));
    }
    requestAnimationFrame(tick);
  }
  function seekBar(bar) {
    hold = false;
    bar = Math.max(0, Math.min(D.measures.length - 1, bar));
    audio.currentTime = fromScoreTime(D, src, D.measures[bar].t) + syncS;
    place(positionAt(D, D.measures[bar].t));
  }
  function onPageClick(e, pageIdx, wrap) {
    const r = wrap.getBoundingClientRect(), p = D.pages[pageIdx];
    const x = ((e.clientX - r.left) / r.width) * p.w, y = ((e.clientY - r.top) / r.height) * p.h;
    let best = -1, bd = 1e9;
    D.measures.forEach((b, i) => {
      if (b.page !== pageIdx || y < b.y - 40 || y > b.y + b.h + 40) return;
      const d = x < b.x ? b.x - x : x > b.x + b.w ? x - b.x - b.w : 0;
      if (d < bd) { bd = d; best = i; }
    });
    if (best >= 0) seekBar(best);
  }
  function setSource(name) {
    const tScore = toScoreTime(D, src, audio.currentTime - syncS), wasPlaying = !audio.paused;
    src = name; audio.src = D.sources[name].file;
    audio.addEventListener("loadedmetadata", function once() {
      audio.removeEventListener("loadedmetadata", once);
      audio.currentTime = fromScoreTime(D, src, tScore) + syncS;
      if (wasPlaying) audio.play();
    });
  }
  function init() {
    $("title").textContent = D.title + " — " + D.tempo + " BPM";
    const sel = $("source");
    Object.keys(D.sources).forEach((k) => sel.add(new Option(D.sources[k].label, k)));
    sel.onchange = () => setSource(sel.value);
    $("rate").onchange = (e) => { audio.playbackRate = parseFloat(e.target.value); };
    const btn = $("play");
    btn.onclick = () => (audio.paused ? audio.play() : audio.pause());
    audio.onplay = () => { hold = false; btn.textContent = "⏸ Pause"; };
    audio.onpause = () => (btn.textContent = "▶ Play");
    // sync offset: default from the browser's reported output latency, then remembered per browser
    let saved = null;
    try { saved = localStorage.getItem("drumscribe.syncMs"); } catch (e) { /* storage blocked */ }
    let ms = saved !== null ? +saved : 0;
    if (saved === null) {
      try {
        const ctx = new (window.AudioContext || window.webkitAudioContext)();
        ms = Math.round(((ctx.outputLatency || 0) + (ctx.baseLatency || 0)) * 1000);
        ctx.close();
      } catch (e) { /* no Web Audio */ }
    }
    const slider = $("sync"), label = $("syncv");
    const setSync = (v) => {
      ms = Math.max(-300, Math.min(300, Math.round(v)));
      syncS = ms / 1000; slider.value = ms; label.textContent = (ms > 0 ? "+" : "") + ms + " ms";
      try { localStorage.setItem("drumscribe.syncMs", String(ms)); } catch (e) { /* ignore */ }
    };
    slider.oninput = () => setSync(+slider.value);
    setSync(ms);
    $("prev").onclick = () => seekBar(curBar - 1);
    $("next").onclick = () => seekBar(curBar + 1);
    addEventListener("keydown", (e) => {
      if (e.target.tagName === "SELECT") return;
      if (e.code === "Space") { e.preventDefault(); btn.click(); }
      if (e.key === "[") setSync(ms - 10);
      if (e.key === "]") setSync(ms + 10);
      if (e.code === "ArrowLeft") seekBar(curBar - 1);
      if (e.code === "ArrowRight") seekBar(curBar + 1);
    });
    audio.src = D.sources.score.file;
    const q = new URLSearchParams(location.search);
    if (q.has("bar")) { hold = true; place(positionAt(D, D.measures[+q.get("bar") - 1].t)); }  // visual test hook
    else if (q.has("t")) {  // test hook: park the playhead where time t of source ?src= belongs
      hold = true; src = q.get("src") || "score";
      place(positionAt(D, toScoreTime(D, src, +q.get("t"))));
    }
    else place(positionAt(D, 0));
    requestAnimationFrame(tick);
  }
  init();
})(typeof window !== "undefined" ? window : globalThis);
