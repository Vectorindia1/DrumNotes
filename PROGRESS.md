# PROGRESS

**Current milestone:** M1 pipeline + M2 (Demucs, run) + first M3/M4/M5 pieces (articulations, playhead player). Tested on SYNTHETIC audio and one unannotated real clip; **acceptance test BLOCKED** (no annotated real data). Nothing is accepted.
**Git:** M0 committed (2551042). M1 work is uncommitted.

## Status by feature (PRD §20)
| Feature | Status | Evidence |
|---|---|---|
| Environment / CPU venv / setup.sh | VERIFIED (M0) | check_environment.py, setup.sh rerun OK; setup.ps1 UNVERIFIED; Docker UNVERIFIED |
| ADTOF wrapper (strict weights, zero-pad first-frame fix) | TESTED (synthetic) | backend/tests/test_detection.py |
| Event model (raw time, sample, confidence, model output, provenance; versioned JSON) | TESTED | test_rhythm, test_pipeline |
| Beat grid + 16th quantization (raw timestamps preserved) | TESTED (synthetic) / EXPERIMENTAL unhinted | test_rhythm.py |
| Simultaneous events (multi-label, same slot) | TESTED | test_rhythm, test_notation, test_pipeline |
| MusicXML writer (perc clef, 2 voices, rests, chords; no tuplets) | TESTED + validated | test_notation (random bars, 3/4), structural validator |
| MuseScore PDF + visual inspection | VERIFIED on synthetic | outputs/pdf/synth_rock.pdf rasterized and inspected |
| MIDI (raw times) / JSON / report | TESTED (files non-empty, JSON schema 1.0) | test_pipeline |
| Benchmarks per category | A runs; B, C BLOCKED | scripts/run_benchmarks.py |
| Ghost notes (parenthesised) | EXPERIMENTAL | snare loudness vs local main level (bimodal on the user clip: loud ~1.0 vs quiet 0.1-0.35) + low-threshold candidates; synthetic tests only |
| Open hi-hat (circled x) | EXPERIMENTAL | HF tail ratio >= 0.10 (guess); synthetic tests only |
| Ride / crash / splash / china (distinct heights, shapes, labels) | EXPERIMENTAL | ride from regular runs; splash/china thresholds placed above this one clip's p90; uncalibrated |
| Flams (grace note) | **NOT WORKING on real audio** | ADTOF activations contain no grace-note peaks (0% before or after 120 loud snares); evidence-gated route yields only 2 low-trust flams |
| Demucs drum separation (`--separate`) | EXPERIMENTAL (executed) | 397.5 s clip in 764 s, 2.5 GB peak RSS; see M2 section |
| Score player with blue playhead (`--player`) | TESTED (maths + browser DOM + sync), EXPERIMENTAL | see Player section; never heard in a real browser by me |
| Tuplets, swing, meter alternatives, editing/correction UI | NOT STARTED | M3, M5 |

`pytest backend/tests -q`: 87 passed. `ruff check backend scripts`: clean.

## Category A (synthetic) results — NOT real-world accuracy
ADTOF-pytorch, F1 @ ±50 ms: kick 0.96–1.00, snare 1.00 (0.77 when stacked with kick+hat in a sparse pattern: 5/8 found, activation 0.12–0.15 < thr 0.24), hi-hat 0.99–1.00. Median onset error ≈ 10 ms (detections sit one 10 ms frame early: 1.19 s vs 1.20 s), so ±10 ms F1 is only 0.61–0.94. Full table: `python scripts/run_benchmarks.py` (writes outputs/json/benchmarks.json).

## Exploratory run on user-provided audio (NOT accuracy; no ground truth)
`audio file/WhatsApp Audio 2026-11-01 at 11.37.24.mpga` (copied to data/incoming/user_clip.mp3, git-ignored): 394.9 s stereo MP3, licence unknown, no annotations. Unhinted run: 23 s wall, ~1.6 GB peak RSS, exit 0. Tempo 107.67 BPM (alternatives 53.8 / 215.3 unresolved), 4/4 assumed. 1763 events (hihat 863, cymbal 327, snare 326, kick 200, tom 47), 180 bars, MusicXML valid, PDF 4 pages (page 1 inspected). Max quantization error 76.9 ms, 4 events beyond half a 16th, 11 same-slot duplicates merged.
Observations to verify by ear (unverified): bars 1-3 are rests, then ~15 bars of an unbroken 16th hi-hat stream before kick/snare enter; events per 30 s window are ~150-230 up to ~240 s and then collapse to 0-36 (drums may end, or the model loses them). Kick count (200) is low relative to snare+cymbal. No claim of correctness. `audio 1` in that folder is an empty file (ignored).

## User feedback round (2026-11-01) on the first PDF
User: true tempo is 110 BPM; open hi-hats missing; crash/ride/splash/china/open hat need distinct symbols; many ghost notes missed; flams absent (their main concern); otherwise "good".
Done: `--tempo 110` now runs the beat tracker at that fixed tempo to find phase (exact grid only with `--offset`); bar-1 guess from kick+snare (reported LOW CONFIDENCE here: use `--beat1 N`, or `--tempo 110 --offset <time of a beat-1 in s>`). New notation keys (MuseScore honours notehead shapes only per-instrument): hi-hat x, open hi-hat circled x, ride x top line, crash x ledger above, splash inverted triangle + label, china triangle + label, ghost in parentheses, flam as slashed grace note; key printed in the footer. MIDI uses GM 46/51/49/55/52, ghost velocity 45.
Result on the clip (outputs/user_clip_v2/, 110 BPM, 4 pages, MusicXML valid): ghost 244 (179 main-level + 60 low-threshold candidates; ~63 % of snares: plausible only if the groove is ghost-heavy), open hi-hat 52, ride 127, crash 190, splash 9, china 1, flams 2. ALL UNVERIFIED by ear.
Lessons: envelope-only flam detection flagged 179 of 326 snares (pure noise from hats/guitars); a china/splash bug (decay cut off by next hit counted as 0) fixed; absolute flatness cut-offs are meaningless in a mix (clip's flatness spans 0.55-0.72).
Flam path forward (needs user decision): (a) user names bars/times with flams so features at those hits can be inspected and thresholds calibrated; (b) M2: Demucs drum stem, then fine-resolution envelope analysis (download + CPU time/RAM risk: ~3 GiB free).

## M2 — separation (2026-11-01)
Demucs htdemucs (weights sha256 in MODEL_REGISTRY; licence of weights UNVERIFIED). `python -m backend.pipeline AUDIO --tempo 110 --separate` detects and analyses articulations on the drum stem; tempo still from the mix. Stems are cached in data/stems (git-ignored).
Comparison (`scripts/compare_separation.py` -> outputs/json/separation_comparison.json), F1 @ ±50 ms:
- **A, synthetic mixes (drums + synthetic bass/chords; crude, NOT real evidence):** separation fixed hi-hat false positives in all 4 mixes (hi-hat P 0.46-0.96 -> 1.00) and snare recall in stacked patterns (0.62 -> 1.00 at -6 dB), but **hurt** snare recall on rock_100bpm at -6 dB (0.50 -> 0.17: snares removed). Kick unchanged (1.00). So separation is not uniformly better.
- **B, real clip, no truth (agreement only):** kick 200 -> 223 (37 stem-only), snare 326 -> 293 (50 original-only), hi-hat 863 -> 828, cymbal 327 -> 289, **tom 47 -> 6** (43 original-only; likely drum-mix leakage or real toms lost: unknown). 90-95 % of events agree within 50 ms. Which is closer to the truth cannot be said without annotations.
- Articulations on the stem (outputs/user_clip_v3): ghost 170 (+15 candidates), open hi-hat 68, ride 241, crash 37, splash 10, china 1, flams 2. Diagnostic: with a clean stem and strict envelope criteria only 1 of 130 loud snares has a grace-note candidate, so flam detection still finds essentially nothing. UNVERIFIED by ear.
- M2 exit ("measured improvement") is NOT met: no annotated real-mix data, and the synthetic result is mixed.

## Player (2026-11-01)
`--player` builds `<out>/player/index.html` (static, no server; open in a browser): MuseScore page images + MuseScore's own measure/segment positions, a blue playhead sweeping note by note across systems and pages, current-bar highlight, play/pause, bar prev/next, click a bar to jump, speed 0.5-1.25x, audio source switch with position carried over: (1) score rendered by MuseScore (what the notation says; measured ~36 ms late vs notated times, no drift, compensated), (2) original recording, (3) separated drum stem; (2)/(3) are mapped to the score through the beat grid (constant tempo grid -> follows the song only as well as the grid does).
Verified: position maths (node test incl. page turn, clamping, latency); builder end-to-end test on a 10-bar score; in headless Chromium the playhead lands on the right bar for 8 sampled real events x 2 sources (16/16) and visually on bars 1, 48, 60; a bug where beat_shift pre-roll made original/stem playheads run up to 1.6 s early was found by that check and fixed (Grid.slot_time now extrapolates). NOT verified: actual audible playback in a real browser (headless cannot play audio), mobile layout, very long scores (74 mostly empty bars on the last page is fine).

## Score audio fix: sampled drum kit (2026-11-01)
User report: the "synthesised score drums" sounded piano-based. Cause (measured): MuseScore's CLI audio export plays the part with a tonal patch (98 % harmonic energy) although its MIDI export is a correct drum track (channel 10, GM 36/38/42). Fix: `backend/player/score_audio.py` renders every notated hit at its exact bar/slot time (from MuseScore's measure positions) through libfluidsynth + MuseScore_General_Lite.sf3 (MIT) drum kits: ghost notes soft (vel 42), flams as a quiet grace hit 35 ms before, open hi-hat GM 46, ride 51, crash 49, splash 55, china 52. Result: 0.01 harmonic energy; onset delay of the drum samples 1.5-4.3 ms (latency set to 0); FLAC output so no encoder delay. Kits: `--kit standard|room|power|electronic|808|jazz|brush`. Fallback (no libfluidsynth/soundfont): MuseScore's tonal render, labelled as not a drum kit. Tests: test_score_audio.py (percussive, placement within 20 ms, ghost quieter, unknown kit rejected). The kit is a General-MIDI soundfont kit (sampled, decent), not a studio drum library; a better SF2/SFZ kit can be dropped in via SOUNDFONTS. Not heard by me; judged by spectral measurements only.

## Playback delay fix + links (2026-11-01)
User: playback sound felt delayed; also wants YouTube/Spotify links to produce notes.
**Delay: measured causes and fixes.** (1) The beat grid sat ~46 ms LATE vs the detected hits (kick -46, snare -46, hi-hat -48, cymbal -46 ms, tight spread), so the original/stem playhead trailed the sound: new `calibrate_phase` shifts tracker-derived grids onto the hits (circular mean; -40.2 ms on the clip). (2) ADTOF event times lead the real attack by 7-15 ms (10 ms frames, 46 ms window): new `refine_onsets` re-locates each attack from the instrument's own band at 1.45 ms resolution, calibrated on synthetic hits with exact truth (within ~0.4 ms median); the detector's own time is kept in `model_output.frame_time_s`; hits in the first 30 ms of a file cannot be refined. (3) Original/stem audio is now lossless FLAC (mp3 adds ~25-50 ms that browsers may not strip); the score FLAC compensates its measured 3 ms. (4) Player: smoothed playback clock + a Sync slider (-300..+300 ms, keys [ ]) preset from the browser's AudioContext output latency and remembered: speaker/Bluetooth delay cannot be known from here, so this one needs the user's ear. After fixes on the clip: hit-vs-grid median error kick +6, snare +2, hi-hat -2, cymbal +1 ms (was -46); kick and hi-hat that coincide differ by 6.5 ms (kick refinement likely a few ms late on real kicks). Bar mapping re-verified in headless Chromium: 16/16. NOT verified: audible sync in a real browser/speaker.
**Links.** `backend/intake.py`: local files and direct audio-file URLs work (any format: non-libsndfile codecs such as m4a/aac are converted to wav with ffmpeg; found because m4a previously could not be read at all). YouTube/Spotify links are refused on purpose with the track title (public oEmbed metadata) and next steps: Spotify audio is DRM-protected with no audio API; YouTube audio needs downloading, which its terms forbid for others' music; the PRD scopes "bypassing Spotify, YouTube, or other platform restrictions" out. An opt-in downloader for videos the user owns/licenses would be a user decision (not built). Player folder is ~118 MB (3 FLAC files).

## Website (2026-11-01)
`./serve.sh` -> http://127.0.0.1:8000 (README.md has settings and deployment notes). FastAPI (`backend/api/main.py`, `jobs.py`) + a no-build vanilla front end (`backend/api/static/`; deliberate deviation from the PRD's React choice: no toolchain to break, the player is already vanilla). Flow: raw-body upload with progress (decode check, 3 s minimum, size/length limits) -> options (separate drums, tempo, drum kit, required rights checkbox) -> job queue (one at a time, each job in its own subprocess `backend/worker.py`; stage progress; timeout; OOM-kill reported as a friendly failure) -> result page (summary chips, PDF/MusicXML/MIDI/JSON downloads, embedded player, notes/warnings) -> delete. Bookmarkable per-job addresses, "Your scores" list in localStorage, auto-delete after 72 h, restart recovery, Range requests for audio seeking, CSP and nosniff headers, path-traversal and bad-id tests, all user text rendered via textContent.
Verified: API tests (validation, security, full real job incl. PDF/MIDI/Range/delete, failure message, cleanup, restart); a REAL-BROWSER end-to-end test (scripts/ui_e2e.mjs, headless Chromium via DevTools; also in pytest) drives upload -> setup -> create -> progress -> result -> PDF download -> embedded player -> recent list -> delete: 16/16 on a synthetic clip, on the real 6.5-minute clip (40 s end to end) and with drum separation. It caught a real UI bug (result chips lost their labels), fixed.
NOT done / not verified: public deployment (no auth, no rate limiting, no HTTPS here: put behind a reverse proxy), Docker (untested, none provided), mobile layout (CSS is responsive but only checked at desktop width), audible playback in a real browser, concurrent heavy use (single worker by design), accessibility audit beyond keyboard/aria basics. LICENCE: the web service uses non-commercial models (ADTOF CC BY-NC-SA): a public commercial service needs permission or a replacement model.

## Website bug report: "something went wrong" (2026-11-01)
User's first real upload failed. Diagnosis from data/jobs/<id>/{job.json,worker.log}: the server had stored a **16,384-byte** file (size_mb 0.0) whose mp3 header still claimed 203.8 s, i.e. an incomplete copy reached the browser/server (likely a WhatsApp file still downloading/syncing; the cause on the user's side is unconfirmed). Three defects of mine: (1) the upload check trusted the header duration and accepted it; (2) the analysis then crashed in `classify_hats` (`min()` of an empty sequence) because a hi-hat sat at the very end of the audio and no ring-down could be measured; (3) the user saw only "Something went wrong". Fixes: upload now decodes the audio and rejects files whose readable length is far below the header's ("incomplete or damaged: says 6:34, only 0:00 readable"), uses the DECODED duration for limits, and rejects a body shorter than its Content-Length; the hi-hat step skips when nothing is measurable (fuzzing 360 awkward cases found only that one crash site); optional analysis failures (ghost/open/cymbal labels) no longer abort the job: the score is produced with plain notes and a note says so; `DRUMSCRIBE_DEBUG=1` shows the technical reason in the page. Verified: regression tests for each, a fresh server rejects a 16 KB prefix of a real WhatsApp file with the clear message and processes the complete file (184 bars, PDF 110 KB). The running server must be restarted to pick up the upload checks.

## Wrong tempo: 114 BPM song generated at 152 (2026-11-01)
Cause (measured on the user's song): 152/114 = 4/3. The song's kick recurs every 0.39 s = three 16ths (dotted eighths), and the beat tracker follows the loudest pulse of the mix. The other drums said 115: snare every 1.04 s (2 beats), kick+snare repeating every 2.09 s (one bar), hi-hat eighths at 115. New `backend/rhythm/tempo.py`: every instrument votes through its onset autocorrelation at beat / 2-beat / BAR (double weight) / 2-bar lags, mild prior around 115, then a <1 % refinement by 16th-grid coherence of the hits (skipped when the hits are not locked to a steady grid, R < 0.25). Result: 152 -> 115.0 (and 110 clip stays 110.0; synthetic 80/140 BPM, which the old method read as 161/70, are exact). Finding: the audio is locked to **115.0 BPM** (grid coherence 0.976 at 115.0 vs 0.061 at 114.0, stable in every 50 s window), not 114, so the user's tempo-114 score drifted ~1.8 s against the audio. `snap_hint`: a typed tempo within 3 % that the hits clearly do not lock to (and a nearby one they do) is corrected, with a note; far-off or unlockable tempos are never overridden; hint+offset grids stay exact. The result page offers the usual confusions (x 3/4, 4/3, 1/2, 2 ...) as one-click "Tempo not right? Try ..." buttons that re-create the score. Tests: dotted-eighth fixture, 6 grooves, live jitter, too few hits, snap rules, pipeline at 90/140 (13 tests) + browser flow. Limits: 4/4 assumed; octave ambiguity remains (synthetic 160 read as 80; 160 is in the alternatives); a steady-tempo assumption (a drummer who speeds up gets a coarse tempo only); evaluated on 2 real songs + synthetic.

## Accuracy round: hi-hat 8ths vs 16ths, groove dropouts (2026-11-01)
User: on the 115-BPM song the hi-hats are mostly eighths but the PDF showed sixteenths everywhere. Measured: kick/snare grooves consistent (`K.K.|S..S|..K.|SK..`); the detector's hi-hat line had hats at slots with ~no energy, dropouts, snare bleed counted as hat, and every faint 16th tick printed. The audio showed eighths 1.4-1.6x louder than the in-between sixteenths in bars 63-94 and genuinely flat sixteenths in bars 47-55. `rhythm/hats.py` rebuilds the hi-hat line from 6-16 kHz onset energy at every grid slot (neighbour-bar voting, detector prior required before adding, decaying-tail rule, in-between 16ths dropped where < 0.8x the 8th level; `--hats auto|eighths|sixteenths`, also in the web UI); only when hits lock to a steady grid. `rhythm/fill.py` fills kick/snare dropouts only when pattern and band energy agree. Dropped detections stay in the JSON (notated=False + reason). On the song: intro hats 22 -> 11, bars as eighths 28 -> 52, bars 73-94 clean eighths, flat-16th section kept; kick +30, snare +26. Synthetic ground-truth tests; not verified against a real answer key.

## Cymbal types and tambourine (2026-11-01)
User: ride/crash/splash not told apart; hi-hat confused with tambourine/jingles/sleigh bells. Measured: the song's hi-hat hits form no second timbre group; cymbal hits split into ~30 tonal/darker (ride-like) vs ~62 noisy/bright (crash-like). ADTOF has no such classes. Existing work: ADT_STR (26 classes incl. all of these) is the best candidate; DrummerNet (classes, no weights); MDX23C DrumSep / LarsNet (ride/crash stems, non-commercial/unlicensed); drum-expander/drumie (ride vs crash by ring length, no tambourine). ADT_STR downloaded and reviewed; running it needs the user's permission (MODEL_REGISTRY). Next: run it on the song, then measure ADTOF vs ADT_STR per class on a small real labelled set (MDB Drums) before wiring it in.

## REAL benchmark: MDB Drums, 23 hand-annotated songs (2026-11-01)
`scripts/eval_mdb.py` (full mixes only, 231 MB of the 2 GB set, data/benchmarks/mdb, CC BY-NC-SA, evaluation only). F1 at +-50 ms:
| class | detector (ADTOF) | pipeline |
|---|---|---|
| kick [1539] | 0.848 | 0.847 |
| snare [2692, incl. 790 ghost] | 0.711 | 0.711 |
| tom [90] | 0.345 | 0.340 |
| hi-hat [2639] | 0.849 | 0.855 with the rebuild OFF (0.717 with it on) |
| cymbal [1002] | 0.834 | 0.835 |
Fine labels (pipeline heuristics): hat closed 0.645, hat open 0.433, ride 0.578, crash 0.251, china 0, splash 0, ghost 0.157, tambourine 0 (not supported). Cymbal type correct for 54 % of detected cymbals (379 of 751 rides called crash).
**Consequences (measured, not assumed):** the hi-hat rebuild (rhythm/hats.py) lowered hi-hat F1 in every variant (0.821-0.842 vs 0.855; the quiet-16th rule deleted real hats, Disco 0.96 -> 0.58), so it is OFF by default and only "Simplify to eighths" (prune-only) uses it. Groove fill: kick recall 0.865 -> 0.886 (F1 +0.002), snare no gain -> kick only. Onset refinement helps hi-hat (0.849 -> 0.855). On the user's 115-BPM song the quiet in-between 16ths are real sounds, plausibly tambourine/shaker (user report) rather than hat: a classification problem.
**ADT_STR:** executes (user granted permission) but its greedy decoder has no key/value cache: >12x slower than real time here (40+ min for a 3.4-min song). `backend/detection/adtstr.py` adds a cached decoder: 14x faster, but tokens are NOT identical to the original, because the original decoder's logits for a position change with sequence length (max diff 2.85), i.e. it does not behave as a strict causal decoder. Diagnosing that (PyTorch fast path?) was blocked by the permission classifier. Not wired into the pipeline; ADT_STR not yet benchmarked on MDB.

## Learned labels + ADT_STR verdict (2026-11-01)
ADT_STR rejected after evaluation (masking bug; mislabels hi-hats on real mixes and drum stems; MODEL_REGISTRY). Instead `labels.py` learns ride/crash and open/closed hat from MDB Drums: leave-one-song-out ride/crash accuracy 0.55 -> 0.90, open-hat F1 0.56 -> 0.68. Wired into the pipeline (and so the website); falls back to the heuristics if the model file is absent. Website: hi-hat option is now "As played" / "Simplify to eighths"; result note states the measured accuracies. Real-browser website test with the user's 115-BPM song: all checks pass. On that song the model calls all 92 cymbals crash and finds 77 open hats (UNVERIFIED by ear). 87 tests pass.

## Known limitations / failures found
- **Unhinted tempo is weak:** 90→89.1, 100→99.4, 120→117.5 BPM (≈2 % off) and 140→69.8 (half-time). Half/double are listed as unresolved alternatives; M1 does not choose. Use `--tempo` until M3. (Synthetic input only.)
- Meter fixed to N/4 (assumed); downbeat not detected (hint offset = beat 1; unhinted: bar 1 starts at first grid beat ≤ 0).
- 16th-note grid only: triplets/swing/fast fills will be mis-snapped until M3.
- ADTOF: 5 classes; stacked-snare recall limited; confidence is raw activation (not calibrated); upstream first-frame bug worked around; weights NON-COMMERCIAL.
- Whole file loaded and run in one pass (BiGRU over all frames): fine for 60–120 s clips, untested for long tracks (M6). ~3 GiB free RAM.
- Note durations: largest aligned value up to next onset + rests; no ties/dots; kick = voice 2. Ghost notes/accents not notated.
- PDF validation = valid file + pages > 0 + page 1 non-blank; per-bar coverage verified only at MusicXML level. Multi-page layouts have not been inspected (only 1-page synthetic output).

## Blocked / needs user
1. **Real data (Step 4):** the supplied clip has no ground truth and is 395 s (PRD acceptance wants 60–120 s with annotations). Need a licensed 60–120 s full-mix clip with kick/snare ground truth (or tell me to fetch ENST / IDMT-SMT-Drums / Slakh and build one). Place as `data/benchmarks/C/<name>/audio.wav` + `truth.json` (`{"kick":[sec,..],"snare":[...],"hihat":[...]}`), then run `python -m backend.pipeline <audio> --tempo <bpm>` and `python scripts/run_benchmarks.py`. Acceptance criteria (PRD §15) cannot be claimed until this passes on a held-out clip.
2. Licence: ADTOF CC BY-NC-SA, prototype only (user confirmed).

## Next three actions
1. Real clip + truth → run M1 acceptance test; fix whatever fails (likely tempo/downbeat and tuning).
2. Flams: need user-named bars/times (detector sees none even on the stem). Then calibrate ghost/open-hat/cymbal thresholds on feedback.
3. M3: tempo/meter alternatives, swing, tuplet protection.

## M0 completion report (PRD 18.2)
- Verified: environment audit, CPU venv, decode, MuseScore render, setup.sh (rerun). Unverified: setup.ps1, Docker, fresh-venv setup. Blocked: real-mix acceptance, production model approval.

## Time signatures + tambourine (2026-11-02)
User: on one song the hi-hat 8ths/16ths were still wrong; learn splash/china/tambourine from any online resource.
**8ths/16ths cause (measured): the song ("Yeh Fitoor Mera") is in 6/8.** Its hits lock to quarter = 82 sixteenths
(coherence 0.94) and repeat every 12 slots with the snare on the 4th eighth; the pipeline assumed 4/4 and scored it at
109.5 BPM (82 x 4/3, lock 0.11), so every hat sat on a wrong slot. New: triplet relock, 12- vs 16-slot recurrence,
6/8 / 12/8 (shuffle) / 3/4 by snare placement, exact grid from the found downbeat; MusicXML x/8 bars (dotted-quarter
marking, beat groups), player, validator; website "Time signature" option + chip; CLI `--meter`. Result on the song:
6/8, hats on eighths (16ths rare). MDB: no F1 change; Rockabilly + CoolJazz (triplet-locked in the hand annotations)
-> 12/8, 21 songs stay 4/4, SwingJazz (also swung) stays 4/4 (miss). User's 115 and 110 BPM songs stay 4/4.
**Splash/china/tambourine:** real labelled data is tiny (MDB: 10 / 15 / 32, tambourine all in one song). Fetched real
one-shots (CrocellKit via HTTP range reads of 151 hits out of a 5.5 GB zip, VSCO-2-CE, 3 soundfonts), mixed them into
the real MDB songs, retrained. Tambourine-vs-hat: 0 -> 7 of 21 real hits found (precision 0.54) -> IN USE. Splash/china:
still not found on real songs (works on synthetic only) -> NOT used. See MODEL_REGISTRY. Full benchmark (labels now
in-sample): hi-hat F1 0.850 -> 0.856, open hat 0.817 -> 0.829, tambourine 0 -> 0.64 (honest out-of-song figure 0.41).
**Bug found and fixed:** the website's ghost/open/crash/ride counts were the heuristic counts from BEFORE the learned
relabelling (so "all 92 cymbals crash" on the 115 song was wrong: the model says 47 ride / 45 crash). Also fixed:
training features were contaminated by the installed model; cymbal relabelling changed hi-hat features.
On the user's songs (UNVERIFIED by ear): 115-BPM song 100 of 802 hats -> tambourine, mostly in-between 16ths;
Fitoor 31 tambourine, mostly on the 4th eighth with the snare. 93 tests pass; browser e2e with Fitoor passes.

## Precision round (2026-11-02, later)
User: the WhatsApp song ("Mujhko Pehchaan Lo", 3.3 MB, the "115-BPM song") still shows 16th hats where 8ths are
played; aim is precision (no notes that are not there). All numbers: MDB Drums, 23 real songs, +-50 ms.
- **Tried and rejected: "darker in-between 16ths are a shaker" rule.** On the song, the in-between 16ths are 3-5 dB
  darker (6-16 kHz vs 2-5 kHz) than the hat eighths. But Disco's REAL soft 16th hats are 3-4 dB darker too: the rule
  cost hi-hat F1 0.856 -> 0.816 (169 real hats removed from Disco alone). No threshold separated the two. Reverted.
- **Tried and rejected: shaker/maracas/cabasa samples (GM soundfonts) in the hat/tambourine model.** LOSO on real hits
  unchanged (tambourine 6/21 vs 7/21, same 6 hats mis-moved); the song's tambourine count fell 100 -> 66. No measured
  gain, so the previous model stays.
- **Drum separation benchmarked on real songs for the first time** (Demucs htdemucs, stems in data/stems/mdb): wrong
  notes 1235 -> 969, missed 1879 -> 1683, kick F1 0.849 -> 0.918, snare 0.711 -> 0.757; hi-hat got MORE false hits.
  Now ON BY DEFAULT on the website (CLI still needs --separate). Matches drum2midi's whole-song figure (micro F1 0.834).
- **Labels on separated audio**: the learned labels were computed on the stem but trained on mixes (open-hat
  precision 0.92 -> 0.76, crash recall 0.84 -> 0.47, tambourine precision 0.14). Now the label features always come
  from the mix: open 0.88, crash recall 0.75, tambourine 0.73 (in-sample: the label model was trained on MDB).
- **Confidence cut-offs** (`MIN_CONFIDENCE`), chosen leave-one-song-out, only those picked on every held-out song:
  mix: tom 0.65 (wrong toms 183 -> 40). Stem: tom 0.65, hi-hat 0.4, snare 0.3 (wrong hats 349 -> 223, snares 229 ->
  176, toms 75 -> 12; F1 equal or better). Kick/cymbal cut-offs gave nothing out-of-sample.
- **Result (website default):** wrong notes 1235 -> 724, missed 1879 -> 1718, precision 0.83 -> 0.90, F1 0.80 -> 0.84.
- Fixes: report/website counts now count only printed notes (hidden toms etc. were counted); "eighths" mode reported
  recovered_from_audio = -52.
- On the WhatsApp song (UNVERIFIED by ear): hats 702 -> 505, almost every bar 8 eighths + 1-4 in-between 16ths; the dark
  16th run in bars 47-69 is gone; toms 100 -> 5; tambourine 100 -> 35; still 162 open hats (looks high). "Simplify to
  eighths" gives pure 8ths on this song.
- Ghost-note marks: precision 0.25 (mix) / 0.41 (stem): mostly wrong. Left on pending the user's decision.
- Researched: drum2midi (ADTOF + DrumSep for articulation; tom-stem confirmation overfit MDB, +0.006 on ENST),
  adtof_plus_drum_transcription, akaihola/drum-transcribe (DrumSep: too many hits, ~10x real time on CPU), STAR Drums
  (181 GB, synthesized drums), LarsNet (CC BY-NC). Nothing adopted beyond the separation finding.

## Clarity round (2026-11-03)
User: "improve all aspects to achieve utmost clarity". MDB Drums numbers, 23 songs, +-50 ms.
- **Ghost notes no longer marked or added.** Ghost marks: at most 44 % right (stem) / 33 % (mix) at ANY
  ghost_probability threshold (swept 0.5-1.0), so a bracket misleads more often than it helps. The low-threshold
  "ghost candidate" snares were 63/139 right on the mix (14/21 on the stem): removed. ghost_probability stays in the JSON.
  Website default: wrong 724 -> 717, missed 1718 -> 1732. Without separation: wrong 1095 -> 1019, missed 1888 -> 1951.
- **Drum-style durations in the score**: a note lasts to the next onset within its beat, so dotted 8th + 16th and
  16th-8th-16th replace "8th, 16th rest, 16th" (4/4, 3/4, 2/4; 6/8 and 12/8 unchanged). Checked on the WhatsApp
  song's PDF: far fewer rests and flags; validate_pdf OK.
- Checked and NOT changed: open hi-hat marks with little room before the next hat (song bars 29-32 alternate open/closed
  16ths). On MDB short-room open marks were 7/7 (mix) and 3/3 (stem) right, all open marks ~90 %: no evidence for a rule.
  The song's 162 open hats still need checking by ear.
- Text fixes: website note and README no longer say flams are undetected (they are marked; unverified on real audio).

## Guide + hosting files (2026-11-04)
- docs/GUIDE.md: complete step-by-step guide (install, website, CLI, score key, pipeline stages, accuracy, tests,
  hosting, licences, troubleshooting, project layout). README links to it.
- Dockerfile + .dockerignore for Hugging Face Spaces / any Docker host: NOT BUILT OR TESTED (Docker daemon not
  running here, only 4 GB disk free). Verified only that its Debian trixie packages and pinned CPU torch wheels exist.
- The user's suggested free-for-dev list (jixserver/free-for-dev) is a 2016-era copy of the list; nothing on it can
  run this app. Recommended a PRIVATE Hugging Face Space: ADTOF-pytorch still has no licence file (checked 2026-11-04).
- Website separation note corrected: 30 % fewer wrong notes, 11 % fewer missed (was "17 % / 10 %", outdated).
