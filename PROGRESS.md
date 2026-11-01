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

`pytest backend/tests -q`: 56 passed. `ruff check backend scripts`: clean.

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
