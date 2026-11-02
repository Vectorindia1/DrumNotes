# DrumScribe

Upload a recording, get drum sheet music (PDF, MusicXML, MIDI, JSON) and a player where a blue line follows the score
while it plays. Research prototype: transcriptions are drafts to check by ear.

## Run the website

```bash
./setup.sh      # once: CPU environment, models, checks
./serve.sh      # http://127.0.0.1:8000   (HOST=0.0.0.0 ./serve.sh to share on your network)
```

Open the address, drop an audio file, tick the rights box, press **Create score**. "Separate the drums first" is on
by default because it prints far fewer wrong notes (see Accuracy); a 3.5-minute song then takes about 6 minutes on this
machine (untick it for about a minute). Your scores are listed on the home page
(saved in your browser) and every score has its own bookmarkable address.

From the command line: `python -m backend.pipeline song.mp3 --tempo 110 --separate --player`.

## Settings (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `DRUMSCRIBE_MAX_UPLOAD_MB` | 200 | largest upload |
| `DRUMSCRIBE_MAX_MINUTES` | 12 | longest recording |
| `DRUMSCRIBE_MAX_SEPARATE_MINUTES` | 8 | longest recording allowed with drum separation |
| `DRUMSCRIBE_RETENTION_HOURS` | 72 | files are deleted this long after the last activity |
| `DRUMSCRIBE_JOB_TIMEOUT_MIN` | 60 | a job is stopped after this long |
| `DRUMSCRIBE_MAX_QUEUE` | 20 | jobs waiting + running before "queue is full" |
| `DRUMSCRIBE_DATA` | `data/jobs` | where uploads and results are stored |

## How it behaves

- One score is created at a time; the rest wait in line. Each runs in its own process, so an out-of-memory crash
  fails that one job, not the site.
- Uploads are checked on arrival (decodable, 3 s minimum, size and length limits). Any common format works (mp3, wav,
  flac, m4a, ogg, opus). YouTube and Spotify links are not supported on purpose: their audio is not available to
  third parties.
- Privacy: a score is reachable only through its unguessable address; there are no accounts and no public listing.
  "Delete my files" removes everything immediately; everything else is deleted after the retention time.

## Before putting it on the public internet

This is a prototype. There is **no login and no rate limiting**: put it behind a reverse proxy (HTTPS, per-IP limits,
upload size/time limits) and keep the limits above low. The drum-detection model (ADTOF) is licensed
**CC BY-NC-SA 4.0, non-commercial only**: a public service that earns money needs permission from its authors or a
different model (see `LICENSE_AUDIT.md`). Demucs weights' licence is unverified. No Dockerfile is provided
(Docker was not available to test it).

## Tempo

With no tempo entered, the tempo is worked out from the drum pattern (all instruments, bar-level repetition), not from
the loudest pulse in the mix. The result page lists the usual mix-ups (x 3/4, 4/3, 1/2, 2) as buttons: click one to
re-create the score. Hi-hat: "As played" (default) or "Simplify to eighths". If you type a tempo that is slightly off from what the drums lock to (within 3 %), the tempo
the audio supports is used and the page tells you. Assumes a steady tempo.

Time signature: detected from the drum pattern (4/4, 3/4, 6/8, or 12/8 for shuffles and swing). A 12-step bar is used only
when kick and snare clearly repeat every 12 steps rather than every 16, and where the snare falls decides between 6/8, 12/8 and 3/4.
Choose it under More options (CLI `--meter`) if the bars look wrong. On MDB Drums, 21 songs stay 4/4; the two whose hand
annotations sit on triplets (Rockabilly, CoolJazz) become 12/8; SwingJazz, also swung, stays 4/4 (missed).

## Accuracy (measured on 23 real, hand-annotated songs: MDB Drums)

All hits, +-50 ms, website default (drums separated first, low-confidence hits left out):

| | wrong notes printed | missed | precision | recall | F1 |
|---|---|---|---|---|---|
| default (separated) | 717 | 1732 | 0.90 | 0.78 | 0.84 |
| without separation | 1019 | 1951 | 0.86 | 0.76 | 0.80 |
| before 2026-11-02 (no separation, no cut-offs) | 1235 | 1879 | 0.83 | 0.76 | 0.80 |

Per instrument (default): kick 0.92, snare 0.76 (quiet ghost notes are mostly missed), hi-hat 0.87, cymbal 0.84, toms 0.51.
The confidence cut-offs (`MIN_CONFIDENCE` in backend/pipeline.py) were chosen leave-one-song-out on these same songs;
nothing has been checked on a second dataset. Ghost notes are NOT marked in the score: the marks were right at most
44 % of the time at any threshold, and the extra quiet snares added for them were mostly not real (76 of 139 without
separation). Their loudness score stays in the JSON (`ghost_probability`).
Labels: ride vs crash correct about 90 % of the time and open vs closed hi-hat about 94 %, measured on songs the
label model had not seen (`scripts/train_labels.py`, leave-one-song-out). Tambourine / jingles played like a hi-hat:
about a third found, about half of the marks right (few real examples; `scripts/fetch_samples.py` then
`scripts/train_cymbal_types.py`). Splash and china are rarely marked; flam marks have not been checked on real recordings. Re-run: `.venv/bin/python scripts/fetch_mdb.py && PYTHONPATH=. .venv/bin/python scripts/eval_mdb.py`.
The label model is trained on non-commercial data (CC BY-NC-SA), like the drum detector itself.

## If something goes wrong

- The page shows a short message. For the technical reason, restart with `DRUMSCRIBE_DEBUG=1 ./serve.sh`: failed
  scores then show the underlying error. The full log of every score is in `data/jobs/<id>/worker.log`.
- "That file looks incomplete or damaged": the upload was shorter than the audio inside it claims (typical for a file
  that is still downloading or syncing). Check the file's size on disk, copy it again, and re-upload.
- After updating the code, restart `./serve.sh` (the analysis picks up changes by itself, the web server does not).

## Tests

```bash
.venv/bin/pytest backend/tests -q     # includes a real-browser end-to-end test when node + chromium exist
```
