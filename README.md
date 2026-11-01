# DrumScribe

Upload a recording, get drum sheet music (PDF, MusicXML, MIDI, JSON) and a player where a blue line follows the score
while it plays. Research prototype: transcriptions are drafts to check by ear.

## Run the website

```bash
./setup.sh      # once: CPU environment, models, checks
./serve.sh      # http://127.0.0.1:8000   (HOST=0.0.0.0 ./serve.sh to share on your network)
```

Open the address, drop an audio file, tick the rights box, press **Create score**. A 6-minute song takes about a minute
(about 13 minutes with "Separate the drums first" on this machine). Your scores are listed on the home page
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
re-create the score. If you type a tempo that is slightly off from what the drums lock to (within 3 %), the tempo
the audio supports is used and the page tells you. Assumes 4/4 and a steady tempo.

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
