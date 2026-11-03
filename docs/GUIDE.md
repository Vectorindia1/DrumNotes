# DrumScribe: the complete step-by-step guide

DrumScribe listens to a song and writes out the drum part as sheet music. You give it an audio file (mp3, wav, m4a and
so on) and it gives back:

- a **PDF** of the drum score, ready to print,
- a **MusicXML** file that opens in MuseScore, Sibelius, Finale or Dorico if you want to edit the score,
- a **MIDI** file with every hit at the exact time it was played,
- a **JSON** file listing every hit it found, including the ones it left out of the score and why,
- a **player** web page where a blue line follows the score while the song plays.

It is a research prototype. The score is a good first draft, not a finished transcription: always check it by ear.

Contents

1. [What you need](#1-what-you-need)
2. [Install it, step by step](#2-install-it-step-by-step)
3. [Use the website, step by step](#3-use-the-website-step-by-step)
4. [Use the command line](#4-use-the-command-line)
5. [How to read the score](#5-how-to-read-the-score)
6. [What happens inside, step by step](#6-what-happens-inside-step-by-step)
7. [How accurate it is](#7-how-accurate-it-is)
8. [Check that everything works](#8-check-that-everything-works)
9. [Put it on the internet](#9-put-it-on-the-internet)
10. [Licences: what you may and may not do](#10-licences-what-you-may-and-may-not-do)
11. [Troubleshooting](#11-troubleshooting)
12. [Where everything is in the project](#12-where-everything-is-in-the-project)

---

## 1. What you need

| | Needed | Notes |
|---|---|---|
| Computer | Linux (tested on Kali / Debian) | Windows: `setup.ps1` exists but is less tested. No graphics card needed: everything runs on the CPU. |
| Memory | 8 GB RAM or more | Drum separation is the hungriest step. |
| Disk | about 15 GB free | The Python packages alone are about 5.5 GB. |
| Time | 3.5-minute song: about 6 minutes with drum separation, about 1 minute without | Measured on an 8-core laptop CPU. |

Software (the install steps below put all of it in place):

- **Python 3.12** and **uv** (a fast Python package installer),
- **ffmpeg** (reads every audio format),
- **MuseScore 3** (`mscore3`, turns the score into a PDF),
- **a drum soundfont + fluidsynth** (the sound of the playback kit),
- **git**.

---

## 2. Install it, step by step

Run these commands in a terminal. Lines starting with `#` are explanations, not commands.

**Step 1. Install the system programs** (Debian, Ubuntu, Kali):

```bash
sudo apt update
sudo apt install -y git ffmpeg musescore3 musescore-general-soundfont-small libfluidsynth3
curl -LsSf https://astral.sh/uv/install.sh | sh     # installs uv; open a new terminal afterwards
```

**Step 2. Get the code:**

```bash
git clone https://github.com/Vectorindia1/DrumNotes.git
cd DrumNotes
```

**Step 3. Get the drum detector (ADTOF-pytorch).** It is not stored in this repository because of its licence
(see section 10). Download it into `third_party/` at the exact version this project was tested with:

```bash
git clone https://github.com/xavriley/ADTOF-pytorch third_party/ADTOF-pytorch
git -C third_party/ADTOF-pytorch checkout 85c192e78f716ea0b111cc8a5ee4a8f6a3a4f8a9
```

**Step 4. Run the setup script.** It creates a private Python environment in `.venv/`, installs the CPU version of
PyTorch (the normal one from PyPI is a 2+ GB graphics-card build you do not need) and the other packages, installs
ADTOF, and checks every part:

```bash
./setup.sh
```

At the end it prints `[OK]` / `[WARN]` / `[FAIL]` for every part and `SETUP SUCCESS` when everything required is there.
Warnings are optional parts, explained in the next two steps.

To install the exact package versions this project was last tested with instead, use the lock file:

```bash
uv pip install --python .venv/bin/python --extra-index-url https://download.pytorch.org/whl/cpu \
    -r <(grep -v '^adtof-pytorch' requirements-lock.txt)
```

**Step 5. Install drum separation (Demucs).** Strongly recommended: it is what removes most wrong notes.

```bash
uv pip install --python .venv/bin/python demucs
.venv/bin/python scripts/download_models.py     # downloads the separation model once (about 80 MB)
```

**Step 6. Train the label model** (open vs closed hi-hat, ride vs crash, tambourine, splash, china). Without it the
program still works but falls back to simpler rules for these labels. This downloads the MDB Drums dataset (about
230 MB, non-commercial licence) and trains on it (about 10 minutes):

```bash
.venv/bin/python scripts/fetch_mdb.py
PYTHONPATH=. .venv/bin/python scripts/train_labels.py           # ride/crash + open/closed hi-hat
PYTHONPATH=. .venv/bin/python scripts/fetch_samples.py          # free one-shot samples (a few hundred MB)
PYTHONPATH=. .venv/bin/python scripts/train_cymbal_types.py     # adds tambourine, splash, china
```

The result is one file, `models/weights/labels.joblib` (about 2.4 MB). It is deliberately not uploaded to GitHub.

**Step 7. Check the installation:**

```bash
.venv/bin/python scripts/check_environment.py
```

---

## 3. Use the website, step by step

**Step 1. Start it:**

```bash
./serve.sh
```

Leave this terminal open. The website now runs at **http://127.0.0.1:8000** on this computer only. To open it from
your phone or another computer on the same Wi-Fi, start it with `HOST=0.0.0.0 ./serve.sh` and open
`http://<this computer's IP address>:8000`. Press `Ctrl+C` to stop it.

**Step 2. Upload a song.** Drag an audio file onto the page or click to choose one. Any common format works. The
page shows the limits (by default up to 12 minutes and 200 MB). YouTube and Spotify links are refused on purpose:
their audio is not available to other programs.

**Step 3. Choose the settings.**

| Setting | What it does | When to change it |
|---|---|---|
| **Separate the drums first** (on) | Pulls the drums out of the mix before listening for hits. About 30 % fewer wrong notes and 11 % fewer missed hits on real songs. | Turn off only if you need the result fast. Not offered for recordings over 8 minutes. |
| **Tempo (BPM)** (empty) | Empty = worked out from the drums. | Type the tempo if you know it. This is the single biggest help for a correct score. |
| **Time signature** (Detect automatically) | 4/4, 3/4, 6/8 or 12/8 (shuffle / swing), detected from the drum pattern. | If the bar lines look wrong. |
| **Hi-hat detail** (As played) | "Simplify to eighths" drops quiet in-between 16th notes on the hi-hat. | If the hi-hat line shows 16ths where you hear 8ths. Off by default because on some songs it deletes real 16ths. |
| **Drum sound for playback** (Standard) | Which drum kit plays the score in the player. | Taste only; it does not change the score. |

**Step 4. Confirm you have the right to use the recording** (tick the box), then press **Create score**. A progress
page shows each stage. You can leave the page: the score keeps being made, and its address is bookmarkable.

**Step 5. Read the result page.**

- The **chips** at the top: tempo, number of bars, time signature, number of hits, length.
- **Other tempos**: if the tempo was detected, buttons such as "57.5 BPM" or "230 BPM" re-create the score at the
  usual mix-ups (half, double, 3/4, 4/3). Use them if the score feels twice too fast or too slow.
- **Downloads**: PDF, MusicXML, MIDI, JSON.
- **The player**: press play; the blue line follows the score. You can switch between the original recording, the
  separated drum track and the score played by the sampled kit.
- **Notes**: what to double-check for this song (for example, if the bar lines are a best guess).

**Step 6. Your scores** are listed on the home page (remembered by your browser). **Delete my files** removes the
upload and every result immediately; otherwise everything is deleted 72 hours after the last activity.

**Server settings** (environment variables, set them before `./serve.sh`, for example
`DRUMSCRIBE_MAX_MINUTES=6 ./serve.sh`):

| Variable | Default | Meaning |
|---|---|---|
| `PORT` / `HOST` | 8000 / 127.0.0.1 | where the website listens |
| `DRUMSCRIBE_MAX_UPLOAD_MB` | 200 | largest upload |
| `DRUMSCRIBE_MAX_MINUTES` | 12 | longest recording |
| `DRUMSCRIBE_MAX_SEPARATE_MINUTES` | 8 | longest recording allowed with drum separation |
| `DRUMSCRIBE_RETENTION_HOURS` | 72 | files are deleted this long after the last activity |
| `DRUMSCRIBE_JOB_TIMEOUT_MIN` | 60 | a job is stopped after this long |
| `DRUMSCRIBE_MAX_QUEUE` | 20 | jobs waiting + running before "queue is full" |
| `DRUMSCRIBE_DATA` | `data/jobs` | where uploads and results are stored |

One score is made at a time; the others wait in line. Each one runs in its own process, so a crash only fails that
one score, never the website.

---

## 4. Use the command line

The same pipeline without the website:

```bash
.venv/bin/python -m backend.pipeline song.mp3 --tempo 115 --separate --player
```

Results go to `outputs/` (`outputs/pdf`, `outputs/musicxml`, `outputs/midi`, `outputs/json`, `outputs/player`).

| Option | Meaning |
|---|---|
| `--tempo 115` | the tempo in BPM. Recommended: without it the tempo is guessed and can be off. |
| `--separate` | separate the drums first (the website does this by default; the command line does not). |
| `--offset 0.42` | time in seconds of the first beat 1. With `--tempo` this makes the grid exact. |
| `--beat1 2` | which beat (0-3) is beat 1, if the bar lines are shifted. |
| `--meter 6/8` | force the time signature (`auto`, `4/4`, `3/4`, `6/8`, `12/8`). |
| `--hats eighths` | simplify the hi-hat to eighths. |
| `--player` | also build the blue-playhead player. |
| `--kit jazz` | playback kit: standard, room, power, electronic, 808, jazz, brush. |
| `--out folder` | where to write the results (default `outputs`). |

The audio can also be a direct link to an audio file (`https://.../song.mp3`).

---

## 5. How to read the score

Two voices on one five-line staff: hands with stems up, kick with stems down.

| Symbol | Instrument |
|---|---|
| note in the bottom space | kick (bass drum) |
| note in the third space | snare |
| small note just before a snare note | flam (grace note) |
| note on the fourth line | tom |
| x above the top line | closed hi-hat |
| circled x above the top line | open hi-hat |
| x on the top line | ride |
| x on a ledger line above the staff | crash |
| x in the fourth space, marked "tamb." | tambourine / jingles |
| down-pointing triangle, marked "splash" | splash |
| up-pointing triangle, marked "china" | china |

Rhythm is written the way drum parts usually are: a note lasts until the next note in the same beat, so you see
"dotted 8th + 16th" and "16th, 8th, 16th" instead of notes separated by small rests.

**Not marked in the score:** ghost notes (very quiet snare hits). In testing, the brackets were wrong more often than
right, so all snare hits are printed the same way. Play quiet snares by ear. The JSON still has a
`ghost_probability` for every snare hit.

---

## 6. What happens inside, step by step

The main file is `backend/pipeline.py`, function `run()`. In order:

1. **Read the audio** (`backend/intake.py`). Accepts a file or a direct audio link and refuses YouTube/Spotify.
   The audio is checked (it can be decoded, at least 3 seconds long, within the size and length limits).

2. **Separate the drums** (optional, on by default on the website; `backend/separation/demucs_sep.py`). The Demucs
   model (htdemucs) splits the song into drums, bass, vocals and other, and only the drum track is kept. This is
   the slowest step. The drum track is cached in `data/stems/`, so a second run on the same song skips it.

3. **Find the drum hits** (`backend/detection/adtof.py`). The ADTOF neural network listens in 10-millisecond steps
   and reports kick, snare, tom, hi-hat and cymbal hits, each with a confidence between 0 and 1.

4. **Look closer at each hit** (`backend/detection/articulation.py`). Using the sound energy in different frequency
   bands at 1.5-millisecond resolution, it:
   - moves each hit to the exact moment of its attack (the network's 10 ms steps are too coarse),
   - finds **flams** (a second, quieter snare onset 15 to 65 ms before a snare hit),
   - measures how loud each snare hit is compared with its neighbours (stored, not printed),
   - gives a first guess at open/closed hi-hat and cymbal type.

5. **Find the tempo, time signature and bar lines** (`backend/rhythm/grid.py`). The tempo comes from the repeating
   drum pattern (all instruments, bar by bar), not from the loudest pulse in the mix. The usual mix-ups (half,
   double, 3/4, 4/3 of the tempo) are kept as alternatives for the result page. The time signature comes from how
   often kick and snare repeat (every 16 or every 12 steps) and where the snare falls.

6. **Name the cymbals and hi-hats with the learned model** (`backend/detection/labels.py`, file
   `models/weights/labels.joblib`): ride or crash, open or closed hi-hat, tambourine, splash, china. It always
   listens to the original song, even when the hits were found on the separated drum track, because that is what it
   was trained on.

7. **Leave out unsure hits.** Hits below a confidence cut-off are hidden from the score (`MIN_CONFIDENCE` in
   `backend/pipeline.py`): toms below 0.65; with separation also hi-hats below 0.4 and snares below 0.3. The cut-offs
   were chosen on 23 real songs, and only those that helped on every song held out of the choice were kept. Hidden
   hits stay in the JSON with the reason.

8. **Snap the hits to the grid** (`backend/rhythm/grid.py`). Every hit is placed on the nearest 16th-note position
   (or 12/8 position). The original time is never overwritten: the JSON keeps both the real time and the distance
   to the grid position.

9. **Optional clean-ups.** "Simplify to eighths" (`backend/rhythm/hats.py`) drops quiet in-between hi-hat 16ths.
   When the playing sits tightly on the grid, a kick missing from an otherwise repeating groove is filled in, but
   only if the audio shows a kick there (`backend/rhythm/fill.py`).

10. **Build the bars** (`backend/reconstruction/bars.py`). Two hits of the same drum on the same grid position
    become one.

11. **Write the MusicXML** (`backend/notation/drum_musicxml.py`). The file is then validated: every bar must add up
    to exactly its length.

12. **Engrave the PDF** with MuseScore 3 (`backend/engraving/musescore.py`), then check the PDF (it opens, has pages,
    page 1 is not blank).

13. **Write the MIDI** (at the real, un-snapped times) **and the JSON** (`backend/exports.py`).

14. **Build the player** (`backend/player/`): MuseScore lays out the score as images, the score is played by a
    sampled drum kit (fluidsynth + soundfont), and the blue line is timed from MuseScore's own measure positions.

15. **Write the report** (`outputs/json/<song>.report.json`): tempo and alternatives, counts, what was hidden and
    why, validation results.

The website (`backend/api/`) wraps this: `main.py` serves the pages, `jobs.py` keeps the queue and the limits, and
`backend/worker.py` runs one job in its own process and turns the report into the notes on the result page.

---

## 7. How accurate it is

Measured on **MDB Drums**: 23 real songs whose drums were annotated by hand. A hit counts as right when it is within
50 milliseconds of a hand-annotated hit of the same drum.

| Website default (drums separated) | |
|---|---|
| Printed notes that are right (precision) | 90 % |
| Real hits that are found (recall) | 78 % |
| Wrong notes printed, all 23 songs | 717 |
| Real hits missed, all 23 songs | 1732 |

Per drum (F1, a combined score of precision and recall): kick 0.92, snare 0.76, hi-hat 0.87, cymbal 0.84, toms 0.51.
Open vs closed hi-hat is right about 94 % of the time and ride vs crash about 90 %, on songs the model had not seen.

What these numbers do **not** tell you:

- They come from one set of 23 songs. Some settings were chosen on the same songs, so results on other music,
  especially very different styles, can be worse.
- Toms are the weakest instrument. Quiet ghost notes are mostly missed.
- It assumes a steady tempo. Songs that speed up or slow down drift off the grid.
- Swing and triplets are only handled as 12/8. Tuplets inside a 4/4 bar are not written.

Full history of every measurement and every rejected idea: `PROGRESS.md`. Re-run the benchmark yourself:

```bash
.venv/bin/python scripts/fetch_mdb.py
PYTHONPATH=. .venv/bin/python scripts/eval_mdb.py
```

---

## 8. Check that everything works

```bash
.venv/bin/pytest backend/tests -q                 # 94 automatic tests
.venv/bin/ruff check backend scripts              # code style
PYTHONPATH=. .venv/bin/python scripts/run_benchmarks.py     # synthetic songs with known answers
```

Browser test of the whole website (needs Node.js 22 or newer and Chromium; the audio path must be absolute):

```bash
PORT=8811 ./serve.sh &
node scripts/ui_e2e.mjs http://127.0.0.1:8811 /full/path/to/song.wav /tmp/shots
```

---

## 9. Put it on the internet

> **Not tested yet.** The `Dockerfile` has not been built: Docker was not running on the development machine and
> there was not enough free disk. Expect to fix small things on the first build.

Before you do this, read section 10. A **private** deployment (only you can open it) avoids the licence problems
of a public one.

The website has **no login and no rate limiting**. Anyone with the address can upload songs and use your CPU. For a
public address, put it behind a reverse proxy with HTTPS and per-IP limits, and keep the upload limits low.

### Option A: Hugging Face Spaces (free)

The free "CPU basic" Space has 2 CPU cores, 16 GB memory and 50 GB disk. It sleeps after about 48 hours without
visitors and wakes up when someone opens it. Files are not kept when it restarts. With only 2 cores, drum
separation will be several times slower than on an 8-core computer.

1. Make a free account at https://huggingface.co.
2. Create an access token with **write** permission at https://huggingface.co/settings/tokens, then sign in:
   ```bash
   .venv/bin/hf auth login
   ```
3. Create a private Docker Space (replace `YOURNAME`):
   ```bash
   .venv/bin/hf repos create YOURNAME/DrumScribe --repo-type space --space-sdk docker --private
   ```
4. Upload the code and the label model, leaving out the large local folders:
   ```bash
   .venv/bin/hf upload YOURNAME/DrumScribe . . --repo-type space \
       --exclude ".venv/*" --exclude "data/*" --exclude "outputs/*" --exclude "third_party/*" \
       --exclude "audio file/*" --exclude ".git/*" --exclude "*.pdf"
   ```
5. Open `https://huggingface.co/spaces/YOURNAME/DrumScribe`. The build (installing everything and downloading the
   separation model) takes a while; the page shows the build log. If the Space reports a configuration error, add
   this block at the very top of `README.md` in the Space:
   ```yaml
   ---
   title: DrumScribe
   sdk: docker
   app_port: 7860
   ---
   ```

### Option B: any computer or server with Docker

```bash
docker build -t drumscribe .
docker run -p 7860:7860 drumscribe        # then open http://localhost:7860
```

The image is roughly 4 to 5 GB. Settings from section 3 are passed with `-e`, for example
`docker run -e DRUMSCRIBE_MAX_MINUTES=6 -p 7860:7860 drumscribe`.

---

## 10. Licences: what you may and may not do

Short version: **personal and non-commercial use only**, unless the parts below are replaced or permission is given.
(Not legal advice; details in `LICENSE_AUDIT.md`.)

| Part | Licence | What it means |
|---|---|---|
| ADTOF drum detector (weights) | CC BY-NC-SA 4.0 | Non-commercial only, credit the authors, share changes under the same licence. |
| ADTOF-pytorch (the code that runs it) | **no licence file** | By default nobody may redistribute it. That is why it is downloaded separately and not stored in this repository. Ask its author (xavriley on GitHub) before publishing it. |
| Label model (`labels.joblib`) | trained on MDB Drums, CC BY-NC-SA 4.0 | Non-commercial only. |
| Demucs (separation) | code MIT; weights' terms not verified | |
| MuseScore 3 | GPL | Used as a separate program, which is its intended use. |
| Training samples | CrocellKit CC BY 4.0, VSCO-2 CC0, GeneralUser GS | Allow commercial use. |

Songs you upload belong to their rights holders. The tick box on the website asks the user to confirm they may use
the recording.

---

## 11. Troubleshooting

| Problem | Fix |
|---|---|
| "Couldn't find a steady beat" | Enter the tempo under More options. |
| The score feels twice too fast or slow | Click one of the other tempos on the result page, or enter the tempo. |
| The bar lines are shifted (beat 1 in the wrong place) | Command line: `--beat1 1`, `2` or `3`. With `--tempo` and `--offset` the grid is exact. |
| Hi-hat shows 16ths where you hear 8ths | Hi-hat detail: **Simplify to eighths**. |
| 6/8 or shuffle song written in 4/4 (or the other way round) | Choose the time signature under More options. |
| "The server ran out of memory" | Untick "Separate the drums first", or use a shorter clip. |
| Separation is very slow | Expected on CPU: about 6 minutes for a 3.5-minute song on 8 cores, slower on fewer. |
| The PDF is missing or empty | Check MuseScore: `QT_QPA_PLATFORM=offscreen mscore3 -o test.pdf outputs/musicxml/<song>.musicxml` |
| The player has no drum sound | Install `musescore-general-soundfont-small` and `libfluidsynth3`. |
| `torch` is huge or wants a graphics card | It came from PyPI. Reinstall from the CPU index: `uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cpu torch torchaudio` |
| "Address already in use" | Another program uses the port: `PORT=8811 ./serve.sh`. |

---

## 12. Where everything is in the project

```
backend/
  pipeline.py            the whole process (section 6), also the command line
  intake.py              accepted inputs, refused links
  separation/            Demucs drum separation
  detection/             ADTOF hits, flams/open hats/cymbal analysis, learned labels
  rhythm/                tempo, time signature, bar lines, grid, hi-hat clean-up, kick fill
  reconstruction/        hits -> bars
  notation/              bars -> MusicXML
  engraving/             MusicXML -> PDF (MuseScore)
  exports.py             MIDI and JSON
  player/                blue-playhead player and score audio
  validation/            checks for MusicXML and PDF, synthetic test songs
  api/                   the website (server, queue, pages in api/static/)
  worker.py              runs one website job in its own process
  tests/                 automatic tests
scripts/                 setup checks, model downloads, training, benchmarks, browser test
models/weights/          the trained label model (not on GitHub)
third_party/             ADTOF-pytorch (downloaded separately, not on GitHub)
data/                    uploads, separated drum tracks, datasets (not on GitHub)
outputs/                 command-line results (not on GitHub)
Dockerfile               container for hosting (untested, section 9)
README.md                short overview
PROGRESS.md              full development log with every measurement
LICENSE_AUDIT.md         licence details
MODEL_REGISTRY.md        exact model versions and checksums
```
