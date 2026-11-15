# Voice test results (15 Nov 2026)

Four blind voices (voice-1 to voice-4) read the same passages: A opening, B final fade, C stress test, D long passage (~1,340 words).
Automatic check: faster-whisper small.en transcribed each sample and compared it word by word with the script. Spelling differences (grey/gray, oilskin/oil skin) are not real errors.

| Voice | Words/min A (incl. pauses) | Long passage pace | Real problems found |
|---|---|---|---|
| voice-1 | 150 | sped up to 171 | "wicks to" merged into a non-word; stray "and round" added at the end of C |
| voice-2 | 136 | steady at 147 | "Saltmere Head" heard as "salt me ahead"; stray sounds in the fade |
| voice-3 | 163 | not tested | cleanest transcription, but fastest pace |
| voice-4 | 117 | not tested | stray sounds in the fade |

Render speed: voice-1/3 model about 0.75x real time on an RTX 4000 Ada ($0.20/h), so a 50-min story costs about $0.13; voice-2/4 model runs on a plain CPU at about 0.8x real time (free).

Audio: voice-2 and voice-4 are in `samples/`. Voice-1 and voice-3 audio stayed on the test pod (no way to copy 24 MB back without a GitHub token on the pod); they can be re-rendered for about $0.15.
