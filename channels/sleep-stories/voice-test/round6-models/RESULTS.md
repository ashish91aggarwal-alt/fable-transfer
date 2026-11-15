# Round 6: two new open-weight models (both Apache-2.0), voice made from a text description

Description used: see `voice.txt`. Same 1-minute "Picture a path..." passage, one take per seed.

Gates (automatic, before the owner hears anything):
- Female narrator: median pitch 165–230 Hz (owner rejected "manly" and "high-pitched")
- Pace: 140 words/min or slower
- Words: under 3% error vs script (speech recognition)

| Clip | Pitch | Pace | Word error | Result |
|---|---|---|---|---|
| Qwen3-TTS VoiceDesign, seed 1 | 204 Hz | 136 wpm | 1.5% | PASS |
| Qwen3-TTS VoiceDesign, seed 2 | 249 Hz | 137 wpm | 1.5% | fail: too high |
| VoxCPM2, seed 1 | 96 Hz | 177 wpm | 1.5% | fail: male, too fast |
| VoxCPM2, seed 2 | 174 Hz | 201 wpm | 1.5% | fail: too fast |

("grey" and "stile" flagged in every clip are spelling differences in the checker, not reading errors.)
Pod cost: about $0.16 (one slow-network host replaced).
