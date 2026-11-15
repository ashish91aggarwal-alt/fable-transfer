# Cast sheets and scene sheets: DRAFT for owner approval

Nothing here is painted yet. These descriptions are what `paint.py` will send in every prompt, so they must be fixed first. Approve or change each one.

## Characters (full description is sent with every prompt)

| id | trigger | description |
| --- | --- | --- |
| tilly | tillytort | tillytort, a small sage-green tortoise with a domed shell of warm tan hexagons and a honey-brown rim, a blue knitted neckerchief, big round dark-brown eyes, pink cheeks, four short sturdy legs |
| hugo | hugohare | hugohare, a caramel-brown hare with a white chest and white tail, tall upright ears pink inside, a red headband, big round amber eyes, long hind legs |
| wren | wrenbird | wrenbird, a tiny round brown wren with a cream chest, a small pointed yellow beak, big round black eyes (non-speaking, small in frame) |

Design notes:
- Colors are far apart on purpose (Tilly green and blue, Hugo caramel and red, wren brown) so features do not blend.
- Each has one accessory that can be checked on every page: neckerchief, headband.
- Flaws are behavior only. No looks are tied to Hugo's bragging.
- Sheets to make per character: front view, side view (walking), and 2-3 faces (happy, sad, surprised). Tilly also needs a walking view; Hugo a running view and a sleeping view.

## Scenes (one approved painting each, then reused as the base for the pages)

| id | scene | light and palette |
| --- | --- | --- |
| s1 meadow | Green meadow, wildflowers, a wooden fence, a dirt path | golden olive, sand, warm cream (morning) |
| s2 start | Same meadow, wren on the fence, a start line drawn in the dirt, red ribbon visible far away | warm gold (midday) |
| s3 path | Long dirt path winding across the meadow with a gentle slope, no water or road | warm gold, then paler and cooler for the hot afternoon |
| s4 oak | A big shady oak tree with soft grass beneath | cool green shade, dappled light |
| s5 finish | Red ribbon stretched between two wooden posts at the end of the path | warm gold, then amber |
| s6 home | The path home at the end of the day | warm amber |

## Page to scene map

cover s1 | p01 s1 | p02 s1 | p03 s1 | p04 s2 | p05 s3 | p06 s3 | p07 s3 | p08 s4 | p09 s4 | p10 s3 (hot, paler) | p11 s5 | p12 s5 | p13 s5 | p14 s6

## LoRA and model decision (needs owner choice)

Findings from a web check today (not tested by me):
- The kit is built for Qwen-Image (paint), Qwen-Image-Edit-2509 (edits) and Wan 2.2 (clips). Training a Qwen-Image LoRA is documented with ai-toolkit and needs 24 GB or more of GPU memory, with 32 GB or more comfortable. The kit's A40 48 GB fits.
- Newer open models also have LoRA training guides (Qwen-Image-2512 and 2.1, FLUX.2, Krea 2, Z-Image). I have not tested any for painted picture-book look, and testing each one costs GPU time.
- Some sources say reference-based editing can replace character LoRAs. The kit's own notes say big pose changes fail often, so a LoRA is still worth trying for the three characters.

Proposed order (owner approves each paid step):
1. Approve this text. (free)
2. Paint cast sheets and 6 scenes with base Qwen-Image, no LoRA, 2-3 seeds each. Reviewer picks. (pod, cheapest part)
3. Make 10-20 approved paintings of the cast in the scenes, plus the sheets, as the training set. Captions follow the kit pattern. (pod)
4. Train one style+cast LoRA (triggers tillytort, hugohare, wrenbird) on the A40, 1500 steps, save every 500. Reviewer picks the best step. (about $1.50-2.50 by the kit's numbers)
5. Paint the 14 pages with the LoRA.

The training set must be separate generated paintings, not one cast image cut into pieces.
