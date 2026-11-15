# AI Video Channels: End-to-End Plan (v1, 15 Nov 2026)

> Superseded in part by CLAUDE_RESPONSE.md (v2 changes after ChatGPT review).

Start two new channels now — cozy sleep stories (UK/US) and American frontier history (US) — and race to monetisation before YouTube doubles the watch-hour bar on 1 Feb 2027. Research: vidIQ, 15 Nov 2026.

## Owner's goal

- Anything AI-generated that holds top-tier viewers (US, UK, Australia, Canada) for long watch time and earns ad revenue. Not limited to kids content.
- Multiple channels are fine.
- Claude is the main operator: research, channel planning, video planning, production, analytics, improvement.
- Owner prefers open-weight models on rented GPUs (Runpod) over paid per-generation services, for cost. Every GPU job is budget-capped with an auto shut-off.

## The deadline that shapes everything

Today a channel needs 1,000 subscribers plus 4,000 watch hours in 12 months to earn ad revenue. From 1 Feb 2027, new applicants need 8,000 watch hours. Channels already in the programme keep their status (source: https://fliki.ai/blog/new-youtube-monetization-requirements-2027).

So the first goal is watch hours, fast. Long videos (1–3 h) earn them fastest, which is why both channels lead with long formats.

Policy guardrail: AI is allowed, but mass-produced, templated videos lose monetisation at channel level, and AI personas giving financial or health advice are not monetisable. Every video must carry original writing and real editorial choices.

## Channels

| # | Channel | Viewers | Format | Evidence | Start |
|---|---|---|---|---|---|
| A | Cozy sleep stories for grown-ups | UK, US | 1.5–3 h, slow narration, painted scenes, rain/fire ambience | Candlebury Tales (UK, 3.6K subs): one video 146K views, 239x its average. "Sleep stories for adults": 115K searches/month, US the second-largest market. "Cozy bedtime story" searches up 88% | Week 1 |
| B | American frontier and Wild West history | US | 45–90 min documentary, plus a 2–3 h "fall asleep to" cut of the same script | Frontier Legend (9.8K subs): O.K. Corral video 68K views. "The wild west" searches up 68%; "wild west outlaws" 28K/month at low competition | Week 1 |
| C | Money history and how wealth works (stories, never advice) | US | 25–40 min | Highest ad value: US Business about $2.7K–7.3K per 1M views. AI-labelled "48 Laws of Money" hit 127x on a 2.2K channel | Month 2, if A and B are on track |
| — | Kids channel (existing) | — | As now | Keep running; no expansion yet | Ongoing |

Why American history, not general history: generic "history documentary" searches come mostly from Pakistan and India, where ads pay about 20x less. US-specific topics pull US viewers.

Dropped for now: stoicism (crowded with near-identical AI channels), ambient music (low engagement), slow-English stories (mostly non-tier-1 learners).

## First six videos per channel (draft titles)

**A. Cozy sleep stories**

1. A Rainy Night at the Lighthouse Inn | Cozy Sleep Story for Grown-Ups
2. The Little Dragon Who Kept the Library Warm | Cozy Fantasy Sleep Story
3. Drifting Past Saturn: A Slow Journey Through Space | Sleep Story
4. The Quiet Mystery of Thornbury Village | Cozy Mystery Sleep Story
5. Snowbound in a Cabin by the Fire | Sleep Story with Rain and Fireplace
6. The Night Train Through the Highlands | Gentle Sleep Story

**B. Frontier history** (each also gets a "Fall Asleep to…" cut)

1. The Northfield Raid: The Day the James-Younger Gang Fell (1876)
2. What Life in a Frontier Town Was Really Like
3. The Great Train Robberies of the Old West
4. The Wild Bunch: The Last Outlaws of the West
5. Deadwood 1876: A Gold Rush Town With No Law
6. How Americans Survived Summer Before Air Conditioning

## Money: what it could earn

vidIQ's estimate for a US education audience is about $2.0K–5.5K per 1M monthly views (Business: $2.7K–7.3K).

| Monthly views (mostly US/UK) | Ad revenue per month |
|---|---|
| 100K | about $200–550 |
| 500K | about $1,000–2,700 |
| 1M | about $2,000–5,500 |

Estimates, not promises. Revenue starts only after monetisation approval, and sleep viewers may watch fewer ads.

Cost per video (approximate): scripts by Claude; voice from an open-weight model; 20–40 images from a rented GPU pod with an auto shut-off. Target under $5 per video. Proposed cap: $60 per month across all channels.

## How each video is made

1. Topic and keywords — Claude with vidIQ; owner picks from a ranked list.
2. Script — Claude writes; a separate reviewer checks facts (history) and originality.
3. Voice — one narrator per channel, approved once from a 60-second sample.
4. Pictures — one approved style sheet per channel; image model on a budget-capped pod.
5. Edit and render — code only: slow camera moves, ambience, chapters.
6. Packaging — 3 titles and 2 thumbnails, scored in vidIQ; owner picks.
7. Upload — private first; AI disclosure set when scenes look realistic; owner makes it public.

Cadence: A posts 2 per week; B posts 1 documentary plus its sleep cut per week.

## Analysis and improvement loop

- 48 hours: impressions, click-through rate, first-minute retention. Under about 4% CTR after 1,000 impressions → new title or thumbnail.
- 7 days: average view duration, traffic sources, viewer countries, returning viewers.
- Monthly: scorecard per channel, then decide kill, hold or scale.
- One change per test, logged per channel (idea, change, result, lesson).

| Signal (after 8–10 videos) | Scale | Kill |
|---|---|---|
| Viewers from US/UK/AU/CA | above 50% | below 25% |
| Average view duration | above 35% (or 40+ min on sleep videos) | below 20% |
| Breakout video (5x the channel's average) | at least one | none after 10 |
| Cost per video | under $5 | over cap twice |

## Timeline

| When | Milestone |
|---|---|
| Week 1 (by 22 Nov) | Owner creates channels A and B; Claude delivers voice samples and style sheets |
| Weeks 2–3 | First 2 videos per channel, pipeline tuned |
| Weeks 3–8 | Full cadence; first 48-hour and 7-day reviews; titles and thumbnails re-tested |
| Week 8 (mid-Jan) | Kill, hold or scale; launch C if A and B are healthy |
| By 31 Jan 2027 | Apply for monetisation on any channel past 1,000 subs and 4,000 hours |
