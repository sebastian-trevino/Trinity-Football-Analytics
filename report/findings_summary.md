# Trinity University Football — EPA Findings Summary

**Project:** Football Data Analytics Internship  
**Analyst:** Sebastian Trevino  
**Data:** 2018–Present (~10 years, 100+ games)  
**Last Updated:** Spring 2026

---

## What This Report Is

This document summarizes the key findings from Trinity University's inaugural football analytics project. The goal was to build a statistically grounded framework for evaluating play-level performance — giving coaches a consistent, situation-aware lens to assess every play on both sides of the ball.

All findings are derived from the Expected Points (EP) and Expected Points Added (EPA) model built from Trinity's own Hudl game data.

---

## The Framework: Expected Points & EPA

### What is Expected Points (EP)?
EP estimates how many points a team is likely to score from a given game situation. It accounts for down, distance, field position, and game context. For example, a 1st-and-10 from the opponent's 30-yard line has a higher EP than a 3rd-and-15 from your own 20.

### What is Expected Points Added (EPA)?
EPA measures how much a single play helped or hurt relative to what was expected. A play with **positive EPA** outperformed the situation — it was a successful play. A play with **negative EPA** fell short of expectation.

This matters because raw yards gained doesn't tell the whole story:
- A 3-yard gain on 3rd-and-2 → **successful** (positive EPA)
- A 3-yard gain on 3rd-and-10 → **not successful** (negative EPA)

EPA gives coaches a consistent way to evaluate every play in context.

---

## Model Performance

The EP model was built using a Generalized Additive Model (GAM) trained on approximately 12,000 plays across all available seasons.

| Metric | Value |
|---|---|
| R² | 0.143 |
| RMSE | 3.019 |
| Pseudo R² | 0.150 |
| AIC | 49,477 |

All five model inputs — down, distance, field position, goal-to-go, and game context — were statistically significant (p < 0.01). The R² of 0.143 is consistent with EP models in the broader football analytics literature, reflecting that individual play outcomes are inherently variable and hard to predict, but the model reliably captures the structural value of game situations.

---

## Key Findings

### Offensive Performance

**Overall EPA:** Trinity's offense averaged **+0.010 EPA per play** — marginally positive, meaning the offense performed slightly above expectation on average across all situations.

**By Down:**
- The offense performs best on 1st down, where play-calling tends to be most diverse and defenses are less prepared.
- 3rd down is the most situationally demanding — EPA drops significantly on 3rd-and-long (11+ yards), where conversion rates fall and negative plays have an outsized impact.
- 4th down EPA is heavily driven by a small sample of aggressive play calls.

**Run vs. Pass:**
- Both run and pass plays produce positive EPA on average, with passing showing higher variance — bigger upside and bigger downside plays.
- Explosive passes (21+ yard gains) are disproportionately valuable, contributing a significant share of total offensive EPA despite being a small percentage of plays.

**By Field Zone:**
- Offensive EPA peaks in opponent territory (41–60 yards from end zone), where play-calling opens up and defenses face more coverage decisions.
- Red zone EPA (inside the 20) is lower than midfield, reflecting increased defensive difficulty and tighter coverage windows — a common pattern in football analytics.
- Own territory (0–20 yards from own end zone) shows negative EPA, consistent with conservative play-calling and the risk of giving up field position.

**Explosive Plays:**
- Explosive runs (12+ yards) and explosive passes (21+ yards) are rare but critical. They account for a small percentage of total snaps but drive a disproportionate share of total offensive EPA.
- Games where Trinity generates multiple explosive plays correlate strongly with positive offensive outcomes.

**By Personnel & Formation:**
- Certain personnel groupings consistently outperform others on an EPA basis. The interactive personnel/formation dashboard allows coaching staff to identify which combinations produce the most value.
- Formations with higher play counts and positive EPA represent Trinity's most reliable offensive packages.

---

### Defensive Performance

**Overall EPA:** Trinity's defense averaged **-0.134 EPA per play** — meaning the defense held opponents meaningfully below their expected point output. Negative defensive EPA is good: it means Trinity was making plays that cost the opponent more than expected.

**Interpretation:** A defensive EPA of -0.134 per play compounds significantly over a game. Across 60+ defensive plays per game, this represents a substantial points-suppressed advantage for Trinity's defense.

---

### Special Teams

**Overall EPA:** Special teams averaged **+3.016 EPA** — driven largely by field goal results. This number reflects a small sample of high-leverage plays (field goals, punts, kick returns) and should be interpreted carefully given sample size.

---

## Win/Loss Fingerprint

One of the most actionable findings from the analysis is the EPA profile difference between wins and losses. Across all seasons, Trinity's wins are characterized by:

- **Higher 1st down EPA** — getting off to good starts on drives
- **Better 3rd down conversion efficiency** — sustaining drives when it matters
- **More explosive plays** — big-gain plays that change field position and momentum
- **Stronger red zone EPA** — finishing drives when in scoring position

In losses, the EPA fingerprint shows the inverse — negative early-down EPA and poor 3rd down efficiency are the clearest leading indicators of difficult games.

---

## 3rd Down Efficiency

3rd down is the most high-leverage situational category in football. Key findings:

| Distance | Conversion Rate | Avg EPA |
|---|---|---|
| Short (≤3 yds) | Highest | Positive |
| Medium (4–7 yds) | Moderate | Near zero |
| Standard (8–10 yds) | Below average | Negative |
| Long (11+ yds) | Lowest | Most negative |

**Implication for coaching:** Play-calling on 2nd down directly shapes 3rd down situation. Getting to 3rd-and-short (via a good 2nd down play) dramatically improves both conversion rate and EPA. Avoiding 3rd-and-long situations is one of the highest-leverage adjustments an offense can make.

---

## Recommendations

1. **Prioritize 2nd down efficiency.** The easiest way to improve 3rd down conversion rate is to gain meaningful yards on 2nd down. Targeting 3rd-and-5 or shorter should be an explicit play-design goal.

2. **Protect explosive play opportunities.** Explosive plays drive a disproportionate share of total offensive EPA. Play designs and personnel groupings that create explosive play potential — even at the cost of some consistency — are worth the tradeoff.

3. **Use EPA to evaluate play-calling, not just outcomes.** A play that gains 4 yards on 3rd-and-3 is a success. The same play on 3rd-and-8 is not. EPA gives coaches a consistent standard to evaluate decisions independent of result.

4. **Monitor red zone EPA as a season-long indicator.** Red zone efficiency separates good offenses from great ones. Targeted red zone package development — informed by which personnel/formation combinations produce the best EPA inside the 20 — is a high-value area for future analysis.

5. **Track the win/loss EPA fingerprint in-season.** The interactive dashboard allows coaching staff to monitor whether the team's EPA profile in a given season resembles their winning profile or their losing one — a useful early warning system for performance trends.

---

## What's Next

- **Interactive Tableau dashboards** for in-game and weekly game-planning use
- **Defensive EPA breakdown** by front, coverage, and blitz package
- **Opponent scouting module** — applying the EPA framework to upcoming opponents' tendencies
- **Season-over-season trend tracking** as the dataset grows with each new season

---

## About This Project

This project represents the first analytics internship in Trinity University Football history. The EP/EPA framework built here establishes a repeatable, statistically grounded process for future analysts to build on. All modeling code, notebooks, and documentation are maintained in the project repository.

**Analyst:** Sebastian Trevino — B.S. Business Analytics & Technology, Minor in Data Science | Trinity University ('26)  
Former Trinity Varsity Football player (2022–2025)

📧 srtrevino03@gmail.com  
LinkedIn: www.linkedin.com/in/sebastian-trevino-131671310
HandShake: https://app.joinhandshake.com/profiles/nckwm7
