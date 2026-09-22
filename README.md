# 🏈 Trinity University Football Analytics

**Internship Project | Trinity University Athletics**  
**Role:** Football Data Analytics Intern  
**Duration:** May 2025 – August 2025 · January 2026 – Present  
**Status:** 🔄 Model complete — interactive dashboards in progress

---

## Overview

This project represents the inaugural data analytics internship for Trinity University's football program. I designed and built the end-to-end analytics workflow from scratch — establishing a repeatable process for future interns — covering everything from raw data collection to predictive modeling and interactive visualization for coaching staff.

The core objective was to give coaches a data-driven edge: understanding what drives play success, where the team performs well, and how they can optimize performance.

---

## What This Project Does

| Component | Description |
|---|---|
| **Data Pipeline** | Collected, cleaned, and processed ~10 years of game data from Hudl |
| **Exploratory Analysis** | Identified trends, strengths, and weaknesses across seasons |
| **Expected Points (EP) Model** | Estimated the point value of each play situation using a Generalized Additive Model (GAM) |
| **Expected Points Added (EPA)** | Measured each play's contribution relative to expectation — offense and defense |
| **Play Success Modeling** | Ensemble model predicting play success probability and identifying high-impact variables |
| **Dashboards** | Interactive Tableau and Python dashboards for coaches to explore play probabilities and performance drivers [Download and open HTML files locally to view interactive dashboards](figures/) |

---

## Technical Stack

| Tool | Use |
|---|---|
| **Python** | Data cleaning, feature engineering, visualizations (matplotlib, seaborn, plotly), modeling (pandas, scikit-learn, pyGAM) |
| **R** | Statistical analysis and supplementary modeling |
| **Tableau** | Interactive dashboard development for coaching staff |
| **Hudl** | Source platform for raw game film and play-by-play data |

---

## Modeling Approach

### Expected Points (EP) — `pyGAM` LinearGAM

To quantify play value, I built a **Generalized Additive Model (GAM)** that estimates the expected points for a given game situation. This forms the foundation for all downstream metrics.

**Model inputs:** down, distance, yard line, goal to go, game context  
**Target:** Drive points scored from that play forward

**Model output on ~16,000 plays:**

```
R²:   0.1510    RMSE: 2.977
Pseudo R²: 0.153    AIC: 66,685
```

> All five features were statistically significant (p < 0.01), confirming that down, distance, field position, goal to go, and game context meaningfully predict scoring outcomes.

**Model EP Summary by Unit:**

| Unit | Mean EP | Std Dev |
|---|---|---|
| Offense | 2.489 | 1.317 |
| Defense | 2.095 | 1.142 |

### Model Expected Points Added (EPA)

EPA measures how much each play helped or hurt relative to expectation — the industry-standard metric for evaluating play-level performance.

| Unit | Mean EPA | Interpretation |
|---|---|---|
| Offense | +0.028 | Marginally positive per play |
| Defense | -0.137 | Holding opponents below expectation |
| Special Teams | +2.915 | Small sample, largely field goal results |

### Play Success — EPA as the Measure
Rather than building a separate classifier, play success is defined directly through the EP/EPA framework. A play with positive EPA means it outperformed the expected points for that situation — that's a successful play. This approach grounds the definition of success in context: a 3-yard gain on 3rd-and-2 is successful; the same gain on 3rd-and-10 is not.
This framework gives coaching staff a consistent, situation-aware lens to evaluate every play on both sides of the ball.

---

## Data

**~10 years of Trinity University game data (2018–present)**

> ⚠️ **Note:** Raw game data is proprietary to Trinity University Athletics and is not included in this repository. `data/TU_Games_synthetic.zip` is a synthetic stand-in with the same file layout and simulated plays so the notebooks can be run end to end. The figures, report, and slideshow in this repo come from the real data. See [data/README.md](data/README.md).

---

## Repository Structure

```
├── notebooks/
│   ├── 01_EPA_TUFB_v001.ipynb               Data cleaning → feature engineering →
│   │                                        GAM Expected Points model → EPA per play
│   ├── 02_EPA_TUFB_Analysis_v001.ipynb      Offensive EPA analysis and visualizations
│   └── 03_EPA_TUFB_Analysis_Markdown.ipynb  Builds the HTML report and slideshow
├── data/
│   ├── TU_Games_synthetic.zip               129 synthetic game files (same layout as
│   │                                        the real Hudl exports, simulated plays;
│   │                                        unzipped automatically by the notebooks)
│   ├── make_synthetic_games.py              Script that generates the synthetic files
│   └── README.md
├── figures/                                 Exported charts (PNG) and interactive
│                                            Plotly dashboards (HTML)
├── docs/
│   ├── TU_EPA_Offensive_Report.html         Full offensive EPA report
│   ├── TU_EPA_Offensive_Report_Slideshow_Final.html   Coaching staff slideshow
│   └── findings_summary.md
├── requirements.txt
└── README.md
```

## Running it

```bash
git clone https://github.com/sebastian-trevino/Trinity-Football-Analytics.git
cd Trinity-Football-Analytics
pip install -r requirements.txt
jupyter notebook notebooks/
```

Run the notebooks in order (01 → 02 → 03). By default they read the synthetic games in `data/TU_Games_synthetic.zip`;
set the `TUFB_DATA_DIR` environment variable to point at the real game files instead.

---

## Key Takeaways

- Engineered the **first-ever analytics internship workflow** for Trinity Football, creating a repeatable process for future analysts
- Processed and modeled **nearly a decade of play-by-play data** to build a statistically grounded EP/EPA framework
- Delivered **actionable insights to coaching staff** via Tableau dashboards updated throughout the season
- Currently deploying **play success probability outputs** into interactive dashboards for real-time game planning

---

## About Me

**Sebastian Trevino** — B.S. Business Analytics & Technology, Minor in Data Science | Trinity University ('26)  
Former Trinity Varsity Football player (2022–2025) with a unique perspective on bridging on-field experience with data-driven analysis.

📧 srtrevino03@gmail.com
LinkedIn: www.linkedin.com/in/sebastian-trevino-131671310
HandShake: https://app.joinhandshake.com/profiles/nckwm7
