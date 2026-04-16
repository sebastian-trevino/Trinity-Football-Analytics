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
| **Dashboards** | Interactive Tableau and Python dashboards for coaches to explore play probabilities and performance drivers [Download and open HTML files locally to view interactive dashboards](dashboards/screenshots/) |

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

**Model output on ~12,000 plays:**

```
R²:   0.143    RMSE: 3.019
Pseudo R²: 0.150    AIC: 49,477
```

> All five features were statistically significant (p < 0.01), confirming that down, distance, field position, goal to go, and game context meaningfully predict scoring outcomes.

**EP Summary by Unit:**

| Unit | Mean EP | Std Dev |
|---|---|---|
| Offense | 2.43 | 1.29 |
| Defense | 2.13 | 1.17 |

### Expected Points Added (EPA)

EPA measures how much each play helped or hurt relative to expectation — the industry-standard metric for evaluating play-level performance.

| Unit | Mean EPA | Interpretation |
|---|---|---|
| Offense | +0.010 | Marginally positive per play |
| Defense | -0.134 | Holding opponents below expectation |
| Special Teams | +3.016 | Small sample, largely field goal results |

### Play Success — EPA as the Measure
Rather than building a separate classifier, play success is defined directly through the EP/EPA framework. A play with positive EPA means it outperformed the expected points for that situation — that's a successful play. This approach grounds the definition of success in context: a 3-yard gain on 3rd-and-2 is successful; the same gain on 3rd-and-10 is not.
This framework gives coaching staff a consistent, situation-aware lens to evaluate every play on both sides of the ball.

---

## Data

**~10 years of Trinity University game data (2018–present)**

> ⚠️ **Note:** Raw game data is proprietary to Trinity University Athletics and is not included in this repository. Sample outputs and anonymized data structures are provided for reference only.

---

## Repository Structure

```
football-analytics/
│
├── README.md
├── data/
│   └── sample_data.csv          # Anonymized sample — not real game data
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_ep_model.ipynb
│   └── 04_play_success_model.ipynb
├── src/
│   ├── preprocessing.py
│   ├── ep_model.py
│   └── play_success_model.py
├── dashboards/
│   └── screenshots/             # Tableau and Python dashboard exports
└── reports/
    └── findings_summary.md
```

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
