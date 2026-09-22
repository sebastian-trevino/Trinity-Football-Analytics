## Data

### TU_Games_synthetic.zip — synthetic stand-in for the real game files
The real `TU_Games` folder holds one Hudl play-list export per game (129 games,
2018–2025). Those files are proprietary to Trinity University Athletics and are
**not** in this repository.

`TU_Games_synthetic.zip` replicates that folder so the notebooks run end to end from a
fresh clone. The notebooks unzip it into `data/TU_Games_synthetic/` automatically on
first run (it is zipped because GitHub's web uploader accepts at most 100 files at a time).

| Kept from the real folder | Simulated |
|---|---|
| File names (teams, date, home/away, W/L, final score — public record) | Every play: down, distance, field position, gain, result, drive flow, kicks and scores |
| Column layout of each file (the exports come in several schema variants, which the cleaning code has to handle) | Every offensive and defensive play-call field (personnel, formation, motion, protection, play name, blocking scheme, front, stunt, coverage, blitz) using invented names |
| Tagging conventions (ODK codes, result labels, play types, yard-line sign convention) | Free-text fields (`TITLE`, `RETURN NAME/TYPE`) are blank or generic |

No real play rows are copied, and no play-call name from the real exports appears in
the synthetic files. The offense and defense use separate invented vocabularies.
Each simulated game is picked to land close to the real final score, so season-level
results look plausible, but the play-by-play is fictional. **Charts produced from this
data will not match the real findings.**

### make_synthetic_games.py
Rebuilds the `TU_Games_synthetic/` folder from `synthetic_manifest.json` (file name, column
layout and row count per game):

```bash
python data/make_synthetic_games.py
```

### processed/ *(generated, not committed)*
`notebooks/01_EPA_TUFB_v001.ipynb` writes `processed/TUFB_EPA_Analysis_FULL_DATA.xlsx`,
which notebooks 02 and 03 read.

### About the committed outputs
The figures in `figures/`, the report and slideshow in `docs/`, and the saved cell
outputs in `notebooks/` come from a run on the **real** data. Cells that printed raw
play-by-play rows were cleared before committing.
