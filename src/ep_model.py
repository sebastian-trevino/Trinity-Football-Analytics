"""
ep_model.py
Trinity University Football Analytics

Builds and evaluates the Expected Points (EP) GAM model, then computes
Expected Points Added (EPA) for every play. Called by 03_ep_model.ipynb.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from pygam import LinearGAM, s, f


# ── Constants ──────────────────────────────────────────────────────────────────

FEATURES = ['dn', 'dist', 'yardline_100', 'series', 'goal_to_go']
TARGET   = 'drive_points'


# ── Model Building ─────────────────────────────────────────────────────────────

def build_model_dataframe(all_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare the modeling DataFrame from the full play-by-play dataset.

    Separates offense/defense plays (used for modeling) from special
    teams plays (k_df), which get EP = NaN and are added back later.

    Args:
        all_df: Fully engineered DataFrame from preprocessing pipeline.

    Returns:
        Tuple of (model_df, k_df).
    """
    model_df = all_df[all_df['odk'].isin(['o', 'd'])].copy()
    k_df     = all_df[all_df['odk'] == 'k'].copy()

    # Rename for model clarity
    model_df = model_df.rename(columns={'yards_to_go': 'yardline_100'})

    # Enforce numeric types and drop rows with missing model inputs
    for col in FEATURES + [TARGET]:
        model_df[col] = pd.to_numeric(model_df[col], errors='coerce')

    model_df = model_df.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)

    print(f"Modeling rows: {len(model_df)}")
    print(f"Target distribution:\n{model_df[TARGET].value_counts().sort_index()}")

    return model_df, k_df


def train_ep_model(model_df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Train a Generalized Additive Model (GAM) to predict Expected Points.

    Model structure:
        f(dn)              — factor term: down is categorical (1-4)
        s(dist)            — spline: continuous distance to first down
        s(yardline_100)    — spline: continuous yards to end zone
        s(series)          — spline: drive number in game (game context)
        f(goal_to_go)      — factor term: binary goal-to-go flag

    gridsearch() auto-tunes the smoothing penalty (lambda) via cross-validation.

    Args:
        model_df:     Prepared DataFrame from build_model_dataframe().
        test_size:    Proportion of data held out for evaluation.
        random_state: Random seed for reproducibility.

    Returns:
        Fitted LinearGAM model.
    """
    X = model_df[FEATURES].astype(float)
    y = model_df[TARGET].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")

    gam = LinearGAM(
        f(0) +                # dn
        s(1, n_splines=10) +  # dist
        s(2, n_splines=10) +  # yardline_100
        s(3, n_splines=8)  +  # series
        f(4)                  # goal_to_go
    ).gridsearch(X_train, y_train)

    y_pred = gam.predict(X_test)

    print(f"\nModel Performance:")
    print(f"  R²:   {r2_score(y_test, y_pred):.4f}")
    print(f"  RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
    gam.summary()

    return gam


def predict_ep(gam, model_df: pd.DataFrame, k_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate EP predictions for all plays and recombine with special teams rows.

    Args:
        gam:      Fitted LinearGAM model.
        model_df: Offense/defense DataFrame (renamed yardline_100 column).
        k_df:     Special teams DataFrame.

    Returns:
        Combined DataFrame with ep column (NaN for special teams).
    """
    model_df['ep'] = gam.predict(model_df[FEATURES].astype(float))
    k_df = k_df.copy()
    k_df['ep'] = np.nan

    combined = pd.concat([model_df, k_df], ignore_index=True)
    combined = combined.sort_values(['game_id', 'play']).reset_index(drop=True)

    print("\nEP by unit:")
    print(combined.groupby('odk')['ep'].describe().round(3))

    return combined


# ── EPA Calculation ────────────────────────────────────────────────────────────

def compute_epa(model_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Expected Points Added (EPA) for every play.

    EPA logic:
        Scoring play (end_of_drive == 1, score_event != 0):
            epa = score_event - ep
            (drive cashed in; ep_after resets to 0)

        Non-scoring drive ending (punt, turnover, end of game):
            epa = 0 - ep
            (drive produced nothing; ep_after = 0)

        Mid-drive play:
            epa = ep_next - ep

    Special teams scoring plays receive epa = score_event directly.

    Args:
        model_df: Combined DataFrame from predict_ep() with ep column.

    Returns:
        DataFrame with end_of_drive, ep_next, and epa columns added.
    """
    model_df = model_df.copy()

    # ── End-of-drive flag ──────────────────────────────────────────────────────
    model_df['end_of_drive'] = 0

    series_mask = model_df['series'] != 0
    model_df.loc[series_mask, 'end_of_drive'] = (
        model_df[series_mask]
        .groupby(['game_id', 'series'])['play']
        .transform('max') == model_df.loc[series_mask, 'play']
    ).astype(int)

    # Scoring plays always mark end of drive
    model_df.loc[
        model_df['odk'].isin(['o', 'd']) & (model_df['score_event'] != 0),
        'end_of_drive'
    ] = 1

    # ── ep_next ────────────────────────────────────────────────────────────────
    model_df['ep_next'] = (
        model_df.groupby(['game_id', 'odk'])['ep'].shift(-1)
    ).fillna(0)

    # ── EPA formula ────────────────────────────────────────────────────────────
    model_df['epa'] = (
        (model_df['ep_next'] * (1 - model_df['end_of_drive']))
        + model_df['score_event']
    ) - model_df['ep']

    # Special teams scoring: EPA = score_event
    k_score_mask = (model_df['odk'] == 'k') & (model_df['score_event'] != 0)
    model_df.loc[k_score_mask, 'epa'] = model_df.loc[k_score_mask, 'score_event']

    print("EPA by unit:")
    print(model_df.groupby('odk')['epa'].describe().round(3))

    return model_df


def merge_ep_epa(all_df: pd.DataFrame, model_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge ep and epa columns back onto the full play-by-play dataset.

    Args:
        all_df:   Full dataset (from preprocessing pipeline).
        model_df: Dataset with ep and epa columns (from compute_epa()).

    Returns:
        all_df with ep and epa columns added via left join on game_id + play.
    """
    all_df = all_df.merge(
        model_df[['game_id', 'play', 'ep', 'epa']],
        on=['game_id', 'play'],
        how='left'
    )
    print(f"\nEP/EPA merged. Null EP count: {all_df['ep'].isna().sum()}")
    return all_df


# ── Full Pipeline ──────────────────────────────────────────────────────────────

def run_ep_pipeline(all_df: pd.DataFrame) -> tuple[pd.DataFrame, object]:
    """
    End-to-end EP/EPA pipeline: build model df → train GAM → predict EP → compute EPA → merge.

    Args:
        all_df: Fully engineered DataFrame from preprocessing pipeline.

    Returns:
        Tuple of (all_df with ep/epa, fitted gam model).

    Example:
        all_df, gam = run_ep_pipeline(all_df)
    """
    model_df, k_df = build_model_dataframe(all_df)
    gam             = train_ep_model(model_df)
    model_df        = predict_ep(gam, model_df, k_df)
    model_df        = compute_epa(model_df)
    all_df          = merge_ep_epa(all_df, model_df)
    return all_df, gam
