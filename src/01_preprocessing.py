"""
preprocessing.py
Trinity University Football Analytics

Reusable functions for loading, cleaning, and feature engineering
raw Hudl game data. Called by 01_data_cleaning.ipynb and 02_eda.ipynb.
"""

import os
import re
import glob
import pandas as pd
import numpy as np
from datetime import datetime


# ── Constants ─────────────────────────────────────────────────────────────────

EXPLOSIVE_RUN_THRESHOLD  = 12
EXPLOSIVE_PASS_THRESHOLD = 21

COLS_TO_DROP = [
    'unnamed_0', 'oline_scheme', 'kick_type', 'return_name_type',
    'points', 'point_differential', 'team_score', 'opponent_score', 'title'
]

OFF_TD_RESULTS = ['rush, td', 'complete, td', 'scramble, td', 'td']
DEF_TD_RESULTS = ['interception, def td', 'fumble, def td',
                  'sack, fumble, def td', 'block, def td']


# ── File Loading ───────────────────────────────────────────────────────────────

def find_game_files(drive_path: str) -> list[str]:
    """
    Return sorted list of .xlsx game files at the given Google Drive path.

    Args:
        drive_path: Glob-style path to TU_Games folder.
                    e.g. '/content/drive/MyDrive/TUFB_EPA/TU_Games/*.xlsx'

    Returns:
        Sorted list of file paths.

    Example:
        files = find_game_files('/content/drive/MyDrive/TUFB_EPA/TU_Games/*.xlsx')
    """
    files = sorted(glob.glob(drive_path))
    print(f"Found {len(files)} game files.")
    return files


def extract_metadata_from_filename(filename: str) -> dict:
    """
    Parse game metadata encoded in Hudl export filenames.

    Expected filename format:
        TU_Opponent_(W|L)_(H|A)_SCORE_YYYY-MM-DD_PlaylistData.xlsx

    Args:
        filename: Basename of the game file.

    Returns:
        Dict with keys: team_name, opp_name, home_away, win, game_date,
                        team_score, opp_score.
    """
    team_name = re.match(r'^([^_]+)', filename)
    team_name = team_name.group(1) if team_name else None

    opp_name = re.match(r'^[^_]+_([^_]+)', filename)
    opp_name = opp_name.group(1) if opp_name else None

    home_away = re.search(r'\((?:W|L)_([HA])\)', filename)
    home_away = home_away.group(1) if home_away else None

    win_match = re.search(r'\((W|L)_[HA]\)', filename)
    win_flag  = 1 if win_match and win_match.group(1) == "W" else 0

    date_match = re.search(r'\d{4}-\d{2}-\d{2}', filename)
    game_date  = (
        datetime.strptime(date_match.group(), "%Y-%m-%d").strftime("%m/%d/%Y")
        if date_match else None
    )

    score_match = re.search(r'\)\d+-\d+_', filename)
    if score_match:
        score_clean = re.sub(r"[)_]", "", score_match.group())
        team_score, opp_score = map(int, score_clean.split("-"))
    else:
        team_score = opp_score = pd.NA

    return {
        'team_name':  team_name,
        'opp_name':   opp_name,
        'home_away':  home_away,
        'win':        win_flag,
        'game_date':  game_date,
        'team_score': team_score,
        'opp_score':  opp_score,
    }


def load_single_game(path: str) -> pd.DataFrame:
    """
    Load one Hudl .xlsx file, standardize columns, attach metadata,
    and compute series counters and drive_down.

    Args:
        path: Full path to the .xlsx game file.

    Returns:
        Cleaned single-game DataFrame.
    """
    game_data = pd.read_excel(path)

    # Standardize column names
    game_data.columns = (
        game_data.columns
        .str.strip().str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )

    # Remove scouting rows
    game_data = game_data[game_data["odk"] != "S"].reset_index(drop=True)
    game_data = game_data.replace(["nan", "NA", "", "None"], pd.NA)

    # Fix missing ODK on Timeout rows
    for i in range(1, len(game_data)):
        if (
            pd.notna(game_data.at[i, "result"])
            and game_data.at[i, "result"] == "Timeout"
            and pd.isna(game_data.at[i, "odk"])
        ):
            game_data.at[i, "odk"] = game_data.at[i - 1, "odk"]

    # Attach metadata
    meta = extract_metadata_from_filename(os.path.basename(path))
    conditions = [game_data["odk"] == "O", game_data["odk"] == "D"]
    game_data["team_name"] = np.select(conditions, [meta['team_name'], meta['opp_name']], default=meta['team_name'])
    game_data["opp_name"]  = np.select(conditions, [meta['opp_name'],  meta['team_name']], default=meta['opp_name'])
    game_data["home_away"] = meta['home_away']
    game_data["win"]       = meta['win']
    game_data["game_date"] = meta['game_date']
    game_data["team_pts"]  = meta['team_score']
    game_data["opp_pts"]   = meta['opp_score']

    # Numeric conversion
    for col in ["dn", "ydline", "to_go", "gn_ls", "team_pts", "opp_pts"]:
        if col in game_data.columns:
            game_data[col] = pd.to_numeric(game_data[col], errors="coerce")

    # Explosive play flag
    game_data["explosive"] = game_data.apply(_is_explosive, axis=1)

    # Series counters
    game_data = _add_series_counters(game_data)

    return game_data


def load_all_games(drive_path: str) -> pd.DataFrame:
    """
    Load and combine all game files from drive_path into one DataFrame.

    Args:
        drive_path: Glob-style path to TU_Games folder.

    Returns:
        Combined play-by-play DataFrame for all games.

    Example:
        all_df = load_all_games('/content/drive/MyDrive/TUFB_EPA/TU_Games/*.xlsx')
    """
    files = find_game_files(drive_path)
    all_dfs = [load_single_game(path) for path in files]
    all_df = pd.concat(all_dfs, ignore_index=True)
    print(f"Total plays loaded: {len(all_df)} across {len(files)} games.")
    return all_df


# ── Cleaning ───────────────────────────────────────────────────────────────────

def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns that are empty in the raw Hudl export or recreated later.

    Args:
        df: Raw combined DataFrame.

    Returns:
        DataFrame with unused columns removed.
    """
    to_drop = [c for c in COLS_TO_DROP if c in df.columns]
    df = df.drop(columns=to_drop)
    print(f"Dropped columns: {to_drop}")
    return df


def clean_down_column(df: pd.DataFrame) -> pd.DataFrame:
    """Replace dn == 0 with 1 (data entry artifact in Hudl exports)."""
    df.loc[df['dn'] == 0, 'dn'] = 1
    return df

def clean_result_column(df: pd.DataFrame) -> pd.DataFrame:
    """Replace bare '1st DN' result with 'Rush' or 'Complete' based on play type."""
    firstdn_mask = (all_df['result'] == '1st DN')
    all_df.loc[firstdn_mask & (all_df['play_type'] == 'Run'),  'result'] = 'Rush'
    all_df.loc[firstdn_mask & (all_df['play_type'] == 'Pass'), 'result'] = 'Complete'
    return df

def clean_two_point_results(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize result labels for 2-point conversion plays."""
    mask = df['play_type'].isin(['2 Pt.', '2 Pt. Defend'])
    df.loc[mask, 'result'] = df.loc[mask, 'result'].replace({
        'Complete':     'Good',
        'Complete, TD': 'Good',
        'Incomplete':   'No Good',
        'Interception': 'No Good',
    })
    return df


def clean_bare_td_results(df: pd.DataFrame) -> pd.DataFrame:
    """Replace bare 'TD' result with 'Rush, TD' or 'Complete, TD' based on play type."""
    mask = (df['result'] == 'TD') & (df['odk'].isin(['O', 'D']))
    df.loc[mask & (df['play_type'] == 'Run'),  'result'] = 'Rush, TD'
    df.loc[mask & (df['play_type'] == 'Pass'), 'result'] = 'Complete, TD'
    return df


def run_cleaning_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run all cleaning steps in order.

    Args:
        df: Raw combined DataFrame from load_all_games().

    Returns:
        Cleaned DataFrame ready for feature engineering.
    """
    df = drop_unused_columns(df)
    df = clean_down_column(df)
    df = clean_two_point_results(df)
    df = clean_bare_td_results(df)
    return df


# ── Feature Engineering ────────────────────────────────────────────────────────

def add_yards_to_go(df: pd.DataFrame) -> pd.DataFrame:
    """Compute yards_to_go (distance from end zone, always positive)."""
    df['yards_to_go'] = np.where(
        df['yard_ln'] < 0,
        100 + df['yard_ln'],
        df['yard_ln']
    )
    return df


def add_goal_to_go(df: pd.DataFrame) -> pd.DataFrame:
    """Flag offensive plays inside the 10-yard line as goal_to_go."""
    df['goal_to_go'] = (
        (df['yards_to_go'] <= 10) & (df['odk'] == 'O')
    ).astype(int)
    return df


def add_drive_points(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate points scored on each drive and assign to every play in that drive.

    Scoring logic:
        - Offensive TD  → +7
        - Defensive TD  → -7
        - Field Goal    → +3
        - No score      →  0

    Args:
        df: DataFrame with off_series, def_series, odk, result, play_type columns.

    Returns:
        DataFrame with drive_points column added.
    """
    df['drive_points'] = 0.0

    for (team, date), game_df in df.groupby(['team_name', 'game_date']):
        game_df = game_df.copy()

        for series_val in game_df.loc[
            (game_df['off_series'] != 0) & (game_df['odk'] == 'O'), 'off_series'
        ].unique():
            pts = _calc_drive_points(game_df, 'off_series', series_val, 'O')
            idx = df[
                (df['team_name'] == team) & (df['game_date'] == date) &
                (df['off_series'] == series_val)
            ].index
            df.loc[idx, 'drive_points'] = pts

        for series_val in game_df.loc[
            (game_df['def_series'] != 0) & (game_df['odk'] == 'D'), 'def_series'
        ].unique():
            pts = _calc_drive_points(game_df, 'def_series', series_val, 'D')
            idx = df[
                (df['team_name'] == team) & (df['game_date'] == date) &
                (df['def_series'] == series_val)
            ].index
            df.loc[idx, 'drive_points'] = pts

        for i, row in game_df[game_df['odk'] == 'K'].iterrows():
            result    = str(row['result']).lower()
            play_type = str(row['play_type']).lower()
            if result == 'td':
                df.loc[i, 'drive_points'] = (
                    -7 if play_type in ['punt', 'ko'] else
                     7 if play_type in ['punt rec', 'ko rec'] else 0
                )
            elif 'good' in result and play_type in ['fg', 'fg block']:
                df.loc[i, 'drive_points'] = 3
            else:
                df.loc[i, 'drive_points'] = 0

    return df


def add_game_id(df: pd.DataFrame) -> pd.DataFrame:
    """Create a unique game_id string from game_date and sorted team names."""
    df['game_id'] = (
        df['game_date'].astype(str) + "_" +
        df[['team_name', 'opp_name']].apply(
            lambda x: "_".join(sorted([x['team_name'], x['opp_name']])), axis=1
        )
    )
    return df


def add_score_event(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign a simplified score_event (+7, -7, +3, 0) to each play.
    Also computes next_score (next non-zero score_event in the game).
    Removes penalty rows before computing.
    """
    df = df[df['result'] != 'penalty'].reset_index(drop=True)

    df['score_event'] = df.apply(_assign_score_event, axis=1)

    df['_score_nonzero'] = df['score_event'].replace(0, np.nan)
    df['next_score'] = (
        df.groupby('game_id')['_score_nonzero']
        .transform(lambda x: x.shift(-1).bfill())
    ).fillna(0)
    df = df.drop(columns=['_score_nonzero'])

    return df


def add_turnover_on_downs(df: pd.DataFrame) -> pd.DataFrame:
    """Flag 4th-down plays where possession changes (turnover on downs)."""
    df['next_odk'] = df.groupby('game_id')['odk'].shift(-1)
    df['turnover_on_downs'] = (
        (df['dn'] == 4) & (df['odk'] != df['next_odk'])
    ).astype(int)
    return df


def run_feature_engineering_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run all feature engineering steps in order.

    Args:
        df: Cleaned DataFrame from run_cleaning_pipeline().

    Returns:
        Fully engineered DataFrame ready for modeling.
    """
    # Standardize casing for key columns
    df.columns = (
        df.columns.str.strip().str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True).str.strip("_")
    )
    for col in ['odk', 'result', 'play_type']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()

    df['play'] = pd.to_numeric(df['play'], errors='coerce')
    df = add_game_id(df)
    df = df.sort_values(['game_id', 'play']).reset_index(drop=True)
    df = add_yards_to_go(df)
    df = add_goal_to_go(df)
    df = add_drive_points(df)
    df = add_score_event(df)
    df = add_turnover_on_downs(df)

    df['off_score'] = df['team_pts']
    df['def_score'] = df['opp_pts']

    return df


# ── Private Helpers ────────────────────────────────────────────────────────────

def _is_explosive(row) -> int:
    try:
        gain = int(row["gn_ls"])
    except (ValueError, TypeError):
        return 0
    if row["play_type"] == "Run"  and gain >= EXPLOSIVE_RUN_THRESHOLD:
        return 1
    if row["play_type"] == "Pass" and gain >= EXPLOSIVE_PASS_THRESHOLD:
        return 1
    return 0


def _add_series_counters(game_data: pd.DataFrame) -> pd.DataFrame:
    off_series_col = [0] * len(game_data)
    def_series_col = [0] * len(game_data)
    series_col     = [0] * len(game_data)

    off_counter = def_counter = series_counter = 0
    current_off = current_def = current_any = ""

    for i, val in enumerate(game_data["odk"]):
        if pd.notna(val) and val == "O":
            if current_off != "O":
                off_counter += 1
                current_off = "O"
            off_series_col[i] = off_counter
        elif pd.notna(val) and val == "D":
            if current_def != "D":
                def_counter += 1
                current_def = "D"
            def_series_col[i] = def_counter
        else:
            current_off = current_def = ""

        if pd.notna(val) and val in ["O", "D"]:
            if val != current_any:
                series_counter += 1
                current_any = val
            series_col[i] = series_counter
        else:
            current_any = ""

    game_data["off_series"] = off_series_col
    game_data["def_series"] = def_series_col
    game_data["series"]     = series_col

    game_data["drive_down"] = game_data["dn"]
    for col, odk_val in [("off_series", "O"), ("def_series", "D")]:
        for sv in game_data.loc[game_data[col] != 0, col].unique():
            idx = game_data.index[(game_data[col] == sv) & (game_data["dn"] == 1)]
            if len(idx) > 0:
                game_data.loc[idx[0], "drive_down"] = 23

    return game_data


def _calc_drive_points(game_df, series_col, series_val, odk_val) -> int:
    series_idx = game_df[
        (game_df[series_col] == series_val) & (game_df['odk'] == odk_val)
    ].index

    if len(series_idx) == 0:
        return 0

    last_play = game_df.loc[series_idx, 'play'].max()
    results   = game_df.loc[series_idx, 'result'].fillna('').str.lower()

    k_after      = game_df[(game_df['play'] > last_play) & (game_df['odk'] == 'K')].head(2)
    k_play_types = k_after['play_type'].fillna('').str.strip().str.lower()
    k_results    = k_after['result'].fillna('').str.strip().str.lower()

    if results.str.contains('def td').any():
        return -7
    if results.str.contains('td').any():
        return 7
    if (k_results.str.contains('good') & k_play_types.isin(['fg', 'fg block'])).any():
        return 3
    return 0


def _assign_score_event(row) -> int:
    result    = str(row['result']).strip().lower()
    play_type = str(row['play_type']).strip().lower()

    if result in OFF_TD_RESULTS:
        return 7
    if result in DEF_TD_RESULTS:
        return -7
    if 'td' in result:
        return -7 if 'def' in result else 7
    if play_type in ['fg', 'fg block'] and result == 'good':
        return 3
    if play_type in ['punt rec', 'ko rec'] and result == 'td':
        return 7
    if play_type in ['punt', 'ko'] and result == 'td':
        return -7
    return 0
