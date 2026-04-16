"""
play_success_model.py
Trinity University Football Analytics

Defines play success using the EPA framework and generates coaching-facing
summaries by down, distance, play type, personnel, and field position.
Called by 04_play_success_model.ipynb.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ── Success Definition ─────────────────────────────────────────────────────────

def add_success_flag(all_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a binary success column to the dataset.

    A play is successful if EPA > 0 — it outperformed the expected points
    for that game situation. This is context-aware: a 3-yard gain on
    3rd-and-2 is successful; the same gain on 3rd-and-10 is not.

    Args:
        all_df: Full dataset with epa column from ep_model pipeline.

    Returns:
        DataFrame with success column (1 = successful, 0 = not, NaN = special teams).
    """
    od_mask = all_df['odk'].isin(['o', 'd'])
    all_df.loc[od_mask, 'success'] = (all_df.loc[od_mask, 'epa'] > 0).astype(int)
    return all_df


# ── Summary Tables ─────────────────────────────────────────────────────────────

def success_by_down(all_df: pd.DataFrame, unit: str = 'o') -> pd.DataFrame:
    """
    Success rate and mean EPA by down for a given unit.

    Args:
        all_df: Dataset with success and epa columns.
        unit:   'o' for offense, 'd' for defense.

    Returns:
        DataFrame with columns: dn, plays, mean_epa, success_rate.
    """
    od_df = _get_unit_df(all_df, unit)
    return (
        od_df.groupby('dn')
        .agg(plays=('epa', 'count'), mean_epa=('epa', 'mean'), success_rate=('success', 'mean'))
        .round(3)
        .reset_index()
    )


def success_by_play_type(all_df: pd.DataFrame, unit: str = 'o', min_plays: int = 10) -> pd.DataFrame:
    """
    Success rate and mean EPA by play type.

    Args:
        all_df:    Dataset with success and epa columns.
        unit:      'o' for offense, 'd' for defense.
        min_plays: Minimum play count to include in results.

    Returns:
        DataFrame sorted by mean_epa descending.
    """
    od_df = _get_unit_df(all_df, unit)
    summary = (
        od_df.groupby('play_type')
        .agg(plays=('epa', 'count'), mean_epa=('epa', 'mean'), success_rate=('success', 'mean'))
        .round(3)
        .reset_index()
    )
    return summary[summary['plays'] >= min_plays].sort_values('mean_epa', ascending=False)


def success_by_personnel(all_df: pd.DataFrame, unit: str = 'o', min_plays: int = 10) -> pd.DataFrame:
    """
    Success rate and mean EPA by personnel grouping.

    Args:
        all_df:    Dataset with success and epa columns.
        unit:      'o' for offense, 'd' for defense.
        min_plays: Minimum play count to include in results.

    Returns:
        DataFrame sorted by plays descending.
    """
    od_df = _get_unit_df(all_df, unit)
    summary = (
        od_df.groupby('personnel')
        .agg(plays=('epa', 'count'), mean_epa=('epa', 'mean'), success_rate=('success', 'mean'))
        .round(3)
        .reset_index()
    )
    return summary[summary['plays'] >= min_plays].sort_values('plays', ascending=False)


def success_by_field_position(all_df: pd.DataFrame, unit: str = 'o') -> pd.DataFrame:
    """
    Mean EPA by field position bucket (yards to end zone).

    Args:
        all_df: Dataset with yards_to_go and epa columns.
        unit:   'o' for offense, 'd' for defense.

    Returns:
        DataFrame with field position buckets and mean EPA.
    """
    od_df = _get_unit_df(all_df, unit).copy()
    od_df['yards_to_go'] = pd.to_numeric(od_df['yards_to_go'], errors='coerce')

    bins   = [0, 20, 40, 60, 80, 100]
    labels = ['Red Zone (0-20)', '21-40', '41-60', '61-80', 'Own Territory (81-100)']
    od_df['field_zone'] = pd.cut(od_df['yards_to_go'], bins=bins, labels=labels)

    return (
        od_df.groupby('field_zone')
        .agg(plays=('epa', 'count'), mean_epa=('epa', 'mean'), success_rate=('success', 'mean'))
        .round(3)
        .reset_index()
    )


def season_epa_trend(all_df: pd.DataFrame) -> pd.DataFrame:
    """
    Mean EPA by season and unit — tracks performance improvement over time.

    Args:
        all_df: Full dataset with epa and game_date columns.

    Returns:
        DataFrame with columns: season, odk, mean_epa.
    """
    od_df = all_df[all_df['odk'].isin(['o', 'd'])].copy()
    od_df['season'] = pd.to_datetime(od_df['game_date'], errors='coerce').dt.year
    return (
        od_df.groupby(['season', 'odk'])['epa']
        .mean().round(3).reset_index()
        .rename(columns={'epa': 'mean_epa'})
    )


def top_plays(all_df: pd.DataFrame, unit: str = 'o', n: int = 10, highest: bool = True) -> pd.DataFrame:
    """
    Return the highest or lowest EPA plays — most impactful moments in the dataset.

    Args:
        all_df:   Full dataset with epa column.
        unit:     'o' for offense, 'd' for defense.
        n:        Number of plays to return.
        highest:  If True, return highest EPA plays. If False, return lowest.

    Returns:
        DataFrame of top/bottom n plays by EPA.
    """
    od_df = _get_unit_df(all_df, unit)
    cols  = ['game_date', 'odk', 'dn', 'dist', 'play_type', 'result', 'gn_ls', 'ep', 'epa']
    cols  = [c for c in cols if c in od_df.columns]
    return od_df.nlargest(n, 'epa')[cols] if highest else od_df.nsmallest(n, 'epa')[cols]


# ── Visualizations ─────────────────────────────────────────────────────────────

def plot_epa_distribution(all_df: pd.DataFrame) -> None:
    """Side-by-side EPA histograms for offense and defense."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, unit, color, label in zip(
        axes, ['o', 'd'], ['steelblue', 'firebrick'], ['Offense', 'Defense']
    ):
        unit_epa = all_df[all_df['odk'] == unit]['epa'].dropna()
        ax.hist(unit_epa, bins=40, color=color, alpha=0.7, edgecolor='white')
        ax.axvline(unit_epa.mean(), color='black', linestyle='--',
                   label=f'Mean: {unit_epa.mean():.3f}')
        ax.set_title(f'{label} EPA Distribution', fontsize=13)
        ax.set_xlabel('EPA', fontsize=11)
        ax.set_ylabel('Play Count', fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    plt.suptitle('EPA Distribution by Unit', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()


def plot_success_heatmap(all_df: pd.DataFrame, unit: str = 'o') -> None:
    """
    Heatmap of success rate by down and distance bucket.

    Args:
        all_df: Dataset with success, dn, and dist columns.
        unit:   'o' for offense, 'd' for defense.
    """
    od_df = _get_unit_df(all_df, unit).copy()
    od_df['dist'] = pd.to_numeric(od_df['dist'], errors='coerce')

    dist_bins   = [0, 3, 7, 10, 15, 100]
    dist_labels = ['0-3', '4-7', '8-10', '11-15', '15+']
    od_df['dist_bucket'] = pd.cut(od_df['dist'], bins=dist_bins, labels=dist_labels)

    heatmap_data = od_df.groupby(['dn', 'dist_bucket'])['success'].mean().unstack()

    label = 'Offense' if unit == 'o' else 'Defense'
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(
        heatmap_data, annot=True, fmt='.2f', cmap='RdYlGn',
        vmin=0, vmax=1, ax=ax, linewidths=0.5
    )
    ax.set_title(f'{label} Success Rate by Down & Distance', fontsize=14)
    ax.set_xlabel('Distance to First Down', fontsize=12)
    ax.set_ylabel('Down', fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_season_epa_trend(all_df: pd.DataFrame) -> None:
    """Line chart of mean EPA per season for offense and defense."""
    trend = season_epa_trend(all_df)
    fig, ax = plt.subplots(figsize=(10, 5))
    for unit, color, label in zip(['o', 'd'], ['steelblue', 'firebrick'], ['Offense', 'Defense']):
        data = trend[trend['odk'] == unit]
        ax.plot(data['season'], data['mean_epa'], marker='o', color=color,
                linewidth=2, label=label)
    ax.axhline(0, color='black', linestyle='--', alpha=0.4)
    ax.set_title('Mean EPA by Season', fontsize=14)
    ax.set_xlabel('Season', fontsize=12)
    ax.set_ylabel('Mean EPA', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ── Full Pipeline ──────────────────────────────────────────────────────────────

def run_play_success_pipeline(all_df: pd.DataFrame) -> dict:
    """
    Run the complete play success analysis and return all summary tables.

    Args:
        all_df: Full dataset with ep and epa from ep_model pipeline.

    Returns:
        Dict of summary DataFrames:
            'success_by_down', 'success_by_play_type', 'success_by_personnel',
            'success_by_field_position', 'season_trend',
            'top_plays_offense', 'top_plays_defense'

    Example:
        results = run_play_success_pipeline(all_df)
        print(results['success_by_down'])
    """
    all_df = add_success_flag(all_df)

    results = {
        'success_by_down':           success_by_down(all_df, 'o'),
        'success_by_play_type':      success_by_play_type(all_df, 'o'),
        'success_by_personnel':      success_by_personnel(all_df, 'o'),
        'success_by_field_position': success_by_field_position(all_df, 'o'),
        'season_trend':              season_epa_trend(all_df),
        'top_plays_offense':         top_plays(all_df, 'o', n=10, highest=True),
        'top_plays_defense':         top_plays(all_df, 'd', n=10, highest=True),
    }

    for name, df in results.items():
        print(f"\n{'='*60}")
        print(f"{name.upper().replace('_', ' ')}")
        print(f"{'='*60}")
        print(df.to_string(index=False))

    return results


# ── Private Helpers ────────────────────────────────────────────────────────────

def _get_unit_df(all_df: pd.DataFrame, unit: str) -> pd.DataFrame:
    """Filter to a single unit and drop rows with missing EPA."""
    return (
        all_df[all_df['odk'] == unit]
        .dropna(subset=['epa'])
        .copy()
    )
