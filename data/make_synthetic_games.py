"""
Generate the synthetic TU_Games dataset.

The real TU_Games folder holds one Hudl play-list export per game. Those files
belong to Trinity University Athletics and stay private. This script builds a
stand-in folder with the same file names, the same column layouts, and the
same tagging conventions (ODK codes, result labels, play types, yard-line sign
convention) so the notebooks run end to end on a fresh clone.

What is kept from the real folder
  * file names  -> teams, dates, home/away, W/L and final score (public record)
  * column layout of each file (the real exports come in several schema variants,
    and the notebook cleaning code has to handle all of them)

What is simulated
  * every play: down, distance, field position, gain, result, drive flow,
    kicks, scores. Plays come from a simple game simulator; no real play rows
    are copied. For each game, many simulations are run and the one whose
    final score is closest to the real score is kept.
  * every play-call field, for both the offense and the defense: personnel,
    formation, motion, protection, play name, blocking scheme, front, stunt,
    coverage, blitz. All of these use invented vocabularies that do not appear
    in the real data. Offensive and defensive calls use separate word lists,
    and the file team's offense and its opponents' offenses use different
    vocabularies too.
  * free-text fields (TITLE, RETURN NAME/TYPE) that carried player names in
    the real exports are blank or generic.

Usage
    python make_synthetic_games.py --schema-from <real TU_Games dir> --out TU_Games_synthetic
    python make_synthetic_games.py --out TU_Games_synthetic        # uses bundled schema map
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Column layouts. The real exports come in four layouts; each synthetic file
# uses the same layout as the real file it stands in for.
# ─────────────────────────────────────────────────────────────────────────────
BASE = ['TITLE', 'PLAY #', 'ODK', 'HASH', 'DN', 'DIST', 'YARD LN', 'GN/LS', 'RESULT',
        'PLAY TYPE', 'PERSONNEL', 'OFF FORM', 'MOTION', 'PROTECTION', 'OFF PLAY', 'BLITZ',
        'DEF FRONT', 'DEF STUNT', 'COVERAGE', 'FORM TREE', 'RB GAP HIT', 'SERIES',
        'O-LINE SCHEME', 'FILM', 'COV: LOADED/BALANCED', 'PRACTICE', 'PASS ZONE']
LOWER = {'TITLE': 'title', 'PLAY #': 'play_', 'ODK': 'odk', 'HASH': 'hash', 'DN': 'dn',
         'DIST': 'dist', 'YARD LN': 'yard_ln', 'GN/LS': 'gn_ls', 'RESULT': 'result',
         'PLAY TYPE': 'play_type', 'PERSONNEL': 'personnel', 'OFF FORM': 'off_form',
         'MOTION': 'motion', 'PROTECTION': 'protection', 'OFF PLAY': 'off_play',
         'BLITZ': 'blitz', 'DEF FRONT': 'def_front', 'DEF STUNT': 'def_stunt',
         'COVERAGE': 'coverage', 'FORM TREE': 'form_tree', 'RB GAP HIT': 'rb_gap_hit',
         'SERIES': 'series', 'O-LINE SCHEME': 'oline_scheme', 'FILM': 'film',
         'COV: LOADED/BALANCED': 'cov_loaded_balanced', 'PRACTICE': 'practice',
         'PASS ZONE': 'pass_zone'}
KICK_COLS = ['TITLE', 'PLAY #', 'ODK', 'KICK TYPE', 'RETURN NAME/TYPE', 'HASH', 'DN', 'DIST',
             'YARD LN', 'GN/LS', 'RESULT', 'PLAY TYPE', 'PERSONNEL', 'OFF FORM', 'MOTION',
             'PROTECTION', 'OFF PLAY', 'BLITZ', 'COVERAGE', 'DEF FRONT', 'DEF STUNT',
             'FORM TREE', 'RB GAP HIT', 'SERIES', 'O-LINE SCHEME', 'FILM',
             'COV: LOADED/BALANCED', 'PRACTICE', 'PASS ZONE']
SCORE_COLS = ['OFF_SERIES', 'DEF_SERIES', 'POINTS', 'POINT_DIFFERENTIAL', 'TEAM_SCORE',
              'OPPONENT_SCORE']
LAYOUTS = {
    'upper27': BASE,
    'lower27': [LOWER[c] for c in BASE],
    'lower28': ['Unnamed: 0'] + [LOWER[c] for c in BASE],
    'kick29': KICK_COLS,
    'kick35': KICK_COLS + SCORE_COLS,
}

# ─────────────────────────────────────────────────────────────────────────────
# Invented play-call vocabularies (none of these strings occur in the real data;
# check_vocab() verifies that when the real folder is available).
# ─────────────────────────────────────────────────────────────────────────────
# Offense of the team named first in the file (the "O" rows)
HOME_OFF = dict(
    personnel={'COMET': (0.55, 1.05), 'RAVEN': (0.35, 1.00), 'ANVIL': (0.70, 0.97),
               'LYNX': (0.25, 1.08), 'BISON': (0.85, 0.92), 'ORCA': (0.45, 1.02)},
    formations={'COMET': ['BLAZE', 'CEDAR', 'DELTA'], 'RAVEN': ['EMBER', 'FROST', 'CEDAR'],
                'ANVIL': ['GRANITE', 'HARBOR'], 'LYNX': ['IVORY', 'FROST', 'EMBER'],
                'BISON': ['GRANITE', 'JUNIPER'], 'ORCA': ['DELTA', 'HARBOR', 'BLAZE']},
    tree={'BLAZE': '3 WIDE', 'CEDAR': '2 WIDE', 'DELTA': '3 WIDE', 'EMBER': '4 WIDE',
          'FROST': '4 WIDE', 'GRANITE': '1 WIDE', 'HARBOR': '2 WIDE', 'IVORY': 'EMPTY 5',
          'JUNIPER': 'NO WIDE'},
    motion=['ZIP', 'ROCKET TO', 'TRADE PLACES', 'FLASH', 'PENDULUM', 'TRAVEL'],
    run=['SAWBLADE', 'CANYON', 'TORCH', 'LANTERN', 'RIVET', 'QUARRY'],
    pas=['SKYLINE', 'MARLIN', 'COMPASS', 'GLACIER', 'VELVET', 'HALO', 'PIVOT', 'KITE'],
    protection=['WALL 5', 'WALL 6', 'SHIELD', 'GATE LEFT', 'GATE RIGHT'],
    run_scheme=['FLOW', 'SURGE', 'PIN AND SWING', 'ARROWHEAD', 'HINGE'],
    pass_scheme=['HALF WALL', 'FULL WALL', 'BOOT SET', 'QUICK SET'],
    gap=['LANE 1', 'LANE 2', 'LANE 3', 'LANE 4', 'CUTBACK'],
)
# Offenses of the opponents (the "D" rows)
AWAY_OFF = dict(
    personnel={'PX-1': (0.50, 1.0), 'PX-2': (0.65, 1.0), 'PX-3': (0.35, 1.0),
               'PX-4': (0.80, 1.0), 'PX-5': (0.25, 1.0)},
    formations={'PX-1': ['NORTH', 'SOUTH'], 'PX-2': ['EAST', 'NORTH'], 'PX-3': ['WEST', 'EAST'],
                'PX-4': ['CENTRAL'], 'PX-5': ['WEST', 'SOUTH']},
    tree={'NORTH': '3 WIDE', 'SOUTH': '2 WIDE', 'EAST': '2 WIDE', 'WEST': '4 WIDE',
          'CENTRAL': '1 WIDE'},
    motion=['GLIDE PATH', 'ORBITER', 'SKIP STEP'],
    run=['OPP RUN A', 'OPP RUN B', 'OPP RUN C', 'OPP OPTION'],
    pas=['OPP QUICK', 'OPP DROPBACK', 'OPP ACTION', 'OPP SCREEN'],
)
# The file team's defense (called on the "D" rows)
HOME_DEF = dict(
    front=['SUMMIT', 'VALLEY', 'RIDGE', 'MESA', 'SUMMIT PINCH', 'VALLEY WIDE'],
    coverage=['CLOUD', 'CANOPY', 'ANCHOR', 'ISLAND', 'TIDE', 'MIRROR'],
    blitz=['TEMPEST', 'FLURRY', 'HAILSTONE', 'BOLT', 'GUST'],
    stunt=['SWIRL', 'CORKSCREW', 'KNOT', 'SPLIT'],
)
# Scouted opponent defense (recorded on the "O" rows)
AWAY_DEF = dict(
    front=['DX EVEN', 'DX ODD', 'DX HEAVY', 'DX LIGHT'],
    coverage=['DX DEEP 2', 'DX DEEP 3', 'DX MAN', 'DX QUARTERS'],
    blitz=['DX PRESSURE', 'DX SIM'],
    stunt=['DX GAME'],
)
KICK_TYPE = {'KO': 'KICKOFF', 'KO Rec': 'KICK RET', 'Punt': 'PUNT TEAM', 'Punt Rec': 'PUNT RET',
             'Extra Pt.': 'XP', 'Extra Pt. Block': 'XP BLOCK', 'FG': 'FIELD GOAL',
             'FG Block': 'FG RUSH', '2 Pt.': 'XP', '2 Pt. Defend': 'XP BLOCK'}

FNAME = re.compile(r'^([^_]+)_([^_]+)_\((W|L)_([HA])\)(\d+)-(\d+)_(\d{4}-\d{2}-\d{2})')


def yard_ln(ytg: int) -> int:
    """Hudl convention: own half negative (own 30 -> -30), opponent half positive."""
    ytg = int(ytg)
    return ytg if ytg <= 50 else -(100 - ytg)


class Game:
    def __init__(self, rng, pts_target, style):
        self.rng = rng
        self.rows = []
        self.score = {'O': 0, 'D': 0}
        # offensive efficiency nudged by the real final score so simulated
        # scores land near the real ones
        self.eff = {s: float(np.clip(0.80 + pts_target[s] / 60, 0.8, 1.9)) for s in 'OD'}
        self.style = style
        self.pass_lean = {'O': rng.normal(0, 0.08), 'D': rng.normal(0, 0.08)}

    # ── helpers ──────────────────────────────────────────────────────────────
    def hash_(self):
        if self.rng.random() > self.style['hash_rate']:
            return np.nan
        return self.rng.choice(['L', 'R', 'M'], p=[0.4, 0.4, 0.2])

    def add(self, **kw):
        row = {'ODK': kw.pop('odk'), 'HASH': self.hash_()}
        row.update(kw)
        self.rows.append(row)
        return row

    def calls(self, side, run):
        """Invented play-call tags for an offensive snap by `side`."""
        rng = self.rng
        voc = HOME_OFF if side == 'O' else AWAY_OFF
        dvoc = AWAY_DEF if side == 'O' else HOME_DEF
        names = list(voc['personnel'])
        # run-heavy personnel more likely on run downs
        w = np.array([voc['personnel'][p][0] if run else 1 - voc['personnel'][p][0] for p in names])
        pers = rng.choice(names, p=w / w.sum())
        base = rng.choice(voc['formations'][pers])
        form = f"{base} {rng.choice(['RIGHT', 'LEFT'])}" if rng.random() < 0.85 else base
        tag = {'PERSONNEL': pers, 'OFF FORM': form}
        if rng.random() < 0.55:
            tag['FORM TREE'] = voc['tree'][base]
        if rng.random() < 0.18:
            tag['MOTION'] = rng.choice(voc['motion'])
        if rng.random() < 0.85:
            tag['OFF PLAY'] = rng.choice(voc['run'] if run else voc['pas'])
        if side == 'O':
            if not run and rng.random() < 0.3:
                tag['PROTECTION'] = rng.choice(voc['protection'])
            if rng.random() < 0.4:
                tag['O-LINE SCHEME'] = rng.choice(voc['run_scheme'] if run else voc['pass_scheme'])
            if run and rng.random() < 0.05:
                tag['RB GAP HIT'] = rng.choice(voc['gap'])
            if not run and rng.random() < 0.08:
                tag['PASS ZONE'] = float(rng.integers(1, 10))
        # defensive tags
        if rng.random() < (0.55 if side == 'D' else 0.35):
            tag['DEF FRONT'] = rng.choice(dvoc['front'])
        if rng.random() < (0.6 if side == 'D' else 0.4):
            tag['COVERAGE'] = rng.choice(dvoc['coverage'])
        if rng.random() < 0.15:
            tag['BLITZ'] = rng.choice(dvoc['blitz'])
        if rng.random() < 0.08:
            tag['DEF STUNT'] = rng.choice(dvoc['stunt'])
        if side == 'O' and rng.random() < 0.004:
            tag['COV: LOADED/BALANCED'] = rng.choice(['LOADED', 'BALANCED'])
        mult = voc['personnel'][pers][1] if side == 'O' else 1.0
        return tag, mult

    # ── kicking plays ────────────────────────────────────────────────────────
    def kickoff(self, kicker):
        """Returns (receiving side, receiving ytg) or None if kicking team keeps it."""
        rng = self.rng
        receiver = 'D' if kicker == 'O' else 'O'
        ptype = 'KO' if kicker == 'O' else 'KO Rec'
        u = rng.random()
        if u < 0.004:
            self.add(odk='K', DN=0, **{'YARD LN': -35, 'GN/LS': 65, 'RESULT': 'TD', 'PLAY TYPE': ptype})
            self.score[receiver] += 6
            self.pat(receiver)
            return self.kickoff(receiver)
        if u < 0.45:
            self.add(odk='K', DN=0, **{'YARD LN': -35, 'GN/LS': 0, 'RESULT': 'Touchback', 'PLAY TYPE': ptype})
            return receiver, 75
        if u < 0.49:
            self.add(odk='K', DN=0, **{'YARD LN': -35, 'GN/LS': 0, 'RESULT': 'Out of Bounds', 'PLAY TYPE': ptype})
            return receiver, 65
        ret = int(np.clip(rng.normal(22, 8), 3, 55))
        start = int(np.clip(rng.normal(8, 4), 0, 20)) + ret
        self.add(odk='K', DN=0, **{'YARD LN': -35, 'GN/LS': ret, 'RESULT': 'Return', 'PLAY TYPE': ptype})
        return receiver, int(np.clip(100 - start, 45, 99))

    def pat(self, side):
        rng = self.rng
        two = rng.random() < 0.04
        if two:
            ptype = '2 Pt.' if side == 'O' else '2 Pt. Defend'
            good = rng.random() < 0.45
            self.add(odk='K', DN=0, **{'YARD LN': 3, 'GN/LS': 3 if good else 0,
                                       'RESULT': 'Good' if good else 'No Good', 'PLAY TYPE': ptype})
            self.score[side] += 2 if good else 0
        else:
            ptype = 'Extra Pt.' if side == 'O' else 'Extra Pt. Block'
            good = rng.random() < 0.93
            self.add(odk='K', DN=0, **{'YARD LN': 3, 'GN/LS': 0,
                                       'RESULT': 'Good' if good else 'No Good', 'PLAY TYPE': ptype})
            self.score[side] += 1 if good else 0

    def punt(self, side, ytg, dist):
        rng = self.rng
        other = 'D' if side == 'O' else 'O'
        ptype = 'Punt' if side == 'O' else 'Punt Rec'
        gross = int(np.clip(rng.normal(37, 7), 15, 60))
        land = ytg - gross
        base = {'DN': 4, 'DIST': dist, 'YARD LN': yard_ln(ytg), 'PLAY TYPE': ptype}
        if land <= 0:
            self.add(odk='K', **base, **{'GN/LS': 0, 'RESULT': 'Touchback'})
            return other, 80
        u = rng.random()
        if u < 0.004:
            self.add(odk='K', **base, **{'GN/LS': 100 - land, 'RESULT': 'TD'})
            self.score[other] += 6
            self.pat(other)
            return self.kickoff(other)
        if u < 0.35:
            ret = int(np.clip(rng.gamma(2.0, 3.5), 0, 40))
            self.add(odk='K', **base, **{'GN/LS': ret, 'RESULT': 'Return'})
            new = 100 - land + ret
        else:
            res = rng.choice(['Fair Catch', 'Downed', 'Out of Bounds'], p=[0.45, 0.35, 0.20])
            self.add(odk='K', **base, **{'GN/LS': 0, 'RESULT': res})
            new = 100 - land
        return (other, int(np.clip(new, 1, 99))) if new < 100 else (other, 80)

    def field_goal(self, side, ytg, dist):
        rng = self.rng
        other = 'D' if side == 'O' else 'O'
        kick = ytg + 17
        p = float(np.clip(1.30 - 0.021 * kick, 0.15, 0.97))
        good = rng.random() < p
        ptype = 'FG' if side == 'O' else 'FG Block'
        self.add(odk='K', DN=4, DIST=dist, **{'YARD LN': yard_ln(ytg), 'GN/LS': 0,
                                              'RESULT': 'Good' if good else 'No Good', 'PLAY TYPE': ptype})
        if good:
            self.score[side] += 3
            return self.kickoff(side)
        return other, int(np.clip(100 - (ytg + 7), 1, 80))

    # ── one drive ────────────────────────────────────────────────────────────
    def drive(self, side, ytg, budget):
        rng = self.rng
        other = 'D' if side == 'O' else 'O'
        eff = self.eff[side]
        down, dist = 1, min(10, ytg)
        first = True
        while True:
            if len(self.rows) >= budget:
                return None
            # occasional timeout / blank rows
            if rng.random() < 0.025:
                self.add(odk=side if rng.random() < 0.8 else np.nan, DN=down, DIST=dist,
                         **{'YARD LN': yard_ln(ytg), 'GN/LS': 0, 'RESULT': 'Timeout'})
            if rng.random() < 0.035:
                self.add(odk='S', **{'DN': np.nan})
            # 4th down decision
            if down == 4:
                if dist <= 3 and ytg <= 60 and rng.random() < 0.75:
                    pass  # go for it
                elif ytg <= 33:
                    return self.field_goal(side, ytg, dist)
                elif ytg <= 50 and dist <= 6 and rng.random() < 0.55:
                    pass
                else:
                    return self.punt(side, ytg, dist)
            dn_tag = down
            if first and side == 'D' and rng.random() < 0.6:
                dn_tag = 0
            first = False
            # play selection
            if down == 1:
                p_pass = 0.45
            elif down == 2:
                p_pass = 0.40 if dist <= 4 else 0.58
            else:
                p_pass = 0.30 if dist <= 2 else (0.60 if dist <= 5 else 0.82)
            p_pass = float(np.clip(p_pass + self.pass_lean[side], 0.05, 0.95))
            is_pass = rng.random() < p_pass
            tag, mult = self.calls(side, not is_pass)
            base = dict(DN=dn_tag, DIST=dist, **{'YARD LN': yard_ln(ytg)})
            # penalty snap (replayed)
            if rng.random() < 0.045:
                yds = int(rng.choice([-10, -5, -5, 5, 5, 15]))
                self.add(odk=side, **base, **{'GN/LS': yds, 'RESULT': 'Penalty',
                                              'PLAY TYPE': ('Pass' if is_pass else 'Run') if rng.random() < 0.8 else np.nan},
                         **tag)
                ytg = int(np.clip(ytg - yds, 1, 99))
                if yds >= dist:
                    down, dist = 1, min(10, ytg)
                else:
                    dist = max(1, dist - yds)
                continue
            turnover = False
            if is_pass:
                u = rng.random()
                if u < 0.06:
                    gain, res = -int(rng.integers(3, 12)), 'Sack'
                    if rng.random() < 0.06:
                        res, turnover = 'Sack, Fumble', True
                elif u < 0.11:
                    gain, res = int(np.clip(rng.gamma(1.6, 4.0 * eff), -2, 60)), 'Scramble'
                elif u < 0.11 + 0.028 / eff:
                    gain, res, turnover = 0, 'Interception', True
                elif u < 0.52 - 0.06 * (eff - 1):
                    gain, res = 0, ('Dropped' if rng.random() < 0.03 else 'Incomplete')
                else:
                    gain = int(round(rng.gamma(1.7, 4.6 * eff * mult)))
                    if rng.random() < 0.09 * eff:
                        gain += int(rng.integers(12, 55))
                    if rng.random() < 0.06:
                        gain = -int(rng.integers(0, 4))
                    res = 'Complete'
                    if rng.random() < 0.012:
                        res, turnover = 'Complete, Fumble', True
                ptype = 'Pass'
            else:
                if rng.random() < 0.09:
                    gain = -int(rng.integers(1, 5))
                else:
                    gain = int(round(rng.gamma(1.5, 2.3 * eff * mult)))
                    if rng.random() < 0.035 * eff:
                        gain += int(rng.integers(8, 45))
                res, ptype = 'Rush', 'Run'
                if rng.random() < 0.012:
                    res, turnover = 'Fumble', True
            gain = min(gain, ytg)
            # safety
            if ytg - gain >= 100:
                self.add(odk=side, **base, **{'GN/LS': gain, 'RESULT': f'{res.split(",")[0]}, Safety' if res in ('Rush', 'Sack') else 'Safety',
                                              'PLAY TYPE': ptype}, **tag)
                self.score[other] += 2
                return self.kickoff(side)  # free kick by the team that gave up the safety
            # touchdown
            if gain >= ytg and not turnover:
                label = {'Rush': 'Rush, TD', 'Complete': 'Complete, TD', 'Scramble': 'Scramble, TD'}.get(res, 'TD')
                if rng.random() < 0.03:
                    label = 'TD'
                self.add(odk=side, **base, **{'GN/LS': gain, 'RESULT': label, 'PLAY TYPE': ptype}, **tag)
                self.score[side] += 6
                self.pat(side)
                return self.kickoff(side)
            if turnover:
                spot = ytg - gain
                if rng.random() < 0.08:
                    res = {'Interception': 'Interception, Def TD', 'Fumble': 'Fumble, Def TD',
                           'Sack, Fumble': 'Sack, Fumble, Def TD'}.get(res, res)
                if 'Def TD' in res:
                    self.add(odk=side, **base, **{'GN/LS': gain, 'RESULT': res, 'PLAY TYPE': ptype}, **tag)
                    self.score[other] += 6
                    self.pat(other)
                    return self.kickoff(other)
                self.add(odk=side, **base, **{'GN/LS': gain, 'RESULT': res, 'PLAY TYPE': ptype}, **tag)
                ret = int(np.clip(rng.gamma(1.5, 5), 0, 40))
                return other, int(np.clip(100 - spot + ret, 1, 99))
            # regular play
            if gain >= dist and rng.random() < 0.25:
                res = '1st DN'
            self.add(odk=side, **base, **{'GN/LS': gain, 'RESULT': res, 'PLAY TYPE': ptype}, **tag)
            ytg -= gain
            if gain >= dist:
                down, dist = 1, min(10, ytg)
            else:
                down, dist = down + 1, dist - gain
                if down > 4:  # turnover on downs
                    return other, int(np.clip(100 - ytg, 1, 99))

    def play(self, n_rows):
        rng = self.rng
        first_kicker = rng.choice(['O', 'D'])
        state = self.kickoff(first_kicker)
        half = False
        while state is not None and len(self.rows) < n_rows:
            if not half and len(self.rows) >= n_rows // 2:
                half = True
                state = self.kickoff('D' if first_kicker == 'O' else 'O')
                continue
            side, ytg = state
            state = self.drive(side, ytg, n_rows)
        return self.rows


def simulate_game(fname, layout, n_rows, tries=80):
    m = FNAME.match(fname)
    if not m:
        raise ValueError(f'Unrecognised file name: {fname}')
    team_pts, opp_pts = int(m.group(5)), int(m.group(6))
    seed = int(hashlib.md5(fname.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    style = {'hash_rate': float(rng.choice([0.55, 0.75, 0.95]))}
    best, best_err = None, None
    for _ in range(tries):
        g = Game(rng, {'O': team_pts, 'D': opp_pts}, style)
        rows = g.play(n_rows)
        err = abs(g.score['O'] - team_pts) + abs(g.score['D'] - opp_pts)
        if best is None or err < best_err:
            best, best_err = rows, err
        if err <= 3:
            break
    return to_frame(best, layout, rng), best_err


def to_frame(rows, layout, rng):
    df = pd.DataFrame(rows)
    for c in BASE:
        if c not in df:
            df[c] = np.nan
    df['PLAY #'] = np.arange(1, len(df) + 1)
    # sparse SERIES tags and FILM label, like the real exports
    df['SERIES'] = pd.Series(rng.integers(1, 18, len(df)).astype(float), index=df.index).where(rng.random(len(df)) < 0.25)
    if rng.random() < 0.3:
        df['FILM'] = pd.Series('GAME', index=df.index).where(df['ODK'].isin(['O', 'D']))
    titles = ['OPENING SCRIPT', 'TWO MINUTE', 'CUTUP']
    df['TITLE'] = pd.Series(rng.choice(titles, len(df)), index=df.index).where(rng.random(len(df)) < 0.02)
    # remove offensive/defensive tags from kick rows
    k = df['ODK'].isin(['K', 'S'])
    tag_cols = ['PERSONNEL', 'OFF FORM', 'MOTION', 'PROTECTION', 'OFF PLAY', 'BLITZ', 'DEF FRONT',
                'DEF STUNT', 'COVERAGE', 'FORM TREE', 'RB GAP HIT', 'O-LINE SCHEME',
                'COV: LOADED/BALANCED', 'PASS ZONE']
    df.loc[k, tag_cols] = np.nan
    s = df['ODK'] == 'S'
    df.loc[s, ['HASH', 'DN', 'DIST', 'YARD LN', 'GN/LS', 'RESULT', 'PLAY TYPE']] = np.nan
    df = df[BASE]
    if layout in ('kick29', 'kick35'):
        df['KICK TYPE'] = df['PLAY TYPE'].map(KICK_TYPE).where(df['ODK'] == 'K')
        df['RETURN NAME/TYPE'] = np.nan
        for c in SCORE_COLS:
            df[c] = np.nan
        return df[LAYOUTS[layout]]
    if layout.startswith('lower'):
        df = df.rename(columns=LOWER)
        if layout == 'lower28':
            df.insert(0, 'Unnamed: 0', np.arange(len(df)))
        return df[LAYOUTS[layout]]
    return df


def layout_of(cols):
    cols = list(cols)
    for k, v in LAYOUTS.items():
        if cols == v:
            return k
    raise ValueError(f'Unknown layout: {cols[:5]}...')


def check_vocab(real_dir, out_dir):
    """Confirm no play-call string from the real exports appears in the synthetic files."""
    fields = ['PERSONNEL', 'OFF FORM', 'MOTION', 'PROTECTION', 'OFF PLAY', 'BLITZ', 'DEF FRONT',
              'DEF STUNT', 'COVERAGE', 'FORM TREE', 'RB GAP HIT', 'O-LINE SCHEME', 'TITLE',
              'RETURN NAME/TYPE', 'KICK TYPE']
    lf = {LOWER.get(f, f) for f in fields} | set(fields)

    def vocab(d):
        out = set()
        for p in Path(d).glob('*.xlsx'):
            x = pd.read_excel(p)
            for c in x.columns:
                if c in lf:
                    out |= {str(v).strip().upper() for v in x[c].dropna().unique()}
        return out
    overlap = vocab(real_dir) & vocab(out_dir)
    return overlap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--schema-from', help='real TU_Games folder (reads only file names, layouts and row counts)')
    ap.add_argument('--out', default=str(Path(__file__).parent / 'TU_Games_synthetic'))
    ap.add_argument('--manifest', default=str(Path(__file__).parent / 'synthetic_manifest.json'))
    args = ap.parse_args()

    if args.schema_from:
        manifest = []
        for p in sorted(Path(args.schema_from).glob('*.xlsx')):
            x = pd.read_excel(p)
            manifest.append({'file': p.name, 'layout': layout_of(x.columns), 'rows': len(x)})
        json.dump(manifest, open(args.manifest, 'w'), indent=1)
    else:
        manifest = json.load(open(args.manifest))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    errs = []
    for item in manifest:
        df, err = simulate_game(item['file'], item['layout'], item['rows'])
        df.to_excel(out / item['file'], index=False)
        errs.append(err)
    print(f'Wrote {len(manifest)} games to {out}  (median score gap vs real final: {np.median(errs):.0f} pts)')

    if args.schema_from:
        ov = check_vocab(args.schema_from, out)
        print('Play-call strings shared with real data:', sorted(ov) if ov else 'none')


if __name__ == '__main__':
    main()
