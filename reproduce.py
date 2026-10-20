#!/usr/bin/env python3
"""Rebuild the main-study figures and tables from included per-agent results.

No RL runtime, network access, model loading or training. NumPy/Matplotlib are
needed only by the plotting functions. All numerical inputs remain in raw units.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics as st

GROUPS = {'Ranked (8 games)': tuple(range(8)),
          'Beyond anchor (4 games)': tuple(range(8, 12)),
          'Random passing (4 games)': tuple(range(12, 16))}
TASKS = ('acrobot', 'lander', 'breakout', 'invaders', 'qbert', 'seaquest')
TITLES = ('Acrobot', 'LunarLander', 'Breakout', 'Space Invaders', 'Qbert', 'Seaquest')
METHODS = ('initial', 'gremlin', 'disco103', 'dqn', 'double-dqn')
LABELS = ('Initial', 'GREMLIN', 'Disco103', 'DQN', 'Double DQN')
COLORS = ('#777777', '#008577', '#c47d0a', '#355ea5', '#885ca8')
ALIASES = {'initial-leaky': 'initial', 'gremlin1000': 'gremlin', 'gremlin-1000': 'gremlin'}


def normalized_panel(rows, tasks, methods, seeds, metrics):
    lookup = {}
    for row in rows:
        key = (row['task'], ALIASES.get(row['method'], row['method']), row['seed'])
        if key in lookup:
            raise ValueError('duplicate task/method/seed')
        lookup[key] = row
    expected = {(t, m, s) for t in tasks for m in methods for s in seeds}
    if set(lookup) != expected:
        raise ValueError('incomplete or unexpected panel')
    points = []
    for task in tasks:
        for metric in metrics:
            def values(method):
                result = [lookup[task, method, s][metric] for s in seeds]
                if not all(math.isfinite(x) for x in result):
                    raise ValueError('nonfinite result')
                return result
            initial, dqn = st.mean(values('initial')), st.mean(values('dqn'))
            gap = dqn - initial
            if gap <= 1e-10:
                raise ValueError('normalization requires a positive DQN–initial gap')
            for method in methods:
                raw = values(method)
                points.append(dict(task=task, metric=metric, method=method,
                    seeds=list(seeds), raw=raw, initial_mean=initial, dqn_mean=dqn,
                    normalized=[(x-initial)/gap for x in raw]))
    return points


def summarize(generated, human, anchor=None):
    g = normalized_panel(generated, range(16), METHODS, range(2), ('endpoint', 'auc'))
    h = normalized_panel(human, TASKS, METHODS[:-1], range(5),
                         ('final_return', 'mean_curve_return'))
    groups = []
    for label, tasks in GROUPS.items():
        for metric in ('endpoint', 'auc'):
            for method in METHODS:
                selected = [p for p in g if p['task'] in tasks and p['metric'] == metric
                            and p['method'] == method]
                # Each task has the same weight, AFTER within-task normalization.
                seed_means = [st.mean(p['normalized'][s] for p in selected) for s in range(2)]
                groups.append(dict(group=label, tasks=list(tasks), metric=metric,
                    method=method, mean=st.mean(seed_means), seed_means=seed_means))
    if anchor is not None:
        lookup = {(r['task'], r['seed']): r for r in anchor}
        if len(anchor) != 32 or set(lookup) != {(t, s) for t in range(16) for s in (0, 1)}:
            raise ValueError('incomplete or duplicate anchor panel')
        for task in range(16):
            reference = next(p for p in g if p['task'] == task and p['metric'] == 'endpoint')
            raw = [lookup[task, s]['endpoint'] for s in (0, 1)]
            if not all(math.isfinite(x) for x in raw):
                raise ValueError('nonfinite anchor score')
            g.append(dict(task=task, metric='endpoint', method='anchor', seeds=[0, 1],
                raw=raw, initial_mean=reference['initial_mean'], dqn_mean=reference['dqn_mean'],
                normalized=[(v-reference['initial_mean']) /
                    (reference['dqn_mean']-reference['initial_mean']) for v in raw]))
        for label, tasks in GROUPS.items():
            selected = [p for p in g if p['task'] in tasks and p['method'] == 'anchor']
            means = [st.mean(p['normalized'][s] for p in selected) for s in (0, 1)]
            groups.append(dict(group=label, tasks=list(tasks), metric='endpoint',
                               method='anchor', mean=st.mean(means), seed_means=means))
    return dict(normalization='(score - task mean initial) / (task mean DQN - task mean initial); '
                'fixed across seeds, separate for each metric; no clipping',
                generated=g, human=h, groups=groups)


def save(fig, path):
    import matplotlib.pyplot as plt
    fig.savefig(path, bbox_inches='tight', metadata={'Date': None, 'Creator': 'GREMLIN results'})
    path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    plt.close(fig)


def plots(summary, output):
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/gremlin-reproduce-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size': 11, 'svg.hashsalt': 'gremlin-main-study-v1',
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.4), sharey=True, constrained_layout=True)
    display = [('gremlin', 'GREMLIN', COLORS[1]), ('disco103', 'Disco103', COLORS[2]),
               ('dqn', 'DQN', COLORS[3])]
    if any(p['method'] == 'anchor' for p in summary['groups']):
        display.append(('anchor', 'Anchor', '#885ca8'))
    for col, group in enumerate(GROUPS):
        ax = axes[col]
        for x, (method, label, color) in enumerate(display):
            p = next(p for p in summary['groups'] if p['group'] == group
                     and p['metric'] == 'endpoint' and p['method'] == method)
            lo, hi = min(p['seed_means']), max(p['seed_means'])
            ax.errorbar(x, p['mean'], yerr=[[p['mean']-lo], [hi-p['mean']]],
                        fmt='o', color=color, capsize=4, markersize=8)
        ax.axhline(1, color=COLORS[3], ls='--', lw=1, alpha=.6)
        ax.set(xticks=range(len(display)), xticklabels=[x[1] for x in display],
               xlim=(-.5, len(display)-.5))
        ax.tick_params(axis='x', rotation=32)
        ax.grid(axis='y', alpha=.2)
        ax.set_title(group, fontsize=12)
        if col == 0:
            ax.set_ylabel('Final return\n(rescaled within each game)')
    active = [p for p in summary['groups'] if p['metric'] == 'endpoint'
              and p['method'] in {x[0] for x in display}]
    lo = min(min(p['seed_means']) for p in active)
    hi = max(max(p['seed_means']) for p in active)
    margin = max(.05, (hi-lo)*.14)
    axes[0].set_ylim(lo-margin, hi+margin)
    fig.suptitle('Generated games: mean relative gain in each selection group\n'
                 'DQN = 1; initial teacher = 0 (not plotted). Vertical axes are zoomed.', fontsize=12)
    save(fig, output/'generated-groups.svg')

    fig, ax = plt.subplots(figsize=(11, 5.2), constrained_layout=True)
    for mi, method in enumerate(METHODS[:-1]):
        offset = (mi-1.5)*.16
        for ti, task in enumerate(TASKS):
            p = next(p for p in summary['human'] if p['task'] == task and p['method'] == method
                     and p['metric'] == 'final_return')
            ys = p['normalized']
            ax.scatter(ti+offset+np.linspace(-.035, .035, len(ys)), ys,
                       color=COLORS[mi], alpha=.4, s=16)
            ax.scatter(ti+offset, st.mean(ys), color=COLORS[mi], s=65, marker='D',
                       edgecolors='white', linewidths=.6, label=LABELS[mi] if ti == 0 else None)
    ax.axhline(0, color='#777777', ls=':', lw=1)
    ax.axhline(1, color=COLORS[3], ls='--', lw=1)
    ax.set(xticks=range(6), xticklabels=TITLES, ylabel='Final return, rescaled within each game',
           title='Human-designed games: final performance\nInitial teacher mean = 0; DQN mean = 1')
    ax.grid(axis='y', alpha=.18)
    ax.legend(loc='upper left', ncol=4, frameon=False)
    save(fig, output/'human-endpoints.svg')


def build(data, output):
    generated = json.loads((data/'generated.json').read_text())['rows']
    human = json.loads((data/'human.json').read_text())['rows']
    anchor_path = data/'anchor.json'
    anchor = json.loads(anchor_path.read_text())['rows'] if anchor_path.exists() else None
    summary = summarize(generated, human, anchor)
    output.mkdir(parents=True, exist_ok=True)
    plots(summary, output)
    (output/'plot-data.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    with (output/'scores.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(('panel', 'task', 'metric', 'method', 'seed', 'raw_score', 'display_score'))
        for panel in ('generated', 'human'):
            for p in summary[panel]:
                for seed, raw, norm in zip(p['seeds'], p['raw'], p['normalized']):
                    writer.writerow((panel, p['task'], p['metric'], p['method'], seed, raw, norm))
    names = ['generated.json', 'human.json'] + (['anchor.json'] if anchor is not None else [])
    (output/'provenance.json').write_text(json.dumps({name: hashlib.sha256((data/name).read_bytes()).hexdigest()
                                                    for name in names}, indent=2)+'\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path(__file__).resolve().parent/'data')
    parser.add_argument('--output', type=Path, default=Path('results'))
    args = parser.parse_args()
    result = build(args.data, args.output)
    print(f"Rebuilt two figures and all per-seed scores in {args.output}; no learning agents ran.")
