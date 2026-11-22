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


def measurement_rows(report, panel):
    """Use measured 4M endpoints; retain the early readout without interpolation."""
    if panel != 'human' or report.get('schema') != 'gremlin-human-replacement-4m-v1':
        return report['rows']
    grouped = {}
    for row in report['rows']:
        key = (row['task'], row['method'], row['seed'])
        points = grouped.setdefault(key, {})
        if row['target'] in points:
            raise ValueError('duplicate measured human readout')
        points[row['target']] = row
    rows = []
    for (task, method, seed), points in sorted(grouped.items()):
        if set(points) != {500000, 4000000}:
            raise ValueError('missing measured human budget')
        measured = [points[n] for n in (500000, 4000000)]
        rows.append(dict(task=task, method=method, seed=seed, budget=4000000,
            interactions=measured[-1]['interactions'], final_return=measured[-1]['sample'],
            actual_grid=[p['interactions'] for p in measured],
            mean_returns=[p['sample'] for p in measured], readouts=measured))
    return rows


def normalized_panel(rows, tasks, methods, seeds, metrics, references=None):
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
            initial, dqn = (references[task, metric] if references is not None else
                            (st.mean(values('initial')), st.mean(values('dqn'))))
            gap = dqn - initial
            if gap <= 1e-10:
                raise ValueError('normalization requires a positive DQN–initial gap')
            for method in methods:
                raw = values(method)
                points.append(dict(task=task, metric=metric, method=method,
                    seeds=list(seeds), raw=raw, initial_mean=initial, dqn_mean=dqn,
                    normalized=[(x-initial)/gap for x in raw]))
    return points


def readout_rows(rows, target, metric, policy='sample'):
    """Select measured assessments only; never interpolate missing readouts."""
    if not any('readouts' in row for row in rows):
        return []
    selected = []
    for row in rows:
        points = [p for p in row.get('readouts', []) if p['target'] == target]
        if len(points) != 1:
            raise ValueError(f'missing or duplicate measured {target}-step readout')
        selected.append(dict(task=row['task'], method=row['method'], seed=row['seed'],
                             **{metric: points[0][policy]}))
    return selected


def selection_groups(points, metrics, methods):
    return [dict(group=label, tasks=list(tasks), metric=metric, method=method,
                 **group_scores([p for p in points if p['task'] in tasks and
                                 p['metric'] == metric and p['method'] == method]))
            for label, tasks in GROUPS.items() for metric in metrics for method in methods]


def group_scores(selected):
    """Equal game weights; descriptive bounds invariant to within-game seed labels."""
    values=[p['normalized'] for p in selected]
    return dict(mean=st.mean(st.mean(v) for v in values),
                within_game_range=[st.mean(min(v) for v in values),
                                   st.mean(max(v) for v in values)])


def summarize(generated, human, anchor=None, random_rows=None):
    generated_methods = METHODS if any(r['method'] == 'double-dqn' for r in generated) else METHODS[:-1]
    def metrics(rows, required, optional):
        if not rows or not all(required in r for r in rows):
            raise ValueError('missing endpoint scores')
        present = [optional in r for r in rows]
        if any(present) and not all(present):
            raise ValueError('incomplete optional metric')
        return (required, optional) if all(present) else (required,)
    g_metrics = metrics(generated, 'endpoint', 'auc')
    h_metrics = metrics(human, 'final_return', 'mean_curve_return')
    g = normalized_panel(generated, range(16), generated_methods, range(2), g_metrics)
    h = normalized_panel(human, TASKS, METHODS[:-1], range(5), h_metrics)
    early_g = readout_rows(generated, 500000, 'endpoint')
    early_h = readout_rows(human, 500000, 'final_return')
    def early_panel(rows, final, tasks, methods, seeds, metric):
        references = {(p['task'], metric): (p['initial_mean'], p['dqn_mean'])
                      for p in final if p['metric'] == metric}
        return normalized_panel(rows, tasks, methods, seeds, (metric,), references) if rows else []
    g_early = early_panel(early_g, g, range(16), generated_methods, range(2), 'endpoint')
    h_early = early_panel(early_h, h, TASKS, METHODS[:-1], range(5), 'final_return')
    groups = selection_groups(g, g_metrics, generated_methods)
    groups_early = selection_groups(g_early, ('endpoint',), generated_methods) if g_early else []
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
            if g_early:
                # The fixed anchor does not train: reuse the same assessment in both panels.
                g_early.append(dict(g[-1]))
        groups.extend(selection_groups(g, ('endpoint',), ('anchor',)))
        if g_early:
            groups_early.extend(selection_groups(g_early, ('endpoint',), ('anchor',)))
    summary = dict(normalization='(score - task mean initial) / (task mean DQN - task mean initial); '
                'final-budget reference means fixed across seeds and early/final readouts; '
                'separate for each task and metric; no clipping',
                group_range='mean of within-game minima to mean of within-game maxima; '
                    'descriptive range, not a confidence interval',
                generated=g, human=h, groups=groups, generated_early=g_early,
                human_early=h_early, groups_early=groups_early)
    if random_rows is not None:
        refs = {(r['panel'], r['task']): r for r in random_rows}
        expected = {('generated',t) for t in range(16)} | {('human',t) for t in TASKS}
        if (set(refs) != expected or len(random_rows) != len(expected) or
            any(r['episodes'] != 1024 or not math.isfinite(r['mean']) or
                not math.isfinite(r['sem']) or r['sem'] < 0 for r in refs.values())):
            raise ValueError('invalid uniform-random baseline')
        for points in (g,g_early):
            for p in points:
                base=refs['generated',p['task']]['mean']
                gap=p['dqn_mean']-base
                if gap<=0: raise ValueError('nonpositive DQN-random gap')
                p.update(random_mean=base,normalized=[(v-base)/gap for v in p['raw']])
        for panel,points in [('groups',g),('groups_early',g_early)]:
            summary[panel]=selection_groups(points,('endpoint',),tuple(dict.fromkeys(p['method'] for p in points)))
        for p in h+h_early:
            del p['normalized']
        summary.update(random=random_rows,normalization='Generated: (return - random mean) / (4M DQN mean - random mean), per game. Human: raw episode return.')
    return summary


def save(fig, path):
    import matplotlib.pyplot as plt
    metadata = ({'CreationDate': None, 'ModDate': None, 'Creator': 'GREMLIN results'}
                if path.suffix == '.pdf' else {'Date': None, 'Creator': 'GREMLIN results'})
    fig.savefig(path, bbox_inches='tight', metadata=metadata)
    if path.suffix == '.svg':
        path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    plt.close(fig)


def human_table(summary):
    """LaTeX cells from the same raw seed scores as the figure, never hand-entered."""
    lines=[r'\begin{tabular}{lrrrr}\toprule',
           r'Task & Initial & GREMLIN & Disco103 & DQN\\\midrule']
    for task,title in zip(TASKS,('Acrobot','LunarLander','Breakout RAM','Space Invaders RAM','Qbert RAM','Seaquest RAM'),strict=True):
        cells=[title]
        for method in METHODS[:-1]:
            values=next(p['raw'] for p in summary['human'] if p['task']==task and
                        p['method']==method and p['metric']=='final_return')
            cells.append(r'\shortstack{$'+f'{st.mean(values):.2f}'+r'\pm'+f'{st.stdev(values):.2f}'+
                         r'$\\$('+f'{max(values):.2f}'+r')$}')
        lines.append(' & '.join(cells)+r'\\[3pt]')
    lines[-1]=lines[-1].replace(r'\\[3pt]',r'\\\bottomrule')
    return '\n'.join([*lines,r'\end{tabular}'])+'\n'


def greedy_means(summary, generated, human):
    """Score the same final agents with argmax, on the main figures' original axes."""
    result = {}
    for panel, rows, tasks, seeds, metric in (
            ('generated', generated, range(16), range(2), 'endpoint'),
            ('human', human, TASKS, range(5), 'final_return')):
        refs = {(p['task'], metric): (p['initial_mean'], p['dqn_mean'])
                for p in summary[panel] if p['metric'] == metric}
        points = normalized_panel(readout_rows(rows, 4000000, metric, 'greedy'),
                                  tasks, METHODS[:-1], seeds, (metric,), refs)
        if panel == 'generated':
            result[panel] = {(p['group'], p['method']): p['mean']
                             for p in selection_groups(points, (metric,), METHODS[1:3])}
        else:
            result[panel] = {(p['task'], p['method']): st.mean(p['normalized'])
                             for p in points if p['method'] in METHODS[1:3]}
    return result


def random_figures(summary,out,fmt):
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator
    plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'svg.hashsalt':'gremlin-article-v1',
                         'axes.spines.top':False,'axes.spines.right':False})
    methods=['initial','gremlin','disco103','dqn','anchor']
    colors=['#686868','#008577','#af6500','#355ea5','#885ca8']
    labels=['Initial teacher','GREMLIN','Disco103','DQN','Anchor']
    rand_color='#9b3a49'
    def lookup(panel, metric, field):
        return {(p['task'],p['method']):np.array(p[field]) for p in summary[panel] if p['metric']==metric}
    g=lookup('generated','endpoint','normalized'); early=lookup('generated_early','endpoint','normalized')
    fig,ax=plt.subplots(figsize=(9.8,4.2),layout='constrained')
    for j,(label,tasks) in enumerate(GROUPS.items()):
        for i,(m,c) in enumerate(zip(methods,colors)):
            scores=np.array([g[t,m] for t in tasks])
            mean=scores.mean(); x=j+(i-2)*.145
            ax.errorbar(x,mean,yerr=[[mean-scores.min(1).mean()],[scores.max(1).mean()-mean]],
                        fmt='*' if m=='anchor' else 'o',ms=12 if m=='anchor' else 6,color=c,capsize=3)
            if m!='anchor':
                ax.scatter(x-.035,np.mean([early[t,m] for t in tasks]),marker='x',color=c,s=36)
    ax.axhline(0,color=rand_color,ls=':',lw=1.3);ax.axhline(1,color=colors[3],ls='--',lw=.9)
    ax.set(xticks=range(3),xticklabels=[l.replace(' (','\n(') for l in GROUPS],
           title='Generated games: random actions = 0, DQN at 4M = 1',
           ylabel='Return rescaled within game',xlim=(-.48,2.48))
    ax.grid(axis='y',alpha=.14)
    handles=[Line2D([],[],color=c,label=l) for l,c in zip(labels[:4],colors[:4])]
    handles.append(Line2D([],[],ls='',marker='*',markersize=12,color=colors[4],label=labels[4]))
    handles.extend(Line2D([],[],ls='',marker=m,color='#555',label=l)
                   for m,l in [('o','4M'),('x','500k mean')])
    ax.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,-.16),ncol=4,frameon=False)
    save(fig,out/f'generated-groups.{fmt}')

    h=lookup('human','final_return','raw'); early=lookup('human_early','final_return','raw')
    fig,axes=plt.subplots(2,3,figsize=(10.8,5.95),layout='constrained')
    for ax,t,title in zip(axes.flat,TASKS,TITLES[:2]+tuple(t+' RAM' for t in TITLES[2:])):
        row=next(r for r in summary['random'] if r['panel']=='human' and r['task']==t); base=row['mean']; se=row['sem']
        ax.axhspan(base-2*se,base+2*se,color=rand_color,alpha=.12,linewidth=0)
        ax.axhline(base,color=rand_color,ls=':',lw=1.3)
        ax.axhline(h[t,'dqn'].mean(),color=colors[3],ls='--',lw=.9)
        for i,(m,c) in enumerate(zip(methods[:4],colors)):
            ys=h[t,m]
            ax.scatter(i+np.linspace(.04,.22,5),ys,color=c,alpha=.46,s=14,edgecolor='none')
            ax.scatter(i,ys.mean(),marker='D',s=42,color=c,edgecolor='white',linewidth=.5)
            ax.scatter(i-.14,early[t,m].mean(),marker='x',s=35,color=c)
        ax.set(xticks=range(4),xticklabels=['Initial',*labels[1:4]],title=title,xlim=(-.45,3.47))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5,integer=True))
        ax.grid(axis='y',alpha=.13)
    fig.supylabel('Episode return (game units; higher is better)',fontsize=11)
    handles=[Line2D([],[],ls='',marker=m,color='#555',markersize=s,label=l)
             for m,s,l in [('D',6,'4M mean'),('o',3,'4M seed'),('x',6,'500k mean')]]
    handles.extend([Line2D([],[],ls=':',color=rand_color,label='Random actions (mean ± 2 SE)'),
                    Line2D([],[],ls='--',color=colors[3],label='DQN mean at 4M')])
    fig.legend(handles=handles,loc='outside lower center',ncol=5,frameon=False)
    fig.suptitle('Human games: raw returns',fontsize=14)
    save(fig,out/f'human-endpoints.{fmt}')


def plots(summary, output, output_format='svg', greedy=None):
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/gremlin-reproduce-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    if 'random' in summary and greedy is None:
        return random_figures(summary,output,output_format)
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import numpy as np
    plt.rcParams.update({'font.size': 11, 'svg.hashsalt': 'gremlin-main-study-v1',
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(11, 4.4), constrained_layout=True)
    display = [('gremlin', 'GREMLIN', COLORS[1]), ('disco103', 'Disco103', COLORS[2]),
               ('dqn', 'DQN', COLORS[3])]
    def add_greedy(ax, panel, key, method, x, color):
        if greedy and method in METHODS[1:3]:
            ax.scatter(x, greedy[panel][key, method], marker='s', facecolors='white',
                       edgecolors=color, s=50, linewidths=1.6, zorder=5)
    def greedy_legend(handles):
        handles.append(Line2D([], [], marker='s', markerfacecolor='white',
                              color='#555555', ls='', label='4M greedy mean (learned rules)'))
    if any(p['method'] == 'anchor' for p in summary['groups']):
        display.append(('anchor', 'Anchor', '#885ca8'))
    for gi, group in enumerate(GROUPS):
        for mi, (method, label, color) in enumerate(display):
            p = next(p for p in summary['groups'] if p['group'] == group
                     and p['metric'] == 'endpoint' and p['method'] == method)
            x = gi + (mi-(len(display)-1)/2)*.17
            lo, hi = p['within_game_range']
            ax.errorbar(x, p['mean'], yerr=[[p['mean']-lo], [hi-p['mean']]],
                        fmt='o', color=color, capsize=4, markersize=7)
            add_greedy(ax, 'generated', group, method, x+.044, color)
            if summary['groups_early'] and method != 'anchor':
                early = next(p for p in summary['groups_early'] if p['group'] == group
                             and p['metric'] == 'endpoint' and p['method'] == method)
                ax.scatter(x-.035, early['mean'], marker='x', color=color, s=60,
                           linewidths=1.8, zorder=4)
    ax.axhline(1, color=COLORS[3], ls='--', lw=1, alpha=.6)
    ax.set(xticks=range(len(GROUPS)), xticklabels=[x.replace(' (', '\n(') for x in GROUPS],
           xlim=(-.5, len(GROUPS)-.5), ylabel='Return\n(rescaled within each game)')
    ax.grid(axis='y', alpha=.2)
    handles = [Line2D([], [], color=c, lw=2, label=label) for _, label, c in display]
    handles.append(Line2D([], [], color='#555555', marker='o', ls='', label='4M mean'))
    if summary['groups_early']:
        handles.append(Line2D([], [], color='#555555', marker='x', ls='', label='500k mean'))
    if greedy:
        greedy_legend(handles)
        ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5,-.16),
                  ncol=4, frameon=False, fontsize=10)
    else:
        ax.legend(handles=handles, loc='lower left', ncol=3, frameon=False, fontsize=10)
    active = [p for p in summary['groups'] if p['metric'] == 'endpoint'
              and p['method'] in {x[0] for x in display}]
    early_means = [p['mean'] for p in summary['groups_early']
                   if p['metric'] == 'endpoint' and p['method'] != 'initial']
    overlay = list(greedy['generated'].values()) if greedy else []
    lo = min([p['within_game_range'][0] for p in active] + early_means + overlay)
    hi = max([p['within_game_range'][1] for p in active] + early_means + overlay)
    margin = max(.05, (hi-lo)*.14)
    ax.set_ylim(lo-margin, hi+margin)
    ax.set_title('Generated games: 4M performance, with 500k means marked X\n'
                 'Shared 4M references: DQN = 1; initial teacher = 0 (not plotted). Axes are zoomed.'
                 if summary['groups_early'] else 'Generated games: final relative gain', fontsize=11)
    save(fig, output/f'generated-groups.{output_format}')

    fig, ax = plt.subplots(figsize=(11, 4.8), constrained_layout=True)
    ceiling = max(3.5, math.ceil(max(greedy['human'].values()) + .2)) if greedy else 3.5
    # Initial scores remain the reference and in the raw table, not another plotted series.
    for mi, method in enumerate(METHODS[1:4], 1):
        offset = (mi-2)*.20
        for ti, task in enumerate(TASKS):
            p = next(p for p in summary['human'] if p['task'] == task and p['method'] == method
                     and p['metric'] == 'final_return')
            ys = p['normalized']
            xs = ti+offset+np.linspace(-.035, .035, len(ys))
            ax.scatter(xs, ys, color=COLORS[mi], alpha=.4, s=16)
            for x, y in zip(xs, ys):
                if y > ceiling:
                    ax.scatter(x, ceiling-.10, marker='^', color='#b22222', s=42, zorder=5)
                    ax.annotate(f'{y:.2f} (off-scale)', xy=(x, ceiling-.10), xytext=(10, -6),
                                textcoords='offset points', color='#b22222', fontsize=10,
                                va='top')
            ax.scatter(ti+offset, st.mean(ys), color=COLORS[mi], s=65, marker='D',
                       edgecolors='white', linewidths=.6)
            add_greedy(ax, 'human', task, method, ti+offset+.055, COLORS[mi])
            if summary['human_early']:
                early = next(p for p in summary['human_early'] if p['task'] == task
                             and p['method'] == method and p['metric'] == 'final_return')
                ax.scatter(ti+offset-.065, st.mean(early['normalized']), marker='x',
                           color=COLORS[mi], s=60, linewidths=1.8, zorder=4)
    ax.axhline(0, color='#777777', ls=':', lw=1)
    ax.axhline(1, color=COLORS[3], ls='--', lw=1)
    ax.set(xticks=range(6), xticklabels=TITLES, ylabel='Return, rescaled within each game',
           ylim=(-1.4, ceiling), yticks=list(range(-1, math.floor(ceiling)+1)))
    ax.grid(axis='y', alpha=.18)
    handles = [Line2D([], [], color=COLORS[mi], lw=2, label=LABELS[mi]) for mi in range(1, 4)]
    handles.append(Line2D([], [], color='#555555', marker='D', ls='', label='4M mean'))
    if summary['human_early']:
        handles.append(Line2D([], [], color='#555555', marker='x', ls='', label='500k mean'))
    if greedy:
        greedy_legend(handles)
        ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5,-.10),
                  ncol=3, frameon=False, fontsize=10)
    else:
        ax.legend(handles=handles, loc='upper left', ncol=3, frameon=False, fontsize=10)
    ax.set_title('Human-designed games: 4M performance, with 500k means marked X\n'
                 'Shared 4M references: initial teacher mean = 0; DQN mean = 1'
                 if summary['human_early'] else 'Human-designed games: final performance', fontsize=12)
    save(fig, output/f'human-endpoints.{output_format}')


def build(data, output, greedy_preview=False):
    generated = json.loads((data/'generated.json').read_text())['rows']
    human = json.loads((data/'human.json').read_text())['rows']
    anchor_path = data/'anchor.json'
    anchor = json.loads(anchor_path.read_text())['rows'] if anchor_path.exists() else None
    baseline = json.loads((data/'random.json').read_text())['rows']
    summary = summarize(generated, human, anchor, baseline)
    output.mkdir(parents=True, exist_ok=True)
    plots(summary, output)
    if greedy_preview:
        preview = output/'greedy-preview'
        preview.mkdir(exist_ok=True)
        legacy = summarize(generated, human, anchor)
        plots(legacy, preview, output_format='pdf', greedy=greedy_means(legacy, generated, human))
    (output/'plot-data.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    with (output/'scores.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(('panel', 'task', 'metric', 'method', 'seed', 'raw_score', 'display_score'))
        for panel in ('generated', 'human'):
            for p in summary[panel]:
                for seed, raw, norm in zip(p['seeds'], p['raw'], p.get('normalized',p['raw'])):
                    writer.writerow((panel, p['task'], p['metric'], p['method'], seed, raw, norm))
    names = ['generated.json', 'human.json','random.json'] + (['anchor.json'] if anchor is not None else [])
    (output/'provenance.json').write_text(json.dumps({name: hashlib.sha256((data/name).read_bytes()).hexdigest()
                                                    for name in names}, indent=2)+'\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path(__file__).resolve().parent/'data')
    parser.add_argument('--output', type=Path, default=Path('results'))
    parser.add_argument('--greedy-preview', action='store_true',
                        help='also draw separate PDFs with greedy 4M learned-policy means')
    args = parser.parse_args()
    result = build(args.data, args.output, greedy_preview=args.greedy_preview)
    print(f"Rebuilt two figures and all per-seed scores in {args.output}; no learning agents ran.")
