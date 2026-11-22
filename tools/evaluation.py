"""Saved-block execution using frozen numerical kernels; no SSH."""
import time

def run_extension(runtime,db,targets,*,deadline,save_guard,resource_check,stopping,budget,on_save=lambda counters:None,evaluation_offset=0,execution_grid=False,evaluation_indices=None):
    """Complete atomic numerical blocks; save before assessment and on a soft stop."""
    if (not targets or targets!=sorted(set(targets)) or
            any(type(n) is not int or n<=(0 if execution_grid else runtime.job['targets'][-1]) for n in targets)):
        raise ValueError('extension targets must strictly follow the original budget')
    if evaluation_indices is not None and (set(map(str,targets))-set(evaluation_indices) or
            any(type(v) is not int or v<0 for v in evaluation_indices.values()) or
            len(set(evaluation_indices.values()))!=len(evaluation_indices)):
        raise ValueError('invalid assessment-stream indices')
    points=[];saved=None;last_save=time.monotonic()
    def save():
        nonlocal saved,last_save
        counters=runtime.counters()
        if counters!=saved:
            payload=runtime.encode()
            with save_guard(len(payload)): db.save(*counters,payload)
            saved=counters
            on_save(counters)
        last_save=time.monotonic()
    save()
    for index,target in enumerate(targets,start=len(runtime.job['targets'])+evaluation_offset):
        if evaluation_indices is not None: index=evaluation_indices[str(target)]
        old=[p for p in db.points() if p['target']==target]
        if old: points.extend(old);continue
        while runtime.counters()[0]<target:
            if stopping() or not resource_check() or not runtime.advance_before_deadline(budget):
                save();return dict(status='paused',interactions=runtime.counters()[0],raw_points=db.points())
            if time.monotonic()-last_save>=120: save()
        save()
        def check():
            if stopping() or time.monotonic()>=deadline-60 or not resource_check():
                raise TimeoutError('save complete; assessment deferred')
        episodes=runtime.job['final_episodes'] if target==targets[-1] else runtime.job['evaluation_episodes']
        try:
            primary=runtime.evaluate(episodes,index,check=check)
            greedy=[] if runtime.is_dqn else runtime.evaluate(episodes,index,greedy=True,check=check)
        except TimeoutError:
            return dict(status='paused',interactions=runtime.counters()[0],raw_points=db.points())
        row=dict(target=target,interactions=runtime.counters()[0],updates=runtime.counters()[1],
                 primary=primary,greedy=greedy,
                 discarded_interactions=0 if runtime.is_dqn else runtime.collector.discarded_transitions)
        with save_guard(2**20): db.record_point(runtime.counters()[0],row)
        points.append(row)
        print(json.dumps(dict(event='evaluation',target=target,interactions=row['interactions'],
            mean_return=sum(x['return_raw'] for x in primary)/len(primary))),flush=True)
    return dict(status='complete',interactions=runtime.counters()[0],raw_points=points)
