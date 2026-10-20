"""Process-local hard expiry for transfer workers; no desktop heartbeat needed.

Import and arm before numerical libraries. SIGALRM uses its OS default action,
not a Python callback, so native computation cannot defer termination. Workers
must not launch subprocesses: a process-local timer does not supervise children.
"""
from datetime import datetime, time as local_time, timedelta
import math
import signal
import time
from zoneinfo import ZoneInfo


PARIS = ZoneInfo('Europe/Paris')


class BoundaryBudget:
    """Stop starting work early; finish accepted blocks and checkpoint normally.

    Times use the monotonic deadline returned by ``arm_hard_deadline``. The
    one-minute reserve lies INSIDE that deadline, never extends it. Admission
    reserves twice the slowest observed block (or the supplied cold estimate)
    plus that minute for saving. It cannot guarantee a slow/hung block finishes:
    the independently armed OS timer remains the final protection.

    A block is the runtime's atomic learner update, not an episode. Never use
    this gate inside a trajectory or halfway through an update. The caller
    checkpoints after completed blocks, including when this returns False.
    """

    def __init__(self, deadline, *, estimated_block_seconds, clock=time.monotonic):
        for value in (deadline, estimated_block_seconds):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError('finite boundary timing required')
        if estimated_block_seconds <= 0:
            raise ValueError('positive cold block estimate required')
        self.deadline = deadline
        self.slowest_block_seconds = estimated_block_seconds
        self.clock = clock

    def run(self, operation):
        before = self.clock()
        if before + 2 * self.slowest_block_seconds + 60 >= self.deadline:
            return False
        operation()
        self.slowest_block_seconds = max(self.slowest_block_seconds, self.clock() - before)
        return True


def transfer_deadline(*, started_unix, expires_unix, planned_seconds, remote):
    if (type(remote) is not bool or type(planned_seconds) is not int or planned_seconds <= 0
            or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
                   for x in (started_unix, expires_unix))):
        raise ValueError('invalid explicit execution window')
    start = datetime.fromtimestamp(started_unix, PARIS)
    if remote and start.weekday() < 5 and 8 <= start.hour < 18:
        raise ValueError('weekday SSH computation is forbidden from 08:00 to 18:00 Paris')
    morning = datetime.combine(start.date(), local_time(7, 40), PARIS)
    if start >= morning:
        # No fresh remote work in the final shutdown interval either.
        if remote and start.hour == 7:
            raise ValueError('morning shutdown interval has begun')
        morning = datetime.combine(start.date() + timedelta(days=1), local_time(7, 40), PARIS)
    if expires_unix > morning.timestamp():
        raise ValueError('transfer expiry exceeds the next 07:40 Paris hard stop')
    deadline = min(expires_unix, started_unix + planned_seconds + 300)
    if deadline <= started_unix:
        raise ValueError('execution window has already expired')
    return deadline


def arm_hard_deadline(deadline):
    if isinstance(deadline, bool) or not isinstance(deadline, (int, float)) or not math.isfinite(deadline):
        raise ValueError('finite process expiry required')
    remaining = deadline - time.time()
    if remaining <= 0:
        raise TimeoutError('worker expiry already reached')
    existing, _ = signal.getitimer(signal.ITIMER_REAL)
    if existing > 0:
        remaining = min(remaining, existing)
    signal.signal(signal.SIGALRM, signal.SIG_DFL)
    signal.setitimer(signal.ITIMER_REAL, max(.001, remaining))
    return time.monotonic() + remaining
