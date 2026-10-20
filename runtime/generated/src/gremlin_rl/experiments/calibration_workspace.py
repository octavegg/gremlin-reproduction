"""One independent calibration job, with transactional resumable checkpoints.

SQLite keeps the latest two complete numerical states and all small evaluation
records in the same durable transaction. DELETE journaling avoids a growing WAL.
This module imports no numerical runtime, so authorization/deadlines precede JAX.
"""
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import time
import os
import fcntl
import shutil
from contextlib import contextmanager


@contextmanager
def locked_workspace(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink() or not root.is_dir():
        raise ValueError('workspace must be a real directory')
    fd = os.open(root/'.run.lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for directory in (root, root.parent):
            dfd = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(dfd)
            finally:
                os.close(dfd)
        yield
    finally:
        os.close(fd)


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(encoded(value).encode()).hexdigest()


class CalibrationWorkspace:
    def __init__(self, root, job, max_bytes, *, read_only=False):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        if self.root.is_symlink() or not self.root.is_dir():
            raise ValueError('workspace must be a real directory')
        self.max_bytes = max_bytes
        allowed = {'calibration.sqlite', 'calibration.sqlite-journal', '.run.lock'}
        if any(p.name not in allowed or p.is_symlink() or not p.is_file()
               for p in self.root.iterdir()):
            raise ValueError('workspace contains unexpected files or symlinks')
        database = self.root/'calibration.sqlite'
        if read_only:
            if not database.is_file():
                raise ValueError('inspection requires an existing database')
            # SQLite may need to roll back a hot journal after process death.
            # Permit that recovery, then prohibit application-level writes.
            self.db = sqlite3.connect(database, timeout=0)
            stored = self.db.execute('SELECT job FROM identity WHERE id=1').fetchone()
            self.db.execute('PRAGMA query_only=ON')
            if stored is None or stored[0] != encoded(job):
                self.db.close()
                raise ValueError('workspace belongs to another scientific job')
            return
        self.db = sqlite3.connect(database, timeout=0)
        self.db.execute('PRAGMA journal_mode=DELETE')
        self.db.execute('PRAGMA synchronous=EXTRA')
        self.db.executescript('''
          CREATE TABLE IF NOT EXISTS identity (id INTEGER PRIMARY KEY CHECK(id=1), job TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS segments (id TEXT PRIMARY KEY, token TEXT, deadline REAL, result TEXT);
          CREATE TABLE IF NOT EXISTS states (step INTEGER PRIMARY KEY, updates INTEGER, payload BLOB, sha256 TEXT);
          CREATE TABLE IF NOT EXISTS points (step INTEGER PRIMARY KEY, updates INTEGER, record TEXT);
        ''')
        stored = self.db.execute('SELECT job FROM identity WHERE id=1').fetchone()
        if stored and stored[0] != encoded(job):
            self.db.close()
            raise ValueError('workspace belongs to another scientific job')
        with self.db:
            self.db.execute('INSERT OR IGNORE INTO identity VALUES (1,?)', (encoded(job),))
        try:
            self.reserve(0)
        except Exception:
            self.db.close()
            raise

    def close(self):
        self.db.close()

    def reserve(self, payload_bytes):
        used = sum(p.stat().st_size for p in self.root.iterdir() if p.is_file())
        # Conservative space for full rollback journal and new/previous states.
        if 2*used + 4*payload_bytes + 2**20 > self.max_bytes:
            raise OSError('calibration workspace lacks checkpoint/journal headroom')
        if used + 4*payload_bytes + 2**20 > shutil.disk_usage(self.root).free:
            raise OSError('filesystem lacks checkpoint/journal headroom')

    def begin_segment(self, segment, now=None, started_unix=None):
        now = time.time() if now is None else now
        token = digest(segment)
        prior = self.db.execute('SELECT token,deadline,result FROM segments WHERE id=?',
                                (segment['segment_id'],)).fetchone()
        if prior:
            if prior[0] != token:
                raise ValueError('segment ID reused with different authorization')
            return prior[1], json.loads(prior[2]) if prior[2] else None
        started_unix = now if started_unix is None else started_unix
        deadline = min(segment['expires_unix'], started_unix+segment['seconds'])
        if not math.isfinite(deadline) or deadline <= now:
            raise TimeoutError('calibration segment expired')
        with self.db:
            self.db.execute('INSERT INTO segments VALUES (?,?,?,NULL)',
                            (segment['segment_id'], token, deadline))
        return deadline, None

    def latest(self):
        row = self.db.execute('SELECT step,updates,payload,sha256 FROM states ORDER BY step DESC LIMIT 1').fetchone()
        if row and hashlib.sha256(row[2]).hexdigest() != row[3]:
            raise ValueError('calibration checkpoint hash mismatch')
        return row[:3] if row else None

    def save(self, step, updates, payload):
        if type(step) is not int or type(updates) is not int or min(step, updates) < 0:
            raise ValueError('checkpoint counters must be nonnegative integers')
        sha = hashlib.sha256(payload).hexdigest()
        prior = self.db.execute('SELECT updates,sha256 FROM states WHERE step=?', (step,)).fetchone()
        if prior:
            if prior != (updates, sha):
                raise ValueError('conflicting checkpoint at the same interaction count')
            return
        latest = self.latest()
        if latest and (step <= latest[0] or updates < latest[1]):
            raise ValueError('checkpoint counters moved backwards')
        self.reserve(len(payload))
        with self.db:
            self.db.execute('INSERT INTO states VALUES (?,?,?,?)', (step,updates,payload,sha))
            self.db.execute('DELETE FROM states WHERE step NOT IN (SELECT step FROM states ORDER BY step DESC LIMIT 2)')

    def record_point(self, step, record):
        latest = self.latest()
        if latest is None or latest[0] != step:
            raise ValueError('evaluation must refer to the latest committed checkpoint')
        data = encoded(record)
        old = self.db.execute('SELECT record FROM points WHERE step=?', (step,)).fetchone()
        if old and old[0] != data:
            raise ValueError('conflicting evaluation retry')
        self.reserve(len(data))
        with self.db:
            self.db.execute('INSERT OR IGNORE INTO points VALUES (?,?,?)', (step,latest[1],data))

    def points(self):
        return [json.loads(row[0]) for row in self.db.execute('SELECT record FROM points ORDER BY step')]

    def learner_updates(self):
        return [{'interactions': step, 'updates': updates}
                for step, updates in self.db.execute('SELECT step,updates FROM points ORDER BY step')]

    def finish(self, segment_id, result):
        data = encoded(result)
        self.reserve(len(data))
        row = self.db.execute('SELECT result FROM segments WHERE id=?', (segment_id,)).fetchone()
        if row is None or (row[0] is not None and row[0] != data):
            raise ValueError('unknown segment or conflicting result')
        with self.db:
            self.db.execute('UPDATE segments SET result=? WHERE id=?', (data,segment_id))
