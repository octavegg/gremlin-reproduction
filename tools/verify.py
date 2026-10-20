"""Verify the allowlisted source/data and, optionally, separate numeric assets."""
import argparse
import hashlib
import json
from pathlib import Path


def verify(root, *, artifacts=False):
    root = Path(root).resolve()
    names = ['data/source-manifest.json'] + (['data/artifact-manifest.json'] if artifacts else [])
    count = 0
    for name in names:
        manifest = json.loads((root / name).read_text())
        for record in manifest['files']:
            relative = Path(record['path'])
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('unsafe manifest path')
            path = root / relative
            if path.is_symlink() or path.resolve() != path.absolute():
                raise ValueError(f'symlink forbidden: {relative}')
            if (path.stat().st_size != record['bytes'] or
                    hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']):
                raise ValueError(f'integrity failure: {relative}')
            count += 1
    return count


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', action='store_true')
    args = parser.parse_args()
    print(f"Verified {verify(Path(__file__).resolve().parents[1], artifacts=args.artifacts)} files.")
