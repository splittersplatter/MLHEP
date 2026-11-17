"""Single-writer runs and checksum-verified completion markers."""
from contextlib import contextmanager
import os
from pathlib import Path
from provenance import atomic_json, file_hash, identity, read_json


@contextmanager
def writer_lock(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / '.writer.lock'
    try:
        fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise ValueError(f'Writer lock exists: {lock}; inspect the recorded PID before manual recovery') from error
    try:
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        yield
    finally:
        lock.unlink()


def complete(directory, run_id, metadata):
    directory = Path(directory)
    files = {str(p.relative_to(directory)): file_hash(p)
             for p in sorted(directory.rglob('*')) if p.is_file() and p.name != 'complete.json'}
    if not files:
        raise ValueError('Cannot complete empty artifact')
    atomic_json(directory / 'complete.json', dict(version=2, run_id=run_id, metadata=metadata, files=files))


def verify(directory, run_id=None):
    directory = Path(directory)
    marker = read_json(directory / 'complete.json')
    if marker['version'] != 2 or (run_id is not None and marker['run_id'] != run_id):
        raise ValueError(f'Artifact run identity mismatch: {directory}')
    actual = {str(p.relative_to(directory)) for p in directory.rglob('*')
              if p.is_file() and p.name != 'complete.json'}
    if not marker['files'] or actual != set(marker['files']):
        raise ValueError(f'Artifact membership mismatch: {directory}')
    for name, expected in marker['files'].items():
        path = directory / name
        if Path(name).is_absolute() or '..' in Path(name).parts or path.is_symlink() or file_hash(path) != expected:
            raise ValueError(f'Artifact checksum mismatch: {path}')
    return marker


def verify_run(run):
    run = Path(run)
    protocol = read_json(run / 'protocol.json')
    if not protocol['identity']['config']['smoke'] and protocol.get('ledger_path') != protocol['identity'].get('ledger_path'):
        raise ValueError('Canonical dataset exposure ledger changed')
    if identity(protocol['identity']) != protocol['run_id']:
        raise ValueError('Invalid run protocol identity')
    from provenance import verify_sources
    verify_sources(run / 'source', protocol['identity']['source'])
    from splits import validate_manifest
    splits = read_json(run / 'splits.json')
    if identity(splits) != protocol['identity']['split_sha256']:
        raise ValueError('Split manifest changed')
    validate_manifest(splits)
    marker = verify(run / 'results', protocol['run_id'])
    expected = protocol['identity']['expected_fits']
    if marker['metadata']['fits'] != expected:
        raise ValueError('Incomplete or mixed fit grid')
    for fit in expected:
        verify(run / 'fits' / fit, protocol['run_id'])
    return protocol
