"""Content identities and strict, atomic JSON for protocol v2."""
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE_FILES = ('phase1.py', 'study.py', 'splits.py', 'metrics.py', 'provenance.py',
                'artifacts.py', 'exposure.py', 'confirm_disco.py', 'tune_disco.py',
                'report_study.py', 'sculpting_diagnostics.py', 'local_closure.py', 'bootstrap.py', 'audit_predictions.py', 'protocol.py', 'preprocessing.py', 'inventory_history.py', 'compare_development.py', 'feature_tail_audit.py')


def read_json(path):
    def reject(value):
        raise ValueError(f'Nonstandard JSON value: {value}')
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'Duplicate JSON key: {key}')
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), parse_constant=reject, object_pairs_hook=unique)


def encoded(value):
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n'


def identity(value):
    return hashlib.sha256(encoded(value).encode()).hexdigest()


def stable_stat(path):
    stat = Path(path).stat()
    return stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns,stat.st_ino


def file_hash(path):
    path = Path(path)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ctime_ns, before.st_ino) != (
            after.st_size, after.st_mtime_ns, after.st_ctime_ns, after.st_ino):
        raise ValueError(f'Input changed during hashing: {path}')
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = encoded(value)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix='.' + path.name)
    try:
        with os.fdopen(fd, 'w') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def source_identity():
    return {name: file_hash(ROOT / name) for name in SOURCE_FILES}


def environment(device='cpu'):
    import torch
    packages = {}
    for name in ('numpy', 'torch', 'scikit-learn', 'scipy', 'h5py', 'hdf5plugin',
                 'fastjet', 'energyflow', 'matplotlib', 'joblib'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = 'unavailable'
    return dict(python=platform.python_version(), platform=platform.platform(), packages=packages,
                device=str(device), cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(0) if str(device).startswith('cuda') else None,
                threads=torch.get_num_threads(), deterministic=torch.are_deterministic_algorithms_enabled(),
                cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                cudnn_deterministic=torch.backends.cudnn.deterministic, cudnn_benchmark=torch.backends.cudnn.benchmark)


def git_state():
    def run(*args):
        result = subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else 'unavailable'
    return dict(commit=run('rev-parse', 'HEAD'), status=run('status', '--porcelain'))


def verify_sources(directory, expected):
    for name, digest in expected.items():
        path = Path(directory) / name
        if not path.is_file() or file_hash(path) != digest:
            raise ValueError(f'Source snapshot mismatch: {name}')
