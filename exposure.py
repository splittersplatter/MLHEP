"""Append-only exposure records; absence of history never means unused."""
from datetime import datetime, timezone
from pathlib import Path
from artifacts import writer_lock
from provenance import atomic_json, identity, read_json


def load_ledger(path, dataset_sha256):
    if not Path(path).exists():
        return dict(version=1, dataset_sha256=dataset_sha256, history='unknown', records=[])
    ledger = read_json(path)
    if ledger['dataset_sha256'] != dataset_sha256 or ledger['version'] != 1:
        raise ValueError('Exposure ledger belongs to another dataset')
    previous = None
    for record in ledger['records']:
        body = {k: v for k, v in record.items() if k != 'record_id'}
        if body['previous'] != previous or identity(body) != record['record_id']:
            raise ValueError('Exposure ledger chain changed')
        previous = record['record_id']
    return ledger


def record_access(path, dataset_sha256, rows, purpose, run_id, require_unused=False):
    path = Path(path)
    with writer_lock(path.parent / (path.name + '.lock')):
        ledger = load_ledger(path, dataset_sha256)
        if require_unused:
            eligible(ledger, rows)
        body = dict(previous=ledger['records'][-1]['record_id'] if ledger['records'] else None,
                    at=datetime.now(timezone.utc).isoformat(), rows=sorted(map(int, rows)),
                    purpose=purpose, run_id=run_id)
        ledger['records'].append(dict(body, record_id=identity(body)))
        atomic_json(path, ledger)


def eligible(ledger, rows):
    # Independent evaluation needs a source-backed history covering these exact
    # rows, established outside this program. No CLI switch blesses unknown use.
    rows = set(map(int, rows))
    evidence = ledger.get('unused_evidence', {})
    if (ledger['history'] != 'source_audited' or not evidence.get('reference')
            or not rows <= set(evidence.get('rows', []))):
        raise ValueError('Independent evaluation refused: unknown or unsupported unused-sample history')
    for record in ledger['records']:
        if rows & set(record['rows']):
            raise ValueError('Independent evaluation refused: sample previously accessed or reserved')
