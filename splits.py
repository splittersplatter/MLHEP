"""Five persisted roles, stratified by truth and lower SB / SR / upper SB."""
import numpy as np
from provenance import identity

ROLES = ('training', 'checkpoint', 'calibration', 'tuning', 'final')
FIXED = ('calibration', 'tuning', 'final')


def strata(data):
    mass = np.asarray(data['mass'])
    return 3 * data['truth'].astype(int) + np.where(mass < 3300, 0, np.where(mass < 3700, 1, 2))


def validate_roles(roles):
    if set(roles) != set(ROLES):
        raise ValueError('Expected all five sample roles')
    seen = set()
    for name in ROLES:
        values = np.asarray(roles[name])
        if values.ndim != 1 or not len(values) or not np.issubdtype(values.dtype, np.integer):
            raise ValueError(f'Invalid or empty {name} rows')
        rows = set(values.tolist())
        if len(rows) != len(values) or rows & seen:
            raise ValueError(f'Duplicate or overlapping sample role: {name}')
        seen.update(rows)
    return seen


def allocate(data, seeds, dataset_sha256, smoke=False):
    rows = np.asarray(data['row'])
    if len(np.unique(rows)) != len(rows) or len(set(seeds)) != len(seeds) or not seeds:
        raise ValueError('Duplicate raw row IDs or seeds')
    # No small-sample fallback. Explicit synthetic mode is recorded by caller.
    if not smoke and len(rows) < 1000:
        raise ValueError('Too few selected events for production roles; use explicit --smoke only for synthetic checks')
    labels = strata(data)
    fixed = {name: [] for name in FIXED}
    development = []
    counts = {}
    rng = np.random.default_rng(9173)
    for label in np.unique(labels):
        ids = rng.permutation(np.flatnonzero(labels == label))
        if len(ids) < 20:
            raise ValueError(f'Insufficient stratum {label}: {len(ids)} events (minimum 20)')
        # 20% calibration, 20% tuning, 20% reserved final; remaining 40% development.
        n = len(ids) // 5
        for i, name in enumerate(FIXED):
            fixed[name].extend(rows[ids[i*n:(i+1)*n]].tolist())
        development.extend(ids[3*n:].tolist())
        counts[str(int(label))] = int(len(ids))
    result = dict(version=2, dataset_sha256=dataset_sha256, mode='synthetic_smoke' if smoke else 'exploratory',
                  allocation='stratified 40% development / 20% calibration / 20% tuning / 20% reserved final',
                  stratum_counts=counts, seeds={})
    development = np.asarray(development, dtype=int)
    for seed in seeds:
        roles = {name: sorted(value) for name, value in fixed.items()}
        roles.update(training=[], checkpoint=[])
        rng = np.random.default_rng(seed)
        for label in np.unique(labels):
            ids = rng.permutation(development[labels[development] == label])
            n = max(1, len(ids) // 4)
            roles['checkpoint'].extend(rows[ids[:n]].tolist())
            roles['training'].extend(rows[ids[n:]].tolist())
        result['seeds'][str(seed)] = {k: sorted(v) for k, v in roles.items()}
    lookup = {int(row): i for i,row in enumerate(rows)}
    result['role_stratum_counts'] = {seed: {role: {str(int(label)): int(np.sum(labels[[lookup[row] for row in members]] == label))
        for label in np.unique(labels)} for role,members in roles.items()} for seed,roles in result['seeds'].items()}
    validate_manifest(result)
    return result


def validate_manifest(manifest):
    if manifest.get('version') != 2 or not manifest.get('seeds'):
        raise ValueError('Unsupported or empty split manifest')
    reference = None
    universe = None
    for roles in manifest['seeds'].values():
        current = validate_roles(roles)
        fixed = {k: roles[k] for k in FIXED}
        if reference is not None and (fixed != reference or current != universe):
            raise ValueError('Fixed sample membership changes across seeds')
        reference, universe = fixed, current


def indices(data, manifest, seed):
    validate_manifest(manifest)
    rows = data['row']
    lookup = {int(row): i for i, row in enumerate(rows)}
    roles = manifest['seeds'][str(seed)]
    if validate_roles(roles) != set(lookup):
        raise ValueError('Data and split membership differ')
    result = {name: np.array([lookup[row] for row in roles[name]], dtype=int) for name in ROLES}
    labels = strata(data)
    counts = {role: {str(int(label)): int(np.sum(labels[ids] == label)) for label in np.unique(labels)} for role,ids in result.items()}
    if counts != manifest['role_stratum_counts'][str(seed)]:
        raise ValueError('Role truth/region stratum counts changed')
    return result
