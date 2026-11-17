"""Predeclared train-only transform variants; no final-performance selection."""
import numpy as np

BASELINE = 'log_floor_1e-12'
VARIANTS = (BASELINE, 'log_floor_1e-10', 'robust_floor_1e-12')


def specification(name):
    if name not in VARIANTS:
        raise ValueError(f'Unknown transform protocol: {name}')
    return dict(name=name, floor=1e-10 if name == 'log_floor_1e-10' else 1e-12,
                scaling='median_iqr' if name == 'robust_floor_1e-12' else 'mean_std',
                mass_transform='log1p', active_min_scale=1e-8)


def raw(data, include_mass, name=BASELINE):
    rule = specification(name)
    values = np.log(np.maximum(data['x'], rule['floor']))
    if include_mass:
        values = np.column_stack((values, np.log1p(np.maximum(data['jet_mass'], 0))))
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite raw transformed inputs')
    return values


def fit(data, train, include_mass, name=BASELINE):
    rule = specification(name)
    values = raw(data, include_mass, name)
    if len(train) == 0:
        raise ValueError('No preprocessing training rows')
    training = values[train]
    if rule['scaling'] == 'median_iqr':
        mean = np.median(training, axis=0)
        scale = (np.quantile(training, .75, axis=0, method='linear') -
                 np.quantile(training, .25, axis=0, method='linear')) / 1.3489795003921634
    else:
        mean, scale = training.mean(0), training.std(0)
    active = scale > rule['active_min_scale']
    if not active.any():
        raise ValueError('No active features')
    standardized = ((values[:, active]-mean[active])/scale[active]).astype('float32')
    if not np.isfinite(standardized).all():
        raise ValueError('Nonfinite standardized inputs')
    floor = data['x'][train] <= rule['floor']
    active_efp = active[:data['x'].shape[1]]
    clipped = standardized[train, :][:, :int(active_efp.sum())][floor[:, active_efp]]
    floor_events = np.flatnonzero(floor.any(1))
    outliers = []
    for local in floor_events:
        i = train[local]
        item = dict(row=int(data['row'][i]), floored_features=np.flatnonzero(floor[local]).tolist(),
                    mass=float(data['mass'][i]))
        # Missing legacy kinematics remain unavailable. Production schema-v3 has them.
        for field in ('constituents','jet_pt','jet_eta','jet_rapidity','delta_eta','delta_rapidity'):
            item[field] = np.asarray(data[field][i]).tolist() if field in data else None
        outliers.append(item)
    diagnostics = dict(training_events=len(train), efp_floor_counts=floor.sum(0).tolist(),
                       efp_floor_rates=floor.mean(0).tolist(), constant_columns=np.flatnonzero(training.std(0) <= 1e-8).tolist(),
                       removed_columns=np.flatnonzero(~active).tolist(),
                       removal_rule='training scale <=1e-8; robust IQR may be zero for a nonconstant feature',
                       active_feature_min=standardized[train].min(0).tolist(),
                       active_feature_max=standardized[train].max(0).tolist(),
                       clipped_standardized_range=[float(clipped.min()),float(clipped.max())] if clipped.size else None,
                       floor_events=outliers, transform=rule,
                       interpretation='Training rows only; no event removed; no causal performance claim')
    return standardized,dict(mean=mean,scale=scale,active=active,transform_id=np.array(name)),diagnostics


def apply(data, preprocessing, include_mass, expected_name=BASELINE):
    if 'transform_id' not in preprocessing or str(preprocessing['transform_id']) != expected_name:
        raise ValueError('Preprocessing transform protocol mismatch; do not infer a legacy transform')
    values = raw(data,include_mass,expected_name)
    active,mean,scale = (preprocessing[k] for k in ('active','mean','scale'))
    if (values.shape[1] != len(active) or active.dtype != np.dtype(bool)
            or mean.shape != scale.shape or mean.shape != active.shape
            or not np.isfinite(mean).all() or not np.isfinite(scale).all()
            or not active.any() or (scale[active] <= 0).any()):
        raise ValueError('Incompatible preprocessing')
    x = ((values[:,active]-mean[active])/scale[active]).astype('float32')
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite transformed features')
    return x
