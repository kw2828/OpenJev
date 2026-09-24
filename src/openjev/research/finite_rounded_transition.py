"""Fixed finite balancing followed by positive-slack transport rounding.

The convention is T[action,next,current]. Exactly four log-domain sweeps
normalize rows (dim2), then columns (dim1). Row and column contractions to
mass at most tau=1-1e-8 precede a residual rank-one addition. This adapts
Algorithm2 of Altschuler, Weed and Rigollet (2017), arXiv:1705.09634. The
positive slack and four-sweep initialization define a different function:
an already doubly stochastic X becomes tau*X+(1-tau)*uniform, in exact
arithmetic. A large pre-round residual can produce a substantial correction.

In exact arithmetic both deficits are at least1e-8, their totals agree, and
T=Z+(r/sum(r))*c^T has positive entries and both marginals equal to one.
In floating arithmetic the deficit totals can differ. Before other rounding,
the row error is r_i*(sum(c)/sum(r)-1), bounded by the absolute difference
of those totals. There is no universal floating-point success guarantee.
Finite and strict probability guards and fixed1e-12 residual guards remain
mandatory; rejection never triggers clipping, retries or additional sweeps.

Autograd differentiates the executed finite algorithm, including its maximum
branches, not an infinite Sinkhorn limit. At a maximum tie its chosen local
derivative is not a claim of differentiability. Returned logs are computed
from the FINAL corrected probabilities and preserve the same graph.

Counters describe explicit forward operations, not FLOPs or backward work.
They exclude validation, tensor copying and scalar extrema/diagnostics, but
include the named diagnostic sums. Increments precede attempted operations;
completed_sweeps advances after both finite guards. A supplied work dictionary
retains all partial counts if any guard or external callback raises.
"""
from __future__ import annotations

import torch

VERSION = 'finite-rounded-transition-v1'
SWEEPS = 4
SLACK = 1e-8
TOLERANCE = 1e-12
WORK_KEYS = (
    'rounded_transition_calls', 'row_logsumexp_calls', 'column_logsumexp_calls',
    'normalization_vectors', 'normalization_entries', 'completed_sweeps',
    'exponential_calls', 'exponential_entries', 'matrix_sum_calls', 'vector_sum_calls',
    'contraction_maximum_calls', 'contraction_division_entries', 'contraction_product_entries',
    'deficit_subtraction_entries', 'deficit_normalization_entries',
    'rank_one_product_entries', 'rank_one_addition_entries', 'correction_mass_sum_calls',
    'logarithm_calls', 'logarithm_entries', 'check_calls',
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _finite(value, name):
    _require(bool(torch.isfinite(value).all()), name + ': finite values required')


def _work(value):
    if value is None:
        value = {}
    _require(type(value) is dict and all(type(count) is int and count >= 0 for count in value.values()),
             'work must be a dictionary of nonnegative Python integer counts')
    for key in WORK_KEYS:
        value.setdefault(key, 0)
    return value


def rounded_transition(logits, *, check=lambda: None, work=None):
    """Return owned transition/log_transition tensors, diagnostics and work.

    Input is finite CPU strided float64 shape(4,8,8), without configuration
    overrides. No input tensor is mutated. Outputs own separate storage and
    retain autograd. Intermediate exponential zeros are permitted; positive
    slack is part of this function, not a fallback. Final probabilities must
    be strictly between zero and one, and final logs must be finite.

    Each successful call performs4+4 logsumexp calls,256 normalized vectors,
    2048 normalized entries,7 matrix sums,2 vector sums,1 correction-mass sum,
    2 maxima,64 contraction divisions,512 contraction products,64 deficit
    subtractions,32 deficit divisions,256 outer products and256 additions,
    256 explicit exponentials and256 final logarithms. Eleven check calls
    occur: entry, each sweep, exponential, each contraction, deficits, final
    probabilities, final logarithms. Callback exceptions propagate unchanged.
    All guards use ValueError; a supplied work dictionary survives failure.
    """
    counts = _work(work)
    counts['rounded_transition_calls'] += 1
    _require(callable(check), 'check must be callable')

    def tick():
        counts['check_calls'] += 1
        check()

    def matrix_sum(value, dim):
        counts['matrix_sum_calls'] += 1
        return value.sum(dim=dim)

    tick()
    _require(isinstance(logits, torch.Tensor) and logits.device.type == 'cpu'
             and logits.layout == torch.strided and logits.dtype == torch.float64
             and tuple(logits.shape) == (4, 8, 8),
             'logits must be CPU strided float64 with shape(4,8,8)')
    _finite(logits, 'logits')
    log_x = logits.clone()
    for _ in range(SWEEPS):
        counts['row_logsumexp_calls'] += 1
        counts['normalization_vectors'] += 32
        row_normalizer = torch.logsumexp(log_x, dim=2, keepdim=True)
        counts['normalization_entries'] += 256
        log_x = log_x - row_normalizer
        _finite(log_x, 'row-normalized logs')
        counts['column_logsumexp_calls'] += 1
        counts['normalization_vectors'] += 32
        column_normalizer = torch.logsumexp(log_x, dim=1, keepdim=True)
        counts['normalization_entries'] += 256
        log_x = log_x - column_normalizer
        _finite(log_x, 'column-normalized logs')
        counts['completed_sweeps'] += 1
        tick()

    counts['exponential_calls'] += 1
    counts['exponential_entries'] += 256
    x = log_x.exp()
    _finite(x, 'pre-round transition')
    _require(bool((x >= 0).all()), 'nonnegative pre-round transition required')
    tick()

    tau = 1 - SLACK
    pre_rows = matrix_sum(x, 2)
    pre_columns = matrix_sum(x, 1)
    _finite(pre_rows, 'pre-round row masses')
    _finite(pre_columns, 'pre-round column masses')
    counts['contraction_maximum_calls'] += 1
    row_denominator = torch.maximum(pre_rows, torch.full_like(pre_rows, tau))
    counts['contraction_division_entries'] += 32
    alpha = tau / row_denominator
    counts['contraction_product_entries'] += 256
    y = alpha[:, :, None] * x
    _finite(y, 'row-contracted transition')
    tick()

    y_columns = matrix_sum(y, 1)
    _finite(y_columns, 'row-contracted column masses')
    counts['contraction_maximum_calls'] += 1
    column_denominator = torch.maximum(y_columns, torch.full_like(y_columns, tau))
    counts['contraction_division_entries'] += 32
    beta = tau / column_denominator
    counts['contraction_product_entries'] += 256
    z = y * beta[:, None, :]
    _finite(z, 'column-contracted transition')
    tick()

    z_rows, z_columns = matrix_sum(z, 2), matrix_sum(z, 1)
    counts['deficit_subtraction_entries'] += 64
    r, c = 1 - z_rows, 1 - z_columns
    _finite(r, 'row deficits')
    _finite(c, 'column deficits')
    _require(bool((r > 0).all()) and bool((c > 0).all()), 'positive deficits required')
    counts['vector_sum_calls'] += 1
    s = r.sum(dim=1, keepdim=True)
    counts['vector_sum_calls'] += 1
    column_total = c.sum(dim=1, keepdim=True)
    _finite(s, 'row deficit totals')
    _finite(column_total, 'column deficit totals')
    _require(bool((s > 0).all()) and bool((column_total > 0).all()), 'positive deficit totals required')
    tick()

    counts['deficit_normalization_entries'] += 32
    normalized_r = r / s
    counts['rank_one_product_entries'] += 256
    correction = normalized_r[:, :, None] * c[:, None, :]
    counts['rank_one_addition_entries'] += 256
    transition = z + correction
    _finite(transition, 'final transition')
    _require(bool(((transition > 0) & (transition < 1)).all()),
             'strict final transition probabilities required')
    final_rows, final_columns = matrix_sum(transition, 2), matrix_sum(transition, 1)
    row_error = float((final_rows - 1).abs().amax().detach())
    column_error = float((final_columns - 1).abs().amax().detach())
    counts['correction_mass_sum_calls'] += 1
    correction_mass = correction.sum(dim=(1, 2))
    _finite(correction_mass, 'correction mass')
    diagnostics = {
        'sweeps': SWEEPS, 'slack': SLACK, 'tolerance': TOLERANCE,
        'pre_round_row_residual_max': float((pre_rows - 1).abs().amax().detach()),
        'pre_round_column_residual_max': float((pre_columns - 1).abs().amax().detach()),
        'row_deficit_totals': s.detach().reshape(4).tolist(),
        'column_deficit_totals': column_total.detach().reshape(4).tolist(),
        'deficit_total_discrepancies': (s - column_total).detach().reshape(4).tolist(),
        'correction_mass': correction_mass.detach().tolist(),
        'correction_maximum_entry': float(correction.amax().detach()),
        'pre_to_final_maximum_change': float((transition - x).abs().amax().detach()),
        'row_residual_max': row_error, 'column_residual_max': column_error,
        'minimum_probability': float(transition.amin().detach()),
        'maximum_probability': float(transition.amax().detach()),
        'admitted': row_error <= TOLERANCE and column_error <= TOLERANCE,
    }
    tick()
    _require(diagnostics['admitted'],
             f'rounded residual exceeded tolerance: row={row_error!r}, column={column_error!r}')
    counts['logarithm_calls'] += 1
    counts['logarithm_entries'] += 256
    log_transition = transition.log()
    _finite(log_transition, 'final log transition')
    tick()
    return {'transition': transition, 'log_transition': log_transition,
            'diagnostics': diagnostics, 'work': counts}
