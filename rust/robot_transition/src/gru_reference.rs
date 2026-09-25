//! GRU32 residual reference, using the frozen public-history equations.
//!
//! Row-major parameters: base[6,25], W_ih[96,24], W_hh[96,32],
//! b_ih[96], b_hh[96], head_W[6,32], head_b[6]. Total 5,916 f32s.
//! State: q[6], previous_q[6], previous_u[6], hidden[32]. Context conditioning
//! assimilates t=1..C-2 only; the first forecast input is u[C-1].
//! All arithmetic is ordinary f32, with no fast-math, fused mul_add, cache,
//! allocation, external BLAS, parameter preparation or hidden input access.
//! Finite-reduction differences from Torch require separate qualification.

use super::{dot, extent, finite, overlap, sigmoid, NONFINITE, OK, POINTER, SHAPE};
use std::slice;

const PARAMETERS: usize = 5916;
const STATE: usize = 50;
const WI: usize = 150;
const WH: usize = WI + 96 * 24;
const BI: usize = WH + 96 * 32;
const BH: usize = BI + 96;
const HEAD: usize = BH + 96;
const HEAD_B: usize = HEAD + 6 * 32;

#[no_mangle]
pub extern "C" fn gr_abi_version() -> u32 { 1 }

#[no_mangle]
pub extern "C" fn gr_parameter_count() -> usize { PARAMETERS }

fn hidden_step(p: &[f32], x: &[f32; 24], h: &[f32]) -> Result<[f32; 32], i32> {
    let mut input_affine = [0.0f32; 96];
    let mut hidden_affine = [0.0f32; 96];
    for i in 0..96 {
        input_affine[i] = dot(&p[WI + i * 24..WI + (i + 1) * 24], x) + p[BI + i];
        hidden_affine[i] = dot(&p[WH + i * 32..WH + (i + 1) * 32], h) + p[BH + i];
    }
    // Finite inputs can overflow an affine to signed infinity. Torch permits
    // sigmoid/tanh to saturate there. Reject NaNs here and require finite states
    // after the activation, rather than excluding that valid numerical path.
    if input_affine.iter().chain(hidden_affine.iter()).any(|value| value.is_nan()) {
        return Err(NONFINITE);
    }
    let mut next = [0.0f32; 32];
    for i in 0..32 {
        let reset_arg = input_affine[i] + hidden_affine[i];
        let update_arg = input_affine[32 + i] + hidden_affine[32 + i];
        let candidate_arg = input_affine[64 + i] + sigmoid(reset_arg) * hidden_affine[64 + i];
        if reset_arg.is_nan() || update_arg.is_nan() || candidate_arg.is_nan() {
            return Err(NONFINITE);
        }
        let candidate = candidate_arg.tanh();
        // Torch GRUCell reset applies to the complete hidden candidate affine.
        next[i] = candidate + sigmoid(update_arg) * (h[i] - candidate);
    }
    if !finite(&next) { return Err(NONFINITE); }
    Ok(next)
}

fn condition(p: &[f32], q: &[f32], u: &[f32], context: usize) -> Result<[f32; STATE], i32> {
    let mut hidden = [0.0f32; 32];
    for t in 1..context - 1 {
        let mut x = [0.0f32; 24];
        x[..6].copy_from_slice(&q[t * 6..(t + 1) * 6]);
        x[6..12].copy_from_slice(&q[(t - 1) * 6..t * 6]);
        x[12..18].copy_from_slice(&u[t * 6..(t + 1) * 6]);
        x[18..].copy_from_slice(&u[(t - 1) * 6..t * 6]);
        hidden = hidden_step(p, &x, &hidden)?;
    }
    let mut state = [0.0f32; STATE];
    state[..6].copy_from_slice(&q[(context - 1) * 6..context * 6]);
    state[6..12].copy_from_slice(&q[(context - 2) * 6..(context - 1) * 6]);
    state[12..18].copy_from_slice(&u[(context - 2) * 6..(context - 1) * 6]);
    state[18..].copy_from_slice(&hidden);
    Ok(state)
}

fn step(p: &[f32], state: &[f32; STATE], u: &[f32]) -> Result<[f32; STATE], i32> {
    let mut x = [0.0f32; 24];
    x[..12].copy_from_slice(&state[..12]);
    x[12..18].copy_from_slice(u);
    x[18..].copy_from_slice(&state[12..18]);
    let hidden = hidden_step(p, &x, &state[18..])?;
    let mut next = [0.0f32; STATE];
    for i in 0..6 {
        let base = dot(&p[i * 25..i * 25 + 24], &x) + p[i * 25 + 24];
        let residual = dot(&p[HEAD + i * 32..HEAD + (i + 1) * 32], &hidden) + p[HEAD_B + i];
        next[i] = base + residual;
    }
    next[6..12].copy_from_slice(&state[..6]);
    next[12..18].copy_from_slice(u);
    next[18..].copy_from_slice(&hidden);
    if !finite(&next) { return Err(NONFINITE); }
    Ok(next)
}

fn pointer_check(raw: &[(usize, usize)], outputs_begin: usize) -> Result<(), i32> {
    // At most six input/output ranges; no heap allocation is retained or used.
    let mut ranges = [(0usize, 0usize); 6];
    for (i, &(address, count)) in raw.iter().enumerate() {
        ranges[i] = extent(address, count).ok_or(POINTER)?;
    }
    for output in outputs_begin..raw.len() {
        for earlier in 0..output {
            if overlap(ranges[output], ranges[earlier]) { return Err(POINTER); }
        }
    }
    Ok(())
}

/// Full context condition plus free rollout in standardized float32 coordinates.
///
/// # Safety
/// Nonempty pointers must designate valid allocations of the declared lengths;
/// callers must not mutate inputs concurrently. Outputs must not overlap any
/// input or one another. Shape, alignment, finite values and detectable overlap
/// are checked; allocation provenance cannot be established from raw pointers.
/// On any nonzero return, discard both output buffers, including partial writes.
#[no_mangle]
pub unsafe extern "C" fn gr_request_v1(
    batch: usize, context: usize, horizon: usize,
    parameters: *const f32, parameter_len: usize,
    q_context: *const f32, q_len: usize,
    u_context: *const f32, u_len: usize,
    future: *const f32, future_len: usize,
    prediction: *mut f32, prediction_len: usize,
    final_state: *mut f32, final_len: usize,
) -> i32 {
    let history = batch.checked_mul(context).and_then(|x| x.checked_mul(6));
    let forecast = batch.checked_mul(horizon).and_then(|x| x.checked_mul(6));
    if batch == 0 || context < 2 || parameter_len != PARAMETERS
        || history != Some(q_len) || history != Some(u_len)
        || forecast != Some(future_len) || forecast != Some(prediction_len)
        || batch.checked_mul(STATE) != Some(final_len) { return SHAPE; }
    let raw = [(parameters as usize, parameter_len), (q_context as usize, q_len),
        (u_context as usize, u_len), (future as usize, future_len),
        (prediction as usize, prediction_len), (final_state as usize, final_len)];
    if let Err(code) = pointer_check(&raw, 4) { return code; }
    let p = slice::from_raw_parts(parameters, parameter_len);
    let q = slice::from_raw_parts(q_context, q_len);
    let u = slice::from_raw_parts(u_context, u_len);
    let future_values = if future_len == 0 { &[] } else { slice::from_raw_parts(future, future_len) };
    if !finite(p) || !finite(q) || !finite(u) || !finite(future_values) { return NONFINITE; }
    let output = if prediction_len == 0 { &mut [] } else { slice::from_raw_parts_mut(prediction, prediction_len) };
    let final_values = slice::from_raw_parts_mut(final_state, final_len);
    for b in 0..batch {
        let begin = b * context * 6;
        let end = begin + context * 6;
        let mut current = match condition(p, &q[begin..end], &u[begin..end], context) {
            Ok(value) => value, Err(code) => return code,
        };
        for t in 0..horizon {
            let offset = (b * horizon + t) * 6;
            current = match step(p, &current, &future_values[offset..offset + 6]) {
                Ok(value) => value, Err(code) => return code,
            };
            output[offset..offset + 6].copy_from_slice(&current[..6]);
        }
        final_values[b * STATE..(b + 1) * STATE].copy_from_slice(&current);
    }
    OK
}

/// Continue a free rollout from an explicit standardized [B,50] state.
///
/// # Safety
/// Same pointer and output-discard contract as `gr_request_v1`.
#[no_mangle]
pub unsafe extern "C" fn gr_rollout_v1(
    batch: usize, horizon: usize,
    parameters: *const f32, parameter_len: usize,
    future: *const f32, future_len: usize,
    initial: *const f32, initial_len: usize,
    prediction: *mut f32, prediction_len: usize,
    final_state: *mut f32, final_len: usize,
) -> i32 {
    let forecast = batch.checked_mul(horizon).and_then(|x| x.checked_mul(6));
    let states = batch.checked_mul(STATE);
    if batch == 0 || parameter_len != PARAMETERS || forecast != Some(future_len)
        || forecast != Some(prediction_len) || states != Some(initial_len)
        || states != Some(final_len) { return SHAPE; }
    let raw = [(parameters as usize, parameter_len), (future as usize, future_len),
        (initial as usize, initial_len), (prediction as usize, prediction_len),
        (final_state as usize, final_len)];
    if let Err(code) = pointer_check(&raw, 3) { return code; }
    let p = slice::from_raw_parts(parameters, parameter_len);
    let u = if future_len == 0 { &[] } else { slice::from_raw_parts(future, future_len) };
    let states = slice::from_raw_parts(initial, initial_len);
    if !finite(p) || !finite(u) || !finite(states) { return NONFINITE; }
    let output = if prediction_len == 0 { &mut [] } else { slice::from_raw_parts_mut(prediction, prediction_len) };
    let final_values = slice::from_raw_parts_mut(final_state, final_len);
    for b in 0..batch {
        let mut current = [0.0f32; STATE];
        current.copy_from_slice(&states[b * STATE..(b + 1) * STATE]);
        for t in 0..horizon {
            let offset = (b * horizon + t) * 6;
            current = match step(p, &current, &u[offset..offset + 6]) {
                Ok(value) => value, Err(code) => return code,
            };
            output[offset..offset + 6].copy_from_slice(&current[..6]);
        }
        final_values[b * STATE..(b + 1) * STATE].copy_from_slice(&current);
    }
    OK
}
