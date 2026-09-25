//! Dependency-free, float32 sequential rollout for prepared structured cells.
//!
//! This is a hybrid inference kernel: the caller must execute the qualified
//! Torch parameter validation/preparation on EVERY request, including dense
//! spectral norms. No prepared parameter cache is part of this library.
//! Arithmetic uses ordinary f32 operations, never mul_add or fast-math.
//! Differences from Torch reduction order and elementary functions are expected;
//! parity is a separately qualified tolerance claim, not bitwise equality.
//!
//! ABI1 kind codes: 0 householder, 1 dense_bounded, 2 dense_unbounded, 3 dense_mlp.
//! Packed row-major fields: scale[12], input_matrix[2,12,6], bias[2,12], then
//! vectors[2,4,12],decay[2,12] (kind0) or matrix[2,12,12] (other kinds), then
//! compact gate: W[24,12],bias_ru[16],bias_in[8],bias_hn[8],head_W[2,8],head_b[2];
//! or MLP gate: W1[8,12],b1[8],W2[2,8],b2[2].
//! Status: 0 success, 1 kind/dimension/length error, 2 pointer/alignment/alias
//! error, 3 nonfinite numerical value, 4 invalid scale/reflection denominator.
//! On failure output buffers may be partially written and must be discarded.
//! Inputs are never mutated. There are no files, threads, global state or caches.

use std::mem::{align_of, size_of};
use std::slice;

mod gru_reference;

const OK: i32 = 0;
const SHAPE: i32 = 1;
const POINTER: i32 = 2;
const NONFINITE: i32 = 3;
const DOMAIN: i32 = 4;

#[no_mangle]
pub extern "C" fn rt_abi_version() -> u32 {
    1
}

#[no_mangle]
pub extern "C" fn rt_parameter_count(kind: u32) -> usize {
    match kind {
        0 => 638,
        1 | 2 => 806,
        3 => 590,
        _ => 0,
    }
}

fn extent(address: usize, count: usize) -> Option<(usize, usize)> {
    if count == 0 {
        return Some((0, 0));
    }
    let bytes = count.checked_mul(size_of::<f32>())?;
    if address == 0 || address % align_of::<f32>() != 0 || bytes > isize::MAX as usize {
        return None;
    }
    Some((address, address.checked_add(bytes)?))
}

fn overlap(a: (usize, usize), b: (usize, usize)) -> bool {
    a.0 < b.1 && b.0 < a.1
}

fn finite(values: &[f32]) -> bool {
    values.iter().all(|x| x.is_finite())
}

fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let e = value.exp();
        e / (1.0 + e)
    }
}

fn dot(left: &[f32], right: &[f32]) -> f32 {
    let mut value = 0.0f32;
    for i in 0..left.len() {
        value += left[i] * right[i];
    }
    value
}

fn reflection_dot(left: &[f32; 12], right: &[f32; 12]) -> f32 {
    // Torch 2.14 CPU SumKernel's contiguous-inner float32 reduction on the
    // qualified ARM NEON backend sums three 4-lane vectors, then the lanes.
    // Match that order for the cancellation-sensitive paired reflections.
    // This is not an architecture-independent bitwise Torch guarantee.
    let mut lanes = [0.0f32; 4];
    for block in 0..3 {
        for lane in 0..4 {
            let i = block * 4 + lane;
            lanes[lane] += left[i] * right[i];
        }
    }
    let mut result = 0.0f32;
    for value in lanes {
        result += value;
    }
    result
}

fn gate(kind: u32, weights: &[f32], x: &[f32; 12]) -> Result<[f32; 2], i32> {
    let mut hidden = [0.0f32; 8];
    let (head_offset, bias_offset);
    if kind == 3 {
        for i in 0..8 {
            let affine = dot(&weights[i * 12..(i + 1) * 12], x) + weights[96 + i];
            // Like Torch, allow signed overflow to saturate through tanh.
            // NaN is never repaired, and final gate logits must remain finite.
            if affine.is_nan() {
                return Err(NONFINITE);
            }
            hidden[i] = affine.tanh();
        }
        head_offset = 104;
        bias_offset = 120;
    } else {
        let mut affine = [0.0f32; 24];
        for i in 0..24 {
            affine[i] = dot(&weights[i * 12..(i + 1) * 12], x);
        }
        if affine.iter().any(|value| value.is_nan()) {
            return Err(NONFINITE);
        }
        for i in 0..8 {
            let reset_arg = affine[i] + weights[288 + i];
            let update_arg = affine[8 + i] + weights[296 + i];
            let candidate_arg = (affine[16 + i] + weights[304 + i])
                + sigmoid(reset_arg) * weights[312 + i];
            // Sigmoid/tanh have finite limits at signed infinity. Rejecting
            // those intermediates would narrow the qualified Torch domain.
            if reset_arg.is_nan() || update_arg.is_nan() || candidate_arg.is_nan() {
                return Err(NONFINITE);
            }
            hidden[i] = (1.0 - sigmoid(update_arg)) * candidate_arg.tanh();
        }
        head_offset = 320;
        bias_offset = 336;
    }
    let logits = [
        dot(&weights[head_offset..head_offset + 8], &hidden) + weights[bias_offset],
        dot(&weights[head_offset + 8..head_offset + 16], &hidden) + weights[bias_offset + 1],
    ];
    if !finite(&logits) {
        return Err(NONFINITE);
    }
    let maximum = logits[0].max(logits[1]);
    let a = (logits[0] - maximum).exp();
    let b = (logits[1] - maximum).exp();
    let denominator = a + b;
    let mixing = [a / denominator, b / denominator];
    if !finite(&mixing) {
        return Err(NONFINITE);
    }
    Ok(mixing)
}

fn step(kind: u32, p: &[f32], current: &[f32; 12], u: &[f32]) -> Result<[f32; 12], i32> {
    let mut features = [0.0f32; 12];
    features[..6].copy_from_slice(&current[..6]);
    features[6..].copy_from_slice(u);
    let gate_offset = if kind == 0 { 300 } else { 468 };
    let mixing = gate(kind, &p[gate_offset..], &features)?;
    let mut value = [0.0f32; 12];
    for i in 0..12 {
        value[i] = current[i] * p[i];
    }
    if !finite(&value) {
        return Err(NONFINITE);
    }
    let mut transition = [0.0f32; 12];
    if kind == 0 {
        for k in 0..4 {
            let mut vector = [0.0f32; 12];
            for i in 0..12 {
                vector[i] = mixing[0] * p[180 + k * 12 + i]
                    + mixing[1] * p[180 + 48 + k * 12 + i];
            }
            let numerator = reflection_dot(&vector, &value);
            let denominator = reflection_dot(&vector, &vector);
            if !numerator.is_finite() || !denominator.is_finite() {
                return Err(NONFINITE);
            }
            if denominator <= 0.0 {
                return Err(DOMAIN);
            }
            for i in 0..12 {
                // Match Torch's elementwise expression association.
                value[i] -= ((2.0 * vector[i]) * numerator) / denominator;
            }
            if !finite(&value) {
                return Err(NONFINITE);
            }
        }
        for i in 0..12 {
            let decay = mixing[0] * p[276 + i] + mixing[1] * p[288 + i];
            transition[i] = decay * value[i];
        }
    } else {
        for i in 0..12 {
            let a = dot(&p[180 + i * 12..180 + (i + 1) * 12], &value);
            let b = dot(&p[324 + i * 12..324 + (i + 1) * 12], &value);
            transition[i] = mixing[0] * a + mixing[1] * b;
        }
    }
    let mut next = [0.0f32; 12];
    for i in 0..12 {
        let a = dot(&p[12 + i * 6..12 + (i + 1) * 6], u) + p[156 + i];
        let b = dot(&p[84 + i * 6..84 + (i + 1) * 6], u) + p[168 + i];
        let forced = mixing[0] * a + mixing[1] * b;
        next[i] = (transition[i] + forced) / p[i];
    }
    if !finite(&next) {
        return Err(NONFINITE);
    }
    Ok(next)
}

/// Roll out row-major [B,H,6] future torques from explicit [B,12] states.
///
/// # Safety
/// Nonempty pointers must designate readable/writable allocations of the given
/// lengths for this call. Caller retains input ownership and cannot mutate them
/// concurrently. Output allocations must not overlap any input or each other.
/// Null, alignment, length, finite-value and detectable overlap checks are made;
/// allocation provenance cannot be proven from a C pointer. The Python wrapper
/// supplies owned contiguous arrays and discards outputs on every nonzero code.
#[no_mangle]
pub unsafe extern "C" fn rt_rollout_v1(
    kind: u32,
    batch: usize,
    horizon: usize,
    parameters: *const f32,
    parameter_len: usize,
    future: *const f32,
    future_len: usize,
    initial: *const f32,
    initial_len: usize,
    prediction: *mut f32,
    prediction_len: usize,
    final_state: *mut f32,
    final_len: usize,
) -> i32 {
    let expected_parameters = rt_parameter_count(kind);
    let expected_future = batch.checked_mul(horizon).and_then(|v| v.checked_mul(6));
    let expected_state = batch.checked_mul(12);
    if expected_parameters == 0 || parameter_len != expected_parameters || batch == 0
        || expected_future != Some(future_len) || expected_future != Some(prediction_len)
        || expected_state != Some(initial_len) || expected_state != Some(final_len)
    {
        return SHAPE;
    }
    let raw_ranges = [
        extent(parameters as usize, parameter_len), extent(future as usize, future_len),
        extent(initial as usize, initial_len), extent(prediction as usize, prediction_len),
        extent(final_state as usize, final_len),
    ];
    if raw_ranges.iter().any(Option::is_none) {
        return POINTER;
    }
    let ranges = raw_ranges.map(|r| r.unwrap());
    for out in 3..5 {
        for input in 0..out {
            if overlap(ranges[out], ranges[input]) {
                return POINTER;
            }
        }
    }
    let p = slice::from_raw_parts(parameters, parameter_len);
    let u = if future_len == 0 { &[] } else { slice::from_raw_parts(future, future_len) };
    let initial_values = slice::from_raw_parts(initial, initial_len);
    if !finite(p) || !finite(u) || !finite(initial_values) {
        return NONFINITE;
    }
    if p[..12].iter().any(|value| *value <= 0.0) {
        return DOMAIN;
    }
    let predictions = if prediction_len == 0 { &mut [] } else { slice::from_raw_parts_mut(prediction, prediction_len) };
    let final_values = slice::from_raw_parts_mut(final_state, final_len);
    for b in 0..batch {
        let mut current = [0.0f32; 12];
        current.copy_from_slice(&initial_values[b * 12..(b + 1) * 12]);
        for t in 0..horizon {
            let offset = (b * horizon + t) * 6;
            current = match step(kind, p, &current, &u[offset..offset + 6]) {
                Ok(value) => value,
                Err(code) => return code,
            };
            predictions[offset..offset + 6].copy_from_slice(&current[..6]);
        }
        final_values[b * 12..(b + 1) * 12].copy_from_slice(&current);
    }
    OK
}
