use std::{env, fs, hint::black_box, time::Instant};

fn kernel(row: &[f64], k: usize) -> Vec<f64> {
    let max = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let candidate_max = row[..k].iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut probs: Vec<f64> = row[..k].iter().map(|x| (x-candidate_max).exp()).collect();
    let total: f64 = probs.iter().sum();
    for p in &mut probs { *p /= total; }
    let full_sum: f64 = row.iter().map(|x| (x-max).exp()).sum();
    let mass: f64 = row[..k].iter().map(|x| (x-max).exp()).sum::<f64>() / full_sum;
    let entropy: f64 = -probs.iter().filter(|p| **p > 0.0).map(|p| p*p.ln()).sum::<f64>();
    probs.extend([mass, entropy]);
    probs
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let width: usize = args[2].parse().unwrap();
    let k: usize = args[3].parse().unwrap();
    let repeats: usize = args[4].parse().unwrap();
    let bytes = fs::read(&args[1]).unwrap();
    assert!(width >= k && k >= 2 && bytes.len() % (width*8) == 0);
    let values: Vec<f64> = bytes.chunks_exact(8).map(|b| f64::from_le_bytes(b.try_into().unwrap())).collect();
    assert!(values.iter().all(|v| v.is_finite()));
    let run = || -> Vec<Vec<f64>> { values.chunks_exact(width).map(|r| kernel(r,k)).collect() };
    for _ in 0..3 { black_box(run()); }
    let mut times = vec![];
    let mut result = vec![];
    for _ in 0..repeats {
        let start = Instant::now();
        result = black_box(run());
        times.push(start.elapsed().as_secs_f64()*1000.0);
    }
    println!("{{\"batch_ms\":{:?},\"outputs\":{:?}}}", times, result);
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn stable_extremes_and_mass() {
        let a=kernel(&[1000.0,999.0,1010.0],2);
        assert!((a[0]+a[1]-1.0).abs()<1e-12);
        assert!(a[2]<0.001);
        assert!(a[3]>0.0);
        assert_eq!(kernel(&[-1000.0,-1000.0],2)[..2], [0.5,0.5]);
    }
}
