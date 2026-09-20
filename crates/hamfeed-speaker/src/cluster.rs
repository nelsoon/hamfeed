//! Per-group cosine clustering with `Unknown-N` labels.
//!
//! Greedy nearest-prototype matcher: the nearest proto with cosine ≥
//! threshold wins (and absorbs the frame into its L2-normalized running
//! mean); otherwise a fresh persisted N is minted via the caller-supplied
//! `alloc` closure (`|| store.alloc_speaker_n(day)` at the pipeline
//! call site).

/// Greedy nearest-proto clusterer over one window (group).
#[derive(Debug, Default)]
pub struct Clusterer {
    threshold: f32,
    protos: Vec<(u32, Vec<f32>)>,
}

impl Clusterer {
    /// New clusterer accepting matches with cosine similarity ≥ `threshold`.
    pub fn new(threshold: f32) -> Self {
        Self {
            threshold,
            protos: Vec::new(),
        }
    }

    /// Assign `emb` to the nearest proto (cosine ≥ threshold) or mint a
    /// fresh id via `alloc`. Returns `(id, is_new)`.
    pub fn assign(&mut self, emb: &[f32], alloc: &mut impl FnMut() -> u32) -> (u32, bool) {
        let norm = emb.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
        let normed: Vec<f32> = emb.iter().map(|x| x / norm).collect();
        let mut best: Option<(usize, f32)> = None;
        for (i, (_, proto)) in self.protos.iter().enumerate() {
            let s = crate::cosine(proto, &normed);
            if s >= self.threshold && best.is_none_or(|(_, b)| s > b) {
                best = Some((i, s));
            }
        }
        if let Some((i, _)) = best {
            let id = self.protos[i].0;
            let proto = &mut self.protos[i].1;
            for (p, e) in proto.iter_mut().zip(normed.iter()) {
                *p += *e;
            }
            let n = proto.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
            for p in proto.iter_mut() {
                *p /= n;
            }
            (id, false)
        } else {
            let id = alloc();
            self.protos.push((id, normed));
            (id, true)
        }
    }

    /// Clear protos only — N comes from the store, so it can never be reused.
    pub fn reset_window(&mut self) {
        self.protos.clear();
    }

    /// Display label for a persisted speaker number.
    pub fn label(n: u32) -> String {
        format!("Unknown-{n}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cluster_alternating_two_ids() {
        let mut c = Clusterer::new(0.55);
        let mut n = 0u32;
        let mut alloc = || {
            n += 1;
            n
        };
        let a = vec![1.0f32, 0.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0, 0.0];
        assert_eq!(c.assign(&a, &mut alloc), (1, true));
        assert_eq!(c.assign(&b, &mut alloc), (2, true));
        assert_eq!(c.assign(&a, &mut alloc), (1, false));
        assert_eq!(c.assign(&b, &mut alloc), (2, false));
    }

    #[test]
    fn near_duplicate_joins_existing() {
        let mut c = Clusterer::new(0.55);
        let mut n = 0u32;
        let mut alloc = || {
            n += 1;
            n
        };
        assert_eq!(c.assign(&[1.0f32, 0.0], &mut alloc), (1, true));
        assert_eq!(c.assign(&[0.98f32, 0.199], &mut alloc), (1, false));
    }

    #[test]
    fn reset_mints_fresh() {
        let mut c = Clusterer::new(0.55);
        let mut n = 0u32;
        let mut alloc = || {
            n += 1;
            n
        };
        let a = vec![1.0f32, 0.0];
        assert_eq!(c.assign(&a, &mut alloc), (1, true));
        c.reset_window();
        // Same voice after reset mints a FRESH N from the store counter.
        assert_eq!(c.assign(&a, &mut alloc), (2, true));
    }
}
