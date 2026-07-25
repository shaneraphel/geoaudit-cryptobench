//! Deterministic geometry kernels for GeoAudit receptor-only pocket detection.
//!
//! Two hot loops are ported from the NumPy reference, operation-for-operation, so
//! results are bit-identical:
//!
//! * `gk_free_grid_mask` — nearest-atom distance per grid point, then the
//!   `keep = dmin > atom_r` / `near = dmin < near_r` masks.
//! * `gk_buriedness` — fraction of probe directions blocked by an atom inside the
//!   ray cylinder.
//! * `gk_local_free_enclosed` — count of probe points that are outside every atom
//!   and still enclosed by at least `enclose_min` atoms.
//!
//! Determinism: every reduction is a `min`, a boolean `any`, or an integer count,
//! and the per-pair arithmetic is evaluated in a fixed order. None is sensitive to
//! thread scheduling, so the parallel result equals the serial result exactly.
//! No floating-point value is accumulated across atoms.

use std::thread;

/// Threads used for the point-parallel loops. Capped so an oversubscribed host
/// does not thrash; parallelism is over independent output slots only.
fn n_threads(work: usize) -> usize {
    if work < 512 {
        return 1;
    }
    let avail = thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    avail.clamp(1, 16).min(work)
}

#[inline(always)]
fn min_dist(p: &[f64; 3], coords: &[f64], n_atoms: usize) -> f64 {
    let mut best = f64::INFINITY;
    for a in 0..n_atoms {
        let dx = p[0] - coords[3 * a];
        let dy = p[1] - coords[3 * a + 1];
        let dz = p[2] - coords[3 * a + 2];
        let d2 = dx * dx + dy * dy + dz * dz;
        if d2 < best {
            best = d2;
        }
    }
    best.sqrt()
}

/// # Safety
/// `pts` has `3 * n_pts` f64s, `coords` has `3 * n_atoms` f64s, and `out_keep` /
/// `out_near` each have `n_pts` bytes. All must be valid, non-overlapping.
#[no_mangle]
pub unsafe extern "C" fn gk_free_grid_mask(
    pts: *const f64,
    n_pts: usize,
    coords: *const f64,
    n_atoms: usize,
    atom_r: f64,
    near_r: f64,
    out_keep: *mut u8,
    out_near: *mut u8,
) {
    if n_pts == 0 || n_atoms == 0 {
        return;
    }
    let pts = std::slice::from_raw_parts(pts, 3 * n_pts);
    let coords = std::slice::from_raw_parts(coords, 3 * n_atoms);
    let keep = std::slice::from_raw_parts_mut(out_keep, n_pts);
    let near = std::slice::from_raw_parts_mut(out_near, n_pts);

    let nt = n_threads(n_pts);
    let chunk = n_pts.div_ceil(nt);
    thread::scope(|s| {
        let mut k_rest: &mut [u8] = keep;
        let mut n_rest: &mut [u8] = near;
        let mut base = 0usize;
        while base < n_pts {
            let take = chunk.min(n_pts - base);
            let (k_head, k_tail) = k_rest.split_at_mut(take);
            let (n_head, n_tail) = n_rest.split_at_mut(take);
            let off = base;
            s.spawn(move || {
                for i in 0..take {
                    let g = off + i;
                    let p = [pts[3 * g], pts[3 * g + 1], pts[3 * g + 2]];
                    let dmin = min_dist(&p, coords, n_atoms);
                    k_head[i] = u8::from(dmin > atom_r);
                    n_head[i] = u8::from(dmin < near_r);
                }
            });
            k_rest = k_tail;
            n_rest = n_tail;
            base += take;
        }
    });
}

/// # Safety
/// `pts` has `3 * n_pts` f64s, `coords` has `3 * n_atoms` f64s, `dirs` has
/// `3 * n_dirs` f64s, and `out` has `n_pts` f64s. All must be valid.
#[no_mangle]
pub unsafe extern "C" fn gk_buriedness(
    pts: *const f64,
    n_pts: usize,
    coords: *const f64,
    n_atoms: usize,
    dirs: *const f64,
    n_dirs: usize,
    cutoff: f64,
    perp2: f64,
    out: *mut f64,
) {
    if n_pts == 0 {
        return;
    }
    let pts = std::slice::from_raw_parts(pts, 3 * n_pts);
    let coords = std::slice::from_raw_parts(coords, 3 * n_atoms);
    let dirs = std::slice::from_raw_parts(dirs, 3 * n_dirs);
    let out = std::slice::from_raw_parts_mut(out, n_pts);
    let cut2 = cutoff * cutoff;

    let nt = n_threads(n_pts);
    let chunk = n_pts.div_ceil(nt);
    thread::scope(|s| {
        let mut rest: &mut [f64] = out;
        let mut base = 0usize;
        while base < n_pts {
            let take = chunk.min(n_pts - base);
            let (head, tail) = rest.split_at_mut(take);
            let off = base;
            s.spawn(move || {
                let mut blocked = vec![false; n_dirs];
                for i in 0..take {
                    let g = off + i;
                    let px = pts[3 * g];
                    let py = pts[3 * g + 1];
                    let pz = pts[3 * g + 2];
                    blocked.iter_mut().for_each(|b| *b = false);
                    let mut any_atom = false;
                    for a in 0..n_atoms {
                        let rx = coords[3 * a] - px;
                        let ry = coords[3 * a + 1] - py;
                        let rz = coords[3 * a + 2] - pz;
                        let r2 = rx * rx + ry * ry + rz * rz;
                        if r2 > cut2 {
                            continue; // outside the cutoff sphere
                        }
                        any_atom = true;
                        for d in 0..n_dirs {
                            if blocked[d] {
                                continue;
                            }
                            let t = rx * dirs[3 * d] + ry * dirs[3 * d + 1]
                                + rz * dirs[3 * d + 2];
                            if t > 0.0 && t <= cutoff && (r2 - t * t) <= perp2 {
                                blocked[d] = true;
                            }
                        }
                    }
                    head[i] = if !any_atom {
                        0.0
                    } else {
                        blocked.iter().filter(|b| **b).count() as f64 / n_dirs as f64
                    };
                }
            });
            rest = tail;
            base += take;
        }
    });
}

/// Count probe points that are free (outside every atom radius) AND still enclosed
/// (at least `enclose_min` atoms within `enclose_cut`).
///
/// Port of the NumPy reference, operation-for-operation:
/// ```text
/// d2   = ((pts[:, None, :] - near[None, :, :]) ** 2).sum(-1)
/// hit  = (sqrt(d2.min(1)) > atom_r) & ((d2 <= enclose_cut**2).sum(1) >= enclose_min)
/// ```
/// The squared distance is accumulated in the same left-to-right order and the
/// wall test is `sqrt(min) > atom_r` (not `min > atom_r^2`) so the rounding of the
/// square root matches NumPy exactly. Per point the reductions are a `min` and an
/// integer count; across points the results are summed as integers. Nothing here
/// depends on how the point range is split, so the thread count is not observable.
///
/// # Safety
/// `pts` has `3 * n_pts` f64s and `coords` has `3 * n_atoms` f64s, both valid for
/// the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn gk_local_free_enclosed(
    pts: *const f64,
    n_pts: usize,
    coords: *const f64,
    n_atoms: usize,
    atom_r: f64,
    enclose_cut: f64,
    enclose_min: u64,
) -> u64 {
    if n_pts == 0 || n_atoms == 0 {
        return 0;
    }
    let pts = std::slice::from_raw_parts(pts, 3 * n_pts);
    let coords = std::slice::from_raw_parts(coords, 3 * n_atoms);
    let cut2 = enclose_cut * enclose_cut;

    let nt = n_threads(n_pts);
    let chunk = n_pts.div_ceil(nt);
    let mut total: u64 = 0;
    thread::scope(|s| {
        let mut handles = Vec::with_capacity(nt);
        let mut base = 0usize;
        while base < n_pts {
            let take = chunk.min(n_pts - base);
            let off = base;
            handles.push(s.spawn(move || {
                let mut hits: u64 = 0;
                for i in 0..take {
                    let g = off + i;
                    let px = pts[3 * g];
                    let py = pts[3 * g + 1];
                    let pz = pts[3 * g + 2];
                    let mut best = f64::INFINITY;
                    let mut within: u64 = 0;
                    for a in 0..n_atoms {
                        let dx = px - coords[3 * a];
                        let dy = py - coords[3 * a + 1];
                        let dz = pz - coords[3 * a + 2];
                        let d2 = dx * dx + dy * dy + dz * dz;
                        if d2 < best {
                            best = d2;
                        }
                        if d2 <= cut2 {
                            within += 1;
                        }
                    }
                    if best.sqrt() > atom_r && within >= enclose_min {
                        hits += 1;
                    }
                }
                hits
            }));
            base += take;
        }
        for h in handles {
            total += h.join().unwrap();
        }
    });
    total
}
