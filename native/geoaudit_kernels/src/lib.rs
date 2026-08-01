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

// ---------------------------------------------------------------------------
// Table-bank addressing.
//
// `table_bank.addresses` is the one function every consumer of the counting
// field goes through: `compile_cells` calls it once per block, `scatter_and_means`
// twice, and `score` once. For a bank of 10,144 tables at width 2 it performs
// 20,288 integer multiply-accumulate passes over each block of 8,192 rows, and
// NumPy runs all of it on one core -- the BLAS that makes the rest of this
// pipeline fast is not involved, because there is no floating-point product here.
//
// This is a straight port, parallel over row blocks. It is bit-identical rather
// than approximately equal, and that is not a courtesy: the addresses index a
// cell array, so an address off by one is a different cell and a different score,
// with no numerical smallness to make the error visible. Every operation is
// integer, the accumulation order per row is the same as NumPy's (column of the
// table, ascending), and no float is touched, so equality is a property of the
// port and not something to be measured with a tolerance.
//
// Parallelism is over disjoint output row ranges, so the result does not depend
// on thread scheduling or on the thread count.

/// # Safety
/// `d` has `n_rows * n_cols` i8s, `cols` has `n_tables * width` i32s each a
/// valid column index, `offsets` has `n_tables` i64s, and `out` has
/// `n_rows * n_tables` i64s. All must be valid and non-overlapping.
#[no_mangle]
pub unsafe extern "C" fn gk_table_addresses(
    d: *const i8,
    n_rows: usize,
    n_cols: usize,
    cols: *const i32,
    n_tables: usize,
    width: usize,
    offsets: *const i64,
    n_levels: i64,
    out: *mut i64,
) -> i32 {
    if n_rows == 0 || n_tables == 0 {
        return 0;
    }
    let d = std::slice::from_raw_parts(d, n_rows * n_cols);
    let cols = std::slice::from_raw_parts(cols, n_tables * width);
    let offsets = std::slice::from_raw_parts(offsets, n_tables);
    let out = std::slice::from_raw_parts_mut(out, n_rows * n_tables);

    // Refuse rather than read out of bounds. A column index past the end of D
    // would be a silent read of another row's digit, which is exactly the class
    // of error this port must not introduce.
    for &c in cols {
        if c < 0 || (c as usize) >= n_cols {
            return -1;
        }
    }

    let nt = n_threads(n_rows);
    let chunk = n_rows.div_ceil(nt);
    thread::scope(|s| {
        let mut rest: &mut [i64] = out;
        let mut base = 0usize;
        while base < n_rows {
            let take = chunk.min(n_rows - base);
            let (head, tail) = rest.split_at_mut(take * n_tables);
            let off = base;
            s.spawn(move || {
                for r in 0..take {
                    let row = &d[(off + r) * n_cols..(off + r + 1) * n_cols];
                    let dst = &mut head[r * n_tables..(r + 1) * n_tables];
                    for k in 0..n_tables {
                        // Ascending column of the table, matching the NumPy
                        // loop, so the powers of n_levels are applied in the
                        // same order.
                        let mut acc: i64 = 0;
                        let mut place: i64 = 1;
                        for t in 0..width {
                            let c = cols[k * width + t] as usize;
                            acc += (row[c] as i64) * place;
                            place *= n_levels;
                        }
                        dst[k] = acc + offsets[k];
                    }
                }
            });
            rest = tail;
            base += take;
        }
    });
    0
}

// ---------------------------------------------------------------------------
// Table-bank cell counts.
//
// `compile_cells` needs two reductions over the same addresses: how many training
// rows land in each cell, and how many of those rows are positive. NumPy does it
// by materialising the whole (block, n_tables) address matrix and calling bincount
// twice -- at 8,192 rows and 10,144 tables that matrix is 665 MB, written once and
// read twice, and both bincounts are single-threaded.
//
// Fusing the addressing into the reduction removes the matrix entirely. Each
// thread keeps its own pair of accumulators over the cell space and they are
// summed at the end.
//
// Bit-identity holds and the reason is worth stating, because a float sum split
// across threads normally would not be reproducible. The positive count is a sum
// of the label, and the label is exactly 0 or 1: the quantity is an integer, it is
// accumulated here as an i64, and an integer sum does not depend on the order the
// terms arrive in. So the parallel result equals the serial result exactly, and
// equals NumPy's float64 bincount exactly as well, for as long as a count fits in
// 2^53 -- which at one increment per training row per table is not close.
//
// The caller divides to get frequencies. That division is left in Python so the
// one place a float appears is the one place it is unavoidable.

/// # Safety
/// `d` has `n_rows * n_cols` i8s, `y` has `n_rows` u8s each 0 or 1, `cols` has
/// `n_tables * width` valid column indices, `offsets` has `n_tables` i64s, and
/// `out_total` / `out_pos` each have `n_cells` i64s zeroed by the caller.
#[no_mangle]
pub unsafe extern "C" fn gk_table_cell_counts(
    d: *const i8,
    n_rows: usize,
    n_cols: usize,
    y: *const u8,
    cols: *const i32,
    n_tables: usize,
    width: usize,
    offsets: *const i64,
    n_levels: i64,
    n_cells: usize,
    out_total: *mut i64,
    out_pos: *mut i64,
) -> i32 {
    if n_rows == 0 || n_tables == 0 || n_cells == 0 {
        return 0;
    }
    let d = std::slice::from_raw_parts(d, n_rows * n_cols);
    let y = std::slice::from_raw_parts(y, n_rows);
    let cols = std::slice::from_raw_parts(cols, n_tables * width);
    let offsets = std::slice::from_raw_parts(offsets, n_tables);
    let total = std::slice::from_raw_parts_mut(out_total, n_cells);
    let pos = std::slice::from_raw_parts_mut(out_pos, n_cells);

    for &c in cols {
        if c < 0 || (c as usize) >= n_cols {
            return -1;
        }
    }
    // An address outside the cell space would be a write past the end of the
    // accumulator. The bound is checked per table from its own offset and the
    // maximum digit, so the row loop below needs no test at all.
    let max_digit = n_levels - 1;
    for k in 0..n_tables {
        let mut top: i64 = 0;
        let mut place: i64 = 1;
        for _ in 0..width {
            top += max_digit * place;
            place *= n_levels;
        }
        if offsets[k] < 0 || (offsets[k] + top) as usize >= n_cells {
            return -2;
        }
    }

    let nt = n_threads(n_rows);
    let chunk = n_rows.div_ceil(nt);
    let parts: Vec<(Vec<i64>, Vec<i64>)> = thread::scope(|s| {
        let mut handles = Vec::new();
        let mut base = 0usize;
        while base < n_rows {
            let take = chunk.min(n_rows - base);
            let off = base;
            handles.push(s.spawn(move || {
                let mut t = vec![0i64; n_cells];
                let mut p = vec![0i64; n_cells];
                for r in off..off + take {
                    let row = &d[r * n_cols..(r + 1) * n_cols];
                    let yr = y[r] as i64;
                    for k in 0..n_tables {
                        let mut acc: i64 = 0;
                        let mut place: i64 = 1;
                        for w in 0..width {
                            let c = cols[k * width + w] as usize;
                            acc += (row[c] as i64) * place;
                            place *= n_levels;
                        }
                        let a = (acc + offsets[k]) as usize;
                        t[a] += 1;
                        p[a] += yr;
                    }
                }
                (t, p)
            }));
            base += take;
        }
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    for (t, p) in parts {
        for i in 0..n_cells {
            total[i] += t[i];
            pos[i] += p[i];
        }
    }
    0
}
