#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Core computational functions used by the WRR drought-propagation workflow.

This release module was refactored from the analysis scripts used for the manuscript.
The numerical algorithms are intentionally kept compatible with the supplied analysis code.
"""

from __future__ import annotations

import hashlib
import math
import time
from pathlib import Path

import geopandas as gpd
import h5py
import numpy as np
import numba as nb
import rasterio
import scipy.stats as st
from numba import njit, prange
from rasterio.features import geometry_mask
from rasterio.transform import from_bounds
from scipy import stats
from scipy.sparse import csr_matrix, eye as speye, diags
from shapely.geometry import mapping

EVE_NUM_MAX = 10000

def arr_to_tiff(arr, lat_arr, lon_arr, out_loc, flip=True):
    if flip:
        arr = np.flipud(arr)
    lat_res = (lat_arr.max() - lat_arr.min()) / (len(lat_arr) - 1)
    lon_res = (lon_arr.max() - lon_arr.min()) / (len(lon_arr) - 1)
    transform = from_bounds(lon_arr.min() - lon_res / 2, lat_arr.min() - lat_res / 2, lon_arr.max() + lon_res / 2, lat_arr.max() + lat_res / 2, len(lon_arr), len(lat_arr))
    with rasterio.open(out_loc, 'w', driver='GTiff', height=arr.shape[-2], width=arr.shape[-1], count=arr.shape[0] if arr.ndim == 3 else 1, dtype='float32', crs='EPSG:4326', transform=transform, nodata=np.nan) as dst:
        if arr.ndim == 3:
            dst.write(arr)
        else:
            dst.write(arr, 1)


def create_mask_from_shapefile(lat, lon, shapefile_path, inv=True):
    gdf = gpd.read_file(shapefile_path)
    if gdf.crs != 'EPSG:4326':
        gdf = gdf.to_crs('EPSG:4326')
    geometry = gdf.geometry.unary_union
    lat_min, lat_max = (lat.min(), lat.max())
    lon_min, lon_max = (lon.min(), lon.max())
    transform = from_bounds(lon_min, lat_min, lon_max, lat_max, len(lon), len(lat))
    mask = geometry_mask([mapping(geometry)], out_shape=(len(lat), len(lon)), transform=transform, invert=inv)
    return mask


@njit(nogil=True)
def drought_ex_parallel(spi_arr, th, d_th, eve_num_max=EVE_NUM_MAX):
    la_dim, lo_dim, t_dim = (spi_arr.shape[1], spi_arr.shape[2], spi_arr.shape[0])
    print('the dim of data (lat, lon, time): ', la_dim, lo_dim, t_dim)
    evt_srs = np.zeros((la_dim, lo_dim, eve_num_max), dtype=np.int32)
    evt_sre = np.zeros((la_dim, lo_dim, eve_num_max), dtype=np.int32)
    evt_num = np.zeros((la_dim, lo_dim), dtype=np.int32)
    for i in prange(evt_num.shape[0]):
        for j in prange(evt_num.shape[1]):
            spi_each = spi_arr[:, i, j]
            evt_tim_e = np.where(spi_each < th)[0]
            csd_srs, csd_end, num = valid_ee_tim(evt_tim_e, d_th, eve_num_max=eve_num_max)
            evt_num[i, j] = num
            evt_srs[i, j, :evt_num[i, j]] = csd_srs
            evt_sre[i, j, :evt_num[i, j]] = csd_end
    return (evt_num, evt_srs, evt_sre)


@njit(nogil=True)
def fd_ex_parallel(spi_arr, min_duration, delta_threshold, final_threshold, eve_num_max=EVE_NUM_MAX):
    la_dim, lo_dim, t_dim = (spi_arr.shape[1], spi_arr.shape[2], spi_arr.shape[0])
    print('the dim of data (lat, lon, time): ', la_dim, lo_dim, t_dim)
    evt_srs = np.zeros((la_dim, lo_dim, eve_num_max), dtype=np.int32)
    evt_sre = np.zeros((la_dim, lo_dim, eve_num_max), dtype=np.int32)
    evt_num = np.zeros((la_dim, lo_dim), dtype=np.int32)
    for i in prange(evt_num.shape[0]):
        for j in prange(evt_num.shape[1]):
            spi_each = spi_arr[:, i, j]
            csd_srs, csd_end, num = valid_fd_tim(spi_each, min_duration, delta_threshold, final_threshold, eve_num_max=eve_num_max)
            evt_num[i, j] = num
            evt_srs[i, j, :evt_num[i, j]] = csd_srs
            evt_sre[i, j, :evt_num[i, j]] = csd_end
    return (evt_num, evt_srs, evt_sre)


@njit
def valid_ee_tim(arr, d_th, eve_num_max=EVE_NUM_MAX):
    out_s = np.zeros(eve_num_max, dtype=np.int32)
    out_e = np.zeros(eve_num_max, dtype=np.int32)
    eve_count = 0
    k = 0
    while k < arr.shape[0] - 1:
        start = k
        while k < arr.shape[0] - 1 and arr[k] + 1 == arr[k + 1]:
            k += 1
        if k - start >= d_th:
            out_s[eve_count] = arr[start]
            out_e[eve_count] = arr[k]
            eve_count += 1
        k += 1
    return (out_s[:eve_count], out_e[:eve_count], eve_count)


@njit
def valid_fd_tim(spi, min_duration, delta_threshold, final_threshold, eve_num_max=EVE_NUM_MAX):
    out_s = np.zeros(eve_num_max, dtype=np.int32)
    out_e = np.zeros(eve_num_max, dtype=np.int32)
    cnt = 0
    t = 0
    T = spi.shape[0]
    delta_spi = spi[min_duration:] - spi[:-min_duration]
    while t + min_duration + 1 < T:
        if delta_spi[t] <= delta_threshold:
            window_min = np.min(spi[t:t + min_duration])
            if window_min <= final_threshold:
                start_in_window = np.where(spi[t:t + min_duration] <= final_threshold)[0]
                if start_in_window.size == 0:
                    t += 1
                    continue
                start = t + start_in_window[0]
                end = start
                while end < T and spi[end] <= final_threshold:
                    end += 1
                end = min(end - 1, T - 1)
                t = end + 1
                out_s[cnt] = start
                out_e[cnt] = end
                cnt += 1
            else:
                t += 1
        else:
            t += 1
    return (out_s[:cnt], out_e[:cnt], cnt)


@njit(parallel=True)
def event_sync_drought_bidirectional(evt_srs, evt_num, tau_m):
    nds = evt_num.shape[0]
    valid_pairs_tmp = []
    for i in range(nds):
        if evt_num[i] >= 3:
            for j in range(i + 1, nds):
                if evt_num[j] >= 3:
                    valid_pairs_tmp.append((i, j))
    n_pairs = len(valid_pairs_tmp)
    pairs = np.zeros((n_pairs, 2), dtype=np.int32)
    for k in range(n_pairs):
        pairs[k, 0] = valid_pairs_tmp[k][0]
        pairs[k, 1] = valid_pairs_tmp[k][1]
    results = np.zeros((n_pairs, 4), dtype=np.float32)
    for idx in prange(n_pairs):
        i = pairs[idx, 0]
        j = pairs[idx, 1]
        j_j2i = es_directional_count(evt_srs[i], evt_srs[j], evt_num[i], evt_num[j], tau_m)
        j_i2j = es_directional_count(evt_srs[j], evt_srs[i], evt_num[j], evt_num[i], tau_m)
        results[idx, 0] = i
        results[idx, 1] = j
        results[idx, 2] = j_j2i
        results[idx, 3] = j_i2j
    return results


def compute_Q_q_from_bidir(results_bidir, evt_num):
    n_pairs = results_bidir.shape[0]
    Q_results = np.zeros((n_pairs, 4), dtype=np.float32)
    for idx in range(n_pairs):
        i = int(results_bidir[idx, 0])
        j = int(results_bidir[idx, 1])
        j_j2i = results_bidir[idx, 2]
        j_i2j = results_bidir[idx, 3]
        sqrt_factor = np.sqrt(float(evt_num[i]) * float(evt_num[j]))
        if sqrt_factor > 0 and j_j2i + j_i2j > 0:
            Q_ij = (j_j2i + j_i2j) / sqrt_factor
            q_ij = (j_j2i - j_i2j) / sqrt_factor
        else:
            Q_ij = 0.0
            q_ij = 0.0
        Q_results[idx, 0] = i
        Q_results[idx, 1] = j
        Q_results[idx, 2] = Q_ij
        Q_results[idx, 3] = q_ij
    return Q_results


@njit(nogil=True)
def es_directional_count(evt_i, evt_j, n_i, n_j, tau_max):
    count = 0
    for i_idx in range(1, n_i):
        t_i = evt_i[i_idx]
        for j_idx in range(1, n_j):
            t_j = evt_j[j_idx]
            time_diff = t_i - t_j
            if abs(time_diff) > tau_max:
                continue
            if time_diff == 0:
                count += 0.5
            if 0 < time_diff < tau_max:
                count += 1
    return count


@njit(nogil=True, cache=True)
def _generate_random_events_numba(n, k, seed):
    state = (seed * 1103515245 + 12345) % 2 ** 31
    result = np.empty(k, dtype=np.int32)
    for i in range(k):
        state = (state * 1103515245 + 12345) % 2 ** 31
        result[i] = state % n
    result.sort()
    return result


@njit(parallel=True, cache=False)
def _batch_Q_critical_numba(ni_arr, nj_arr, len_time, tau_m, n_surrogates, quantile):
    n_combos = len(ni_arr)
    crit = np.zeros(n_combos, dtype=np.float32)
    for cidx in prange(n_combos):
        ni = int(ni_arr[cidx])
        nj = int(nj_arr[cidx])
        inv_sqrt = 1.0 / math.sqrt(float(ni) * float(nj))
        q_buf = np.empty(n_surrogates, dtype=np.float32)
        for s in range(n_surrogates):
            seed_i = int((cidx + 1) * 2654435769 ^ (s + 1) * 2246822519) & 2147483647
            seed_j = int((cidx + 1) * 2246822519 ^ (s + 1) * 2654435769) & 2147483647
            ev_i = _generate_random_events_numba(len_time, ni, seed_i)
            ev_j = _generate_random_events_numba(len_time, nj, seed_j)
            j_j2i = es_directional_count(ev_i, ev_j, ni, nj, tau_m)
            j_i2j = es_directional_count(ev_j, ev_i, nj, ni, tau_m)
            q_buf[s] = (j_j2i + j_i2j) * inv_sqrt
        q_sorted = np.sort(q_buf)
        pos = quantile / 100.0 * (n_surrogates - 1)
        lo = int(math.floor(pos))
        hi = min(lo + 1, n_surrogates - 1)
        frac = pos - lo
        crit[cidx] = q_sorted[lo] * (1.0 - frac) + q_sorted[hi] * frac
    return crit


def significance_surrogate_Q_v2(evt_num, len_time, tau_m, n_surrogates=200, quantile=95.0, cache_dir=None):
    t_total = time.perf_counter()
    valid_nodes = np.where(evt_num >= 3)[0].astype(np.int32)
    n_valid = len(valid_nodes)
    print(f'  [surr-v2] valid nodes={n_valid:,}  n_surr={n_surrogates}  q={quantile}')
    if n_valid < 2:
        return ([], np.zeros(0, dtype=np.float32))
    t1 = time.perf_counter()
    ii, jj = np.triu_indices(n_valid, k=1)
    pairs_i = valid_nodes[ii].astype(np.int32)
    pairs_j = valid_nodes[jj].astype(np.int32)
    n_pairs = len(pairs_i)
    print(f'  valid pairs={n_pairs:,}  generation time={time.perf_counter() - t1:.2f}s')
    ni_raw = evt_num[pairs_i].astype(np.int32)
    nj_raw = evt_num[pairs_j].astype(np.int32)
    swap = ni_raw > nj_raw
    ni_key = np.where(swap, nj_raw, ni_raw)
    nj_key = np.where(swap, ni_raw, nj_raw)
    MULT = np.int64(1000000)
    combo_keys = ni_key.astype(np.int64) * MULT + nj_key.astype(np.int64)
    unique_keys, inverse_idx = np.unique(combo_keys, return_inverse=True)
    unique_ni = (unique_keys // MULT).astype(np.int32)
    unique_nj = (unique_keys % MULT).astype(np.int32)
    n_unique = len(unique_keys)
    print(f'  unique combinations={n_unique:,}  (speed-up {n_pairs / max(n_unique, 1):.0f}x)')
    crit_table = None
    cache_path = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        raw = unique_keys.tobytes() + f'|{len_time}|{tau_m}|{n_surrogates}|{quantile}'.encode()
        tag = hashlib.sha256(raw).hexdigest()[:20]
        cache_path = cache_dir / f'Qcrit_{tag}.npy'
        if cache_path.exists():
            crit_table = np.load(str(cache_path))
            print(f'  [cache hit] {cache_path.name}  (Numba calculation skipped)')
    if crit_table is None:
        t2 = time.perf_counter()
        crit_table = _batch_Q_critical_numba(unique_ni, unique_nj, int(len_time), int(tau_m), int(n_surrogates), float(quantile))
        elapsed_numba = time.perf_counter() - t2
        print(f'  Numba batch calculation: {elapsed_numba:.1f}s  [{n_unique} combinations x {n_surrogates} surrogates]')
        if cache_path is not None:
            np.save(str(cache_path), crit_table)
            print(f'  [cache saved] {cache_path.name}')
    crit_Q = crit_table[inverse_idx].astype(np.float32)
    valid_pairs = list(zip(pairs_i.tolist(), pairs_j.tolist()))
    critical_values = crit_Q
    print(f'  [surr-v2] total time={time.perf_counter() - t_total:.1f}s')
    return (valid_pairs, critical_values)


@njit(parallel=True, cache=False)
def _batch_q_critical_numba(ni_arr, nj_arr, len_time, tau_m, n_surrogates, quantile):
    n_combos = len(ni_arr)
    crit = np.zeros(n_combos, dtype=np.float32)
    for cidx in prange(n_combos):
        ni = int(ni_arr[cidx])
        nj = int(nj_arr[cidx])
        inv_sqrt = 1.0 / math.sqrt(float(ni) * float(nj))
        q_abs_buf = np.empty(n_surrogates, dtype=np.float32)
        for s in range(n_surrogates):
            seed_i = int((cidx + 1) * 2654435769 ^ (s + 1) * 2246822519) & 2147483647
            seed_j = int((cidx + 1) * 2246822519 ^ (s + 1) * 2654435769) & 2147483647
            ev_i = _generate_random_events_numba(len_time, ni, seed_i)
            ev_j = _generate_random_events_numba(len_time, nj, seed_j)
            j_j2i = es_directional_count(ev_i, ev_j, ni, nj, tau_m)
            j_i2j = es_directional_count(ev_j, ev_i, nj, ni, tau_m)
            q_abs_buf[s] = abs(j_j2i - j_i2j) * inv_sqrt
        q_sorted = np.sort(q_abs_buf)
        pos = quantile / 100.0 * (n_surrogates - 1)
        lo = int(math.floor(pos))
        hi = min(lo + 1, n_surrogates - 1)
        frac = pos - lo
        crit[cidx] = q_sorted[lo] * (1.0 - frac) + q_sorted[hi] * frac
    return crit


def significance_surrogate_q_v2(evt_num, len_time, tau_m, n_surrogates=200, quantile=95.0, cache_dir=None):
    t_total = time.perf_counter()
    valid_nodes = np.where(evt_num >= 3)[0].astype(np.int32)
    n_valid = len(valid_nodes)
    print(f'  [surr-q-v2] valid nodes={n_valid:,}  n_surr={n_surrogates}  q={quantile}')
    if n_valid < 2:
        return ([], np.zeros(0, dtype=np.float32))
    ii, jj = np.triu_indices(n_valid, k=1)
    pairs_i = valid_nodes[ii].astype(np.int32)
    pairs_j = valid_nodes[jj].astype(np.int32)
    ni_raw = evt_num[pairs_i].astype(np.int32)
    nj_raw = evt_num[pairs_j].astype(np.int32)
    swap = ni_raw > nj_raw
    ni_key = np.where(swap, nj_raw, ni_raw)
    nj_key = np.where(swap, ni_raw, nj_raw)
    MULT = np.int64(1000000)
    combo_keys = ni_key.astype(np.int64) * MULT + nj_key.astype(np.int64)
    unique_keys, inverse_idx = np.unique(combo_keys, return_inverse=True)
    unique_ni = (unique_keys // MULT).astype(np.int32)
    unique_nj = (unique_keys % MULT).astype(np.int32)
    n_unique = len(unique_keys)
    print(f'  unique combinations={n_unique:,}')
    crit_table = None
    cache_path = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        raw = unique_keys.tobytes() + f'|q|{len_time}|{tau_m}|{n_surrogates}|{quantile}'.encode()
        tag = hashlib.sha256(raw).hexdigest()[:20]
        cache_path = cache_dir / f'qcrit_{tag}.npy'
        if cache_path.exists():
            crit_table = np.load(str(cache_path))
            print(f'  [cache hit] {cache_path.name}')
    if crit_table is None:
        t2 = time.perf_counter()
        crit_table = _batch_q_critical_numba(unique_ni, unique_nj, int(len_time), int(tau_m), int(n_surrogates), float(quantile))
        elapsed = time.perf_counter() - t2
        print(f'  Numba batch calculation: {elapsed:.1f}s')
        if cache_path is not None:
            np.save(str(cache_path), crit_table)
            print(f'  [cache saved] {cache_path.name}')
    q_crit_values = crit_table[inverse_idx].astype(np.float32)
    valid_pairs = list(zip(pairs_i.tolist(), pairs_j.tolist()))
    print(f'  [surr-q-v2] total time={time.perf_counter() - t_total:.1f}s')
    return (valid_pairs, q_crit_values)


def build_significant_Q_q_matrices_dual(Q_results, Q_crit_values, q_crit_values, n_nodes):
    Q_matrix = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    q_matrix = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    n_Q_sig = 0
    n_q_sig = 0
    for idx in range(len(Q_results)):
        i = int(Q_results[idx, 0])
        j = int(Q_results[idx, 1])
        Q_ij = float(Q_results[idx, 2])
        q_ij = float(Q_results[idx, 3])
        Q_crit = float(Q_crit_values[idx]) if idx < len(Q_crit_values) else 0.0
        q_crit = float(q_crit_values[idx]) if idx < len(q_crit_values) else 0.0
        if Q_ij > Q_crit:
            Q_matrix[i, j] = Q_ij
            Q_matrix[j, i] = Q_ij
            n_Q_sig += 1
        if abs(q_ij) > q_crit:
            q_matrix[i, j] = q_ij
            q_matrix[j, i] = -q_ij
            n_q_sig += 1
    total = max(len(Q_results), 1)
    print(f'  Q-significant pairs: {n_Q_sig:,} / {total:,} ({100 * n_Q_sig / total:.1f}%)')
    print(f'  q-significant pairs: {n_q_sig:,} / {total:,} ({100 * n_q_sig / total:.1f}%)')
    return (Q_matrix, q_matrix)


@njit(parallel=True, nogil=True)
def compute_degree_centrality_standard(adj_binary):
    n_nodes = adj_binary.shape[0]
    dc = np.zeros(n_nodes, dtype=np.float32)
    for i in prange(n_nodes):
        degree = 0
        for j in range(n_nodes):
            if i != j and adj_binary[i, j] > 0:
                degree += 1
        dc[i] = degree / (n_nodes - 1) if n_nodes > 1 else 0.0
    return dc


@njit(parallel=True, nogil=True)
def compute_degree_centrality_weighted(Q_matrix):
    n_nodes = Q_matrix.shape[0]
    wdc = np.zeros(n_nodes, dtype=np.float32)
    for i in prange(n_nodes):
        total_strength = 0.0
        for j in range(n_nodes):
            if i != j:
                total_strength += Q_matrix[i, j]
        wdc[i] = total_strength / (n_nodes - 1) if n_nodes > 1 else 0.0
    return wdc


@njit(parallel=True, nogil=True, cache=True)
def bfs_shortest_paths_parallel(adj_matrix, source):
    n_nodes = adj_matrix.shape[0]
    distance = np.full(n_nodes, -1, dtype=np.int32)
    n_paths = np.zeros(n_nodes, dtype=np.float64)
    queue = np.zeros(n_nodes, dtype=np.int32)
    distance[source] = 0
    n_paths[source] = 1.0
    queue[0] = source
    front = 0
    rear = 1
    while front < rear:
        current = queue[front]
        front += 1
        current_row = adj_matrix[current]
        for neighbor in range(n_nodes):
            if current_row[neighbor] <= 0 or neighbor == current:
                continue
            if distance[neighbor] == -1:
                distance[neighbor] = distance[current] + 1
                n_paths[neighbor] = n_paths[current]
                queue[rear] = neighbor
                rear += 1
            elif distance[neighbor] == distance[current] + 1:
                n_paths[neighbor] += n_paths[current]
    return (distance, n_paths, queue[:rear])


@njit(parallel=True, nogil=True, cache=True)
def accumulate_betweenness_parallel(adj_matrix, sources, n_sources):
    n_nodes = adj_matrix.shape[0]
    bc_local = np.zeros((n_sources, n_nodes), dtype=np.float64)
    for idx in prange(n_sources):
        source = sources[idx]
        distance, n_paths, visited = bfs_shortest_paths_parallel(adj_matrix, source)
        dependency = np.zeros(n_nodes, dtype=np.float64)
        for i in range(len(visited) - 1, -1, -1):
            w = visited[i]
            for v in range(n_nodes):
                if adj_matrix[v, w] > 0 and distance[v] == distance[w] - 1:
                    if n_paths[w] > 0:
                        dependency[v] += n_paths[v] / n_paths[w] * (1.0 + dependency[w])
        dependency[source] = 0
        bc_local[idx] = dependency
    return bc_local


@njit(parallel=True, nogil=True, cache=True)
def compute_betweenness_centrality_parallel_numba(adj_matrix, k=None):
    n_nodes = adj_matrix.shape[0]
    if k is None or k >= n_nodes:
        sample_size = n_nodes
        sample_nodes = np.arange(n_nodes)
    else:
        sample_size = min(k, n_nodes)
        step = max(1, n_nodes // sample_size)
        sample_nodes = np.arange(0, n_nodes, step)[:sample_size]
    batch_size = min(100, sample_size)
    n_batches = (sample_size + batch_size - 1) // batch_size
    bc = np.zeros(n_nodes, dtype=np.float64)
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, sample_size)
        batch_sources = sample_nodes[start_idx:end_idx]
        bc_batch = accumulate_betweenness_parallel(adj_matrix, batch_sources, end_idx - start_idx)
        for i in prange(bc_batch.shape[0]):
            for j in range(n_nodes):
                bc[j] += bc_batch[i, j]
    if sample_size < n_nodes:
        bc *= n_nodes / sample_size
    scale = 1.0 / ((n_nodes - 1) * (n_nodes - 2)) if n_nodes > 2 else 1.0
    bc *= scale
    return bc.astype(np.float32)


@njit(parallel=True, nogil=True)
def compute_mean_synchronization_distance(adj_matrix, lat_arr, lon_arr):
    n_nodes = adj_matrix.shape[0]
    lon_size = lon_arr.shape[0]
    msd = np.zeros(n_nodes, dtype=np.float32)
    R = 6371.0
    for i in prange(n_nodes):
        lat_i_idx = i // lon_size
        lon_i_idx = i % lon_size
        lat_i = lat_arr[lat_i_idx]
        lon_i = lon_arr[lon_i_idx]
        lat_i_rad = np.radians(lat_i)
        lon_i_rad = np.radians(lon_i)
        total_weighted_distance = 0.0
        total_weight = 0.0
        for j in range(n_nodes):
            if i != j and adj_matrix[i, j] > 0:
                lat_j_idx = j // lon_size
                lon_j_idx = j % lon_size
                lat_j = lat_arr[lat_j_idx]
                lon_j = lon_arr[lon_j_idx]
                lat_j_rad = np.radians(lat_j)
                lon_j_rad = np.radians(lon_j)
                dlat = lat_j_rad - lat_i_rad
                dlon = lon_j_rad - lon_i_rad
                a = np.sin(dlat / 2) ** 2 + np.cos(lat_i_rad) * np.cos(lat_j_rad) * np.sin(dlon / 2) ** 2
                c = 2 * np.arcsin(np.sqrt(a))
                distance_km = R * c
                weight = adj_matrix[i, j]
                total_weighted_distance += distance_km * weight
                total_weight += weight
        if total_weight > 0:
            msd[i] = total_weighted_distance / total_weight
        else:
            msd[i] = 0.0
    return msd


@njit(parallel=True, nogil=True)
def compute_network_divergence(adj_qa):
    n_nodes = adj_qa.shape[0]
    in_degree = np.zeros(n_nodes, dtype=np.int32)
    out_degree = np.zeros(n_nodes, dtype=np.int32)
    for i in prange(n_nodes):
        in_deg = 0
        out_deg = 0
        for j in range(n_nodes):
            if i != j:
                if adj_qa[i, j] < 0:
                    out_deg += 1
                if adj_qa[i, j] > 0:
                    in_deg += 1
        in_degree[i] = in_deg
        out_degree[i] = out_deg
    return (in_degree, out_degree)


@njit(parallel=True, nogil=True)
def compute_network_divergence_weighted(q_matrix):
    n_nodes = q_matrix.shape[0]
    in_strength = np.zeros(n_nodes, dtype=np.float32)
    out_strength = np.zeros(n_nodes, dtype=np.float32)
    for i in prange(n_nodes):
        for j in range(n_nodes):
            if i != j:
                v = q_matrix[i, j]
                if v < 0:
                    out_strength[i] += abs(v)
                elif v > 0:
                    in_strength[i] += v
    return (in_strength, out_strength)


def save_compressed_hdf5(matrices_dict, filepath):
    with h5py.File(filepath, 'w') as f:
        total_original = 0
        for name, matrix in matrices_dict.items():
            f.create_dataset(name, data=matrix, compression='gzip', compression_opts=4, shuffle=True, fletcher32=True)
            total_original += matrix.nbytes
        file_size = os.path.getsize(filepath)
        compression_ratio = total_original / file_size
        print(f'HDF5 compression: {total_original / 1000000000.0:.2f}GB → {file_size / 1000000000.0:.2f}GB (ratio: {compression_ratio:.1f}x)')

def vectorized_rolling_sum(pre_arr: np.ndarray, scale: int) -> np.ndarray:
    valid = ~np.isnan(pre_arr)
    pre_filled = np.where(valid, pre_arr, 0.0).astype(np.float64)
    cumsum_val = np.cumsum(pre_filled, axis=0)
    roll_val = (cumsum_val[scale:] - cumsum_val[:-scale]).astype(np.float32)
    cumsum_cnt = np.cumsum(valid.astype(np.float32), axis=0)
    roll_cnt = cumsum_cnt[scale:] - cumsum_cnt[:-scale]
    roll_val = np.where(roll_cnt > 0, roll_val, np.nan)
    pad = np.full((scale, *pre_arr.shape[1:]), np.nan, dtype=np.float32)
    return np.concatenate([pad, roll_val], axis=0)



def _fit_gamma(pos):
    lm = np.log(np.mean(pos))
    ml = np.mean(np.log(pos))
    A = max(lm - ml, 1e-10)
    a = (1.0 + np.sqrt(1.0 + 4.0 * A / 3.0)) / (4.0 * A)
    b = np.mean(pos) / a
    ll = float(np.sum(st.gamma.logpdf(pos, a=a, scale=b)))
    return ((a, b), ll, 2)


def _fit_lognormal(pos):
    log_pos = np.log(pos)
    mu = float(np.mean(log_pos))
    sigma = float(np.std(log_pos, ddof=1))
    ll = float(np.sum(st.lognorm.logpdf(pos, s=sigma, scale=np.exp(mu))))
    return ((mu, sigma), ll, 2)


def _fit_pearson3(pos):
    try:
        skew_p, loc_p, scale_p = st.pearson3.fit(pos, floc=max(0.0, pos.min() - 0.001))
        with np.errstate(invalid='ignore', divide='ignore'):
            ll = float(np.sum(st.pearson3.logpdf(pos, skew=skew_p, loc=loc_p, scale=scale_p)))
        if not np.isfinite(ll):
            ll = -np.inf
    except Exception:
        skew_p, loc_p, scale_p = (0.0, 0.0, 1.0)
        ll = -np.inf
    return ((skew_p, loc_p, scale_p), ll, 3)


def _fit_weibull(pos):
    try:
        c_w, loc_w, scale_w = st.weibull_min.fit(pos, floc=0.0)
        ll = float(np.sum(st.weibull_min.logpdf(pos, c=c_w, loc=0.0, scale=scale_w)))
        if not np.isfinite(ll):
            ll = -np.inf
    except Exception:
        c_w, loc_w, scale_w = (1.0, 0.0, 1.0)
        ll = -np.inf
    return ((c_w, scale_w), ll, 2)


def _fit_gengamma(pos):
    try:
        a_gg, c_gg, loc_gg, scale_gg = st.gengamma.fit(pos, floc=0.0)
        ll = float(np.sum(st.gengamma.logpdf(pos, a=a_gg, c=c_gg, loc=0.0, scale=scale_gg)))
        if not np.isfinite(ll):
            ll = -np.inf
    except Exception:
        a_gg, c_gg, scale_gg = (1.0, 1.0, 1.0)
        ll = -np.inf
    return ((a_gg, c_gg, scale_gg), ll, 3)


def _cdf_gamma(x, params):
    a, b = params
    return st.gamma.cdf(x, a=a, scale=b)


def _cdf_lognormal(x, params):
    mu, sigma = params
    return st.lognorm.cdf(x, s=sigma, scale=np.exp(mu))


def _cdf_pearson3(x, params):
    skew_p, loc_p, scale_p = params
    return st.pearson3.cdf(x, skew=skew_p, loc=loc_p, scale=scale_p)


def _cdf_weibull(x, params):
    c_w, scale_w = params
    return st.weibull_min.cdf(x, c=c_w, loc=0.0, scale=scale_w)


def _cdf_gengamma(x, params):
    a_gg, c_gg, scale_gg = params
    return st.gengamma.cdf(x, a=a_gg, c=c_gg, loc=0.0, scale=scale_gg)


def _ppf_gamma(m_i, params):
    a, b = params
    return st.gamma.ppf(m_i, a=a, scale=b)


def _ppf_lognormal(m_i, params):
    mu, sigma = params
    return st.lognorm.ppf(m_i, s=sigma, scale=np.exp(mu))


def _ppf_pearson3(m_i, params):
    skew_p, loc_p, scale_p = params
    return st.pearson3.ppf(m_i, skew=skew_p, loc=loc_p, scale=scale_p)


def _ppf_weibull(m_i, params):
    c_w, scale_w = params
    return st.weibull_min.ppf(m_i, c=c_w, loc=0.0, scale=scale_w)


def _ppf_gengamma(m_i, params):
    a_gg, c_gg, scale_gg = params
    return st.gengamma.ppf(m_i, a=a_gg, c=c_gg, loc=0.0, scale=scale_gg)

def haversine_km(lon1, lat1, lon2, lat2):
    R = 6371.0
    phi1, phi2 = (np.radians(lat1), np.radians(lat2))
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def build_queen_weights(valid_mask_2d: np.ndarray):
    h, w = valid_mask_2d.shape
    idx_2d = np.full((h, w), -1, dtype=np.int32)
    flat = np.where(valid_mask_2d.ravel())[0]
    n = len(flat)
    for k, fi in enumerate(flat):
        idx_2d[fi // w, fi % w] = k
    rows, cols = ([], [])
    for k, fi in enumerate(flat):
        r0, c0 = (fi // w, fi % w)
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                r1, c1 = (r0 + dr, c0 + dc)
                if 0 <= r1 < h and 0 <= c1 < w and (idx_2d[r1, c1] >= 0):
                    rows.append(k)
                    cols.append(idx_2d[r1, c1])
    data_arr = np.ones(len(rows), dtype=np.float32)
    W = csr_matrix((data_arr, (rows, cols)), shape=(n, n))
    row_sums = np.array(W.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1.0
    return diags(1.0 / row_sums) @ W


def getis_ord_gi_star(x_valid, W_sparse):
    n = len(x_valid)
    xbar, s = (x_valid.mean(), x_valid.std())
    W_star = W_sparse + speye(n, format='csr')
    row_s = np.array(W_star.sum(axis=1)).ravel()
    row_s[row_s == 0] = 1.0
    W_star = diags(1.0 / row_s) @ W_star
    Wx = np.array(W_star @ x_valid).ravel()
    Ww = np.array(W_star.sum(axis=1)).ravel()
    Ww2 = np.array(W_star.power(2).sum(axis=1)).ravel()
    gi_z = (Wx - xbar * Ww) / (s * np.sqrt(np.maximum((n * Ww2 - Ww ** 2) / (n - 1), 1e-12)))
    return (gi_z, 2 * (1 - stats.norm.cdf(np.abs(gi_z))))

@nb.njit(parallel=True, cache=True)
def compute_climatology_numba(data_2d, doy_array):
    T, P = data_2d.shape
    clim = np.full((366, P), np.nan, dtype=np.float32)
    count = np.zeros((366, P), dtype=np.int32)
    sums = np.zeros((366, P), dtype=np.float64)
    for t in range(T):
        d = doy_array[t] - 1
        for p in nb.prange(P):
            v = data_2d[t, p]
            if not np.isnan(v):
                sums[d, p] += v
                count[d, p] += 1
    for d in range(366):
        for p in nb.prange(P):
            if count[d, p] > 0:
                clim[d, p] = sums[d, p] / count[d, p]
    return clim


@nb.njit(parallel=True, cache=True)
def compute_anomaly_numba(data_2d, doy_array, clim):
    T, P = data_2d.shape
    anom = np.empty_like(data_2d)
    for t in nb.prange(T):
        d = doy_array[t] - 1
        for p in range(P):
            c = clim[d, p]
            v = data_2d[t, p]
            anom[t, p] = v - c if not np.isnan(v) and (not np.isnan(c)) else np.nan
    return anom


@nb.njit(parallel=True, cache=True)
def extract_event_windows_numba(data_2d, starts, lag_min, lag_max, max_time):
    n_events = len(starts)
    win_len = lag_max - lag_min + 1
    out = np.full((n_events, win_len), np.nan, dtype=np.float32)
    for i in nb.prange(n_events):
        t0 = starts[i]
        for t in range(win_len):
            t_idx = t0 + lag_min + t
            if 0 <= t_idx < max_time:
                val_sum = 0.0
                cnt = 0
                for p in range(data_2d.shape[1]):
                    val = data_2d[t_idx, p]
                    if not np.isnan(val):
                        val_sum += val
                        cnt += 1
                if cnt > 0:
                    out[i, t] = val_sum / cnt
    return out


def get_nc_var_name(nc_obj):
    best_var, max_dims = (None, 0)
    for var in nc_obj.variables:
        vl = var.lower()
        if vl in ['lat', 'lon', 'latitude', 'longitude', 'time', 'expver', 'level', 'crs', 'spatial_ref']:
            continue
        if 'bnds' in vl or 'bounds' in vl:
            continue
        dims = len(nc_obj.variables[var].dimensions)
        if dims > max_dims:
            max_dims = dims
            best_var = var
    return best_var


def load_full_nc(nc_path):
    import netCDF4 as nc
    with nc.Dataset(nc_path, 'r') as ds:
        lat_name = 'latitude' if 'latitude' in ds.variables else 'lat'
        lon_name = 'longitude' if 'longitude' in ds.variables else 'lon'
        var_name = get_nc_var_name(ds)
        if var_name is None:
            raise ValueError(f'Could not identify a valid primary variable in {nc_path}')
        lats = ds.variables[lat_name][:]
        lons = ds.variables[lon_name][:]
        var_obj = ds.variables[var_name]
        dim_names = var_obj.dimensions
        slices = []
        for dim in dim_names:
            if dim == lat_name:
                slices.append(slice(None))
            elif dim == lon_name:
                slices.append(slice(None))
            elif dim == 'time':
                slices.append(slice(None))
            elif dim in ['expver', 'level']:
                slices.append(0)
            else:
                slices.append(slice(None))
        raw = np.asarray(var_obj[tuple(slices)], dtype=np.float32)
        if raw.ndim > 3:
            raw = np.squeeze(raw)
        if lats[0] > lats[-1]:
            lats = lats[::-1]
            raw = raw[:, ::-1, :]
        return (raw, np.asarray(lats, dtype=np.float32), np.asarray(lons, dtype=np.float32))


def extract_pixels_from_full(data_3d, full_lats, full_lons, target_lats, target_lons):
    row_idx = np.array([np.abs(full_lats - lat).argmin() for lat in target_lats])
    col_idx = np.array([np.abs(full_lons - lon).argmin() for lon in target_lons])
    return data_3d[:, row_idx, col_idx]

DIST_NAMES = ["Gamma", "LogNormal", "PearsonIII", "Weibull", "GenGamma"]
FIT_FUNCS = [_fit_gamma, _fit_lognormal, _fit_pearson3, _fit_weibull, _fit_gengamma]
CDF_FUNCS = [_cdf_gamma, _cdf_lognormal, _cdf_pearson3, _cdf_weibull, _cdf_gengamma]
PPF_FUNCS = [_ppf_gamma, _ppf_lognormal, _ppf_pearson3, _ppf_weibull, _ppf_gengamma]


def load_precipitation(path: Path):
    import xarray as xr
    ds = xr.open_dataset(path)
    names = list(ds.data_vars)
    var = next((x for x in ["precipitation", "precip", "pre", "pr", "tp"] if x in names), None)
    if var is None:
        if len(names) != 1:
            raise KeyError(f"Cannot infer precipitation variable from {names}")
        var = names[0]
    lat_name = next((x for x in ["lat", "latitude", "Latitude", "LAT"] if x in ds.variables), None)
    lon_name = next((x for x in ["lon", "longitude", "Longitude", "LON"] if x in ds.variables), None)
    time_name = next((x for x in ["time", "Time", "date"] if x in ds.variables), None)
    if not all([lat_name, lon_name, time_name]):
        raise KeyError("Could not infer time/latitude/longitude coordinates")
    data = ds[var].values.astype(np.float32)
    lat = ds[lat_name].values.astype(np.float32)
    lon = ds[lon_name].values.astype(np.float32)
    time_values = ds[time_name].values
    ds.close()
    if lat[0] > lat[-1]:
        lat = lat[::-1]
        data = data[:, ::-1, :]
    return data, lat, lon, time_values


def conservative_best_distribution(aic_values, threshold=2.0):
    aic_values = np.asarray(aic_values, dtype=float)
    if not np.any(np.isfinite(aic_values)):
        return -1
    best = int(np.nanargmin(aic_values))
    gamma_delta = aic_values[0] - np.nanmin(aic_values)
    return 0 if gamma_delta <= threshold else best


def fit_distribution_set(positive_values):
    records = []
    for fit_fn in FIT_FUNCS:
        try:
            params, loglik, k = fit_fn(positive_values)
            aic = 2 * k - 2 * loglik if np.isfinite(loglik) else np.inf
        except Exception:
            params, loglik, k, aic = (), -np.inf, 0, np.inf
        records.append((params, loglik, k, aic))
    return records


def spi_from_selected_distribution(accumulated, dist_id):
    ts = np.asarray(accumulated, dtype=float)
    valid = np.isfinite(ts)
    result = np.full(ts.shape, np.nan, dtype=np.float32)
    sample = ts[valid]
    if sample.size < 10:
        return result
    positive = sample[sample > 0]
    if positive.size < 6:
        return result
    q = float(np.sum(sample == 0)) / float(sample.size)
    params, _, _ = FIT_FUNCS[int(dist_id)](positive)
    cdf_fn = CDF_FUNCS[int(dist_id)]
    pos_mask = valid & (ts > 0)
    zero_mask = valid & (ts <= 0)
    if np.any(pos_mask):
        F = cdf_fn(ts[pos_mask], params)
        H = q + (1.0 - q) * F
        result[pos_mask] = st.norm.ppf(np.clip(H, 1e-8, 1 - 1e-8)).astype(np.float32)
    if np.any(zero_mask):
        result[zero_mask] = np.float32(st.norm.ppf(np.clip(q, 1e-8, 1 - 1e-8)))
    return result



def filter_events_by_season(evt_num_flat, evt_srs_flat, times, months, burn_in=365):
    import pandas as pd
    if months is None:
        return evt_num_flat.copy(), evt_srs_flat.copy(), len(pd.DatetimeIndex(times[burn_in:]))
    times_effective = pd.DatetimeIndex(times[burn_in:])
    season_mask = np.isin(times_effective.month, months)
    season_length = int(np.sum(season_mask))
    n_nodes, emax = evt_srs_flat.shape
    out_num = np.zeros(n_nodes, dtype=np.int32)
    out_srs = np.zeros((n_nodes, emax), dtype=np.int32)
    for node in range(n_nodes):
        n_evt = int(evt_num_flat[node])
        if n_evt <= 0:
            continue
        starts = evt_srs_flat[node, :n_evt]
        kept = starts[(starts >= 0) & (starts < len(times_effective)) & season_mask[starts]]
        out_num[node] = len(kept)
        out_srs[node, :len(kept)] = kept
    return out_num, out_srs, season_length


def build_network(evt_num, evt_srs, lat, lon, tau_max, len_time, n_surrogates=1000, quantile=95.0,
                  cache_dir=None, compute_bc=False, bc_sample_k=1000):
    bidirectional = event_sync_drought_bidirectional(evt_srs, evt_num, tau_max)
    results = compute_Q_q_from_bidir(bidirectional, evt_num)
    _, qcrit = significance_surrogate_Q_v2(evt_num, len_time, tau_max, n_surrogates=n_surrogates,
                                            quantile=quantile, cache_dir=cache_dir)
    _, dcrit = significance_surrogate_q_v2(evt_num, len_time, tau_max, n_surrogates=n_surrogates,
                                            quantile=quantile, cache_dir=cache_dir)
    Q, q = build_significant_Q_q_matrices_dual(results, qcrit, dcrit, int(evt_num.shape[0]))
    AQ = (Q > 0).astype(np.int32)
    np.fill_diagonal(AQ, 0)
    Aq = np.zeros_like(q, dtype=np.int32)
    Aq[q > 0] = 1
    Aq[q < 0] = -1
    np.fill_diagonal(Aq, 0)
    dc = compute_degree_centrality_standard(AQ.astype(np.float32))
    dc_w = compute_degree_centrality_weighted(Q.astype(np.float32))
    msd = compute_mean_synchronization_distance(AQ.astype(np.float32), lat, lon)
    msd_w = compute_mean_synchronization_distance(Q.astype(np.float32), lat, lon)
    indeg, outdeg = compute_network_divergence(Aq.astype(np.float32))
    nd = indeg - outdeg
    indeg_w, outdeg_w = compute_network_divergence_weighted(q.astype(np.float32))
    nd_w = indeg_w - outdeg_w
    bc = compute_betweenness_centrality_parallel_numba(AQ.astype(np.float32), k=bc_sample_k) if compute_bc else None
    return {"Q": Q, "q": q, "AQ": AQ, "Aq": Aq, "dc": dc, "dc_w": dc_w,
            "msd": msd, "msd_w": msd_w, "nd": nd, "nd_w": nd_w, "bc": bc}
