import argparse
import json
import os
import random
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import yaml

import data_loader
from models.moe_ode_tau import MoEODETau


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(config: Dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def map_roi_columns(tau_df: pd.DataFrame, roi_list: List[str], suffix: str) -> List[str]:
    columns = [c for c in tau_df.columns if c.endswith(suffix)]
    mapping = {}
    for col in columns:
        base = col[: -len(suffix)]
        key = data_loader.normalize_roi_name(base)
        mapping.setdefault(key, []).append(col)
    ordered = []
    missing = {}
    for roi in roi_list:
        key = data_loader.normalize_roi_name(roi)
        cols = mapping.get(key, [])
        if len(cols) == 1:
            ordered.append(cols[0])
        else:
            missing[roi] = cols
    if missing:
        raise ValueError(f"ROI alignment failed for suffix {suffix}: {len(missing)} missing.")
    return ordered


def get_subject_splits(
    subject_ids: List[str], seed: int, train_frac: float, val_frac: float, test_frac: float
) -> Dict[str, List[str]]:
    if not np.isclose(train_frac + val_frac + test_frac, 1.0):
        raise ValueError("Split fractions must sum to 1.0.")
    rng = np.random.default_rng(seed)
    ids = np.array(subject_ids)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    n_train = min(n_train, n)
    n_val = min(n_val, n - n_train)
    n_test = n - n_train - n_val
    return {
        "train": ids[:n_train].tolist(),
        "val": ids[n_train : n_train + n_val].tolist(),
        "test": ids[n_train + n_val : n_train + n_val + n_test].tolist(),
    }


def parse_tau_visits(
    tau_df: pd.DataFrame,
    roi_cols: List[str],
    subject_col: str,
) -> Dict[str, Dict[str, np.ndarray]]:
    tau_df = tau_df.copy()
    tau_df["t_month"] = tau_df["VISCODE2"].apply(data_loader.parse_viscode2)
    tau_df = tau_df.dropna(subset=["t_month"])
    before_rows = len(tau_df)
    tau_df = tau_df.dropna(subset=roi_cols)
    dropped = before_rows - len(tau_df)
    if dropped > 0:
        print(f"Dropped {dropped} rows with NaN in ROI columns.")

    subject_data = {}
    for subject_id, sdf in tau_df.groupby(subject_col):
        sdf = sdf.sort_values("t_month")
        baseline = sdf.iloc[0]
        sdf = sdf[sdf["t_month"] >= baseline["t_month"]]
        if sdf["t_month"].nunique() < 2:
            continue
        times = sdf["t_month"].to_numpy(dtype=float)
        obs = sdf[roi_cols].to_numpy(dtype=float)
        if times[0] != baseline["t_month"]:
            times = np.insert(times, 0, baseline["t_month"])
            obs = np.vstack([baseline[roi_cols].to_numpy(dtype=float), obs])
        times = times - baseline["t_month"]
        subject_data[str(subject_id)] = {
            "times": times,
            "obs": obs,
        }
    return subject_data


def compute_winsorize_bounds(train_arrays: List[np.ndarray], p1: float, p99: float) -> Tuple[float, float]:
    flat = np.concatenate([arr.reshape(-1) for arr in train_arrays], axis=0)
    low = np.nanpercentile(flat, p1)
    high = np.nanpercentile(flat, p99)
    return float(low), float(high)


def apply_winsorize(arr: np.ndarray, low: float, high: float) -> np.ndarray:
    return np.clip(arr, low, high)


def compute_minmax(train_arrays: List[np.ndarray]) -> Tuple[float, float]:
    flat = np.concatenate([arr.reshape(-1) for arr in train_arrays], axis=0)
    return float(np.nanmin(flat)), float(np.nanmax(flat))


def transform_values(arr: np.ndarray, transform: str, minmax: Tuple[float, float]) -> np.ndarray:
    if transform == "log1p":
        return np.log1p(arr)
    if transform == "minmax":
        min_v, max_v = minmax
        denom = max(max_v - min_v, 1e-8)
        return (arr - min_v) / denom
    if transform == "none":
        return arr
    raise ValueError(f"Unknown transform: {transform}")


def scale_times(subjects: Dict[str, Dict[str, np.ndarray]], time_unit: str) -> Dict[str, Dict[str, np.ndarray]]:
    if time_unit == "month":
        return subjects
    if time_unit != "year":
        raise ValueError(f"Unknown time_unit: {time_unit}")
    scaled = {}
    for key, subj in subjects.items():
        scaled[key] = {
            "times": subj["times"] / 12.0,
            "obs": subj["obs"],
        }
    return scaled


def sample_global(T_grid: torch.Tensor, t_query: torch.Tensor, grid_min: float, dt: float) -> torch.Tensor:
    t_query = torch.nan_to_num(t_query, nan=grid_min, posinf=grid_min, neginf=grid_min)
    idx_float = (t_query - grid_min) / dt
    idx0 = torch.floor(idx_float)
    idx0 = torch.clamp(idx0, 0, T_grid.shape[0] - 2)
    idx0_long = idx0.to(dtype=torch.long)
    frac = idx_float - idx0
    t0 = T_grid[idx0_long]
    t1 = T_grid[idx0_long + 1]
    return t0 * (1.0 - frac.unsqueeze(-1)) + t1 * frac.unsqueeze(-1)


def sample_global_numpy(T_grid: np.ndarray, t_query: np.ndarray, grid_min: float, dt: float) -> np.ndarray:
    t_query = np.nan_to_num(t_query, nan=grid_min, posinf=grid_min, neginf=grid_min)
    idx_float = (t_query - grid_min) / dt
    idx0 = np.floor(idx_float)
    idx0 = np.clip(idx0, 0, T_grid.shape[0] - 2)
    frac = idx_float - idx0
    idx0 = idx0.astype(int)
    t0 = T_grid[idx0]
    t1 = T_grid[idx0 + 1]
    return t0 * (1.0 - frac[..., None]) + t1 * frac[..., None]


def _alpha_candidates(alpha_min: float, alpha_max: float, n_alpha: int) -> np.ndarray:
    if n_alpha <= 1 or alpha_min >= alpha_max:
        return np.array([alpha_min], dtype=float)
    return np.linspace(alpha_min, alpha_max, n_alpha, dtype=float)


def _alpha_refine_bounds(alpha: float, alpha_min: float, alpha_max: float, width: float) -> Tuple[float, float]:
    lo = max(alpha_min, alpha - width)
    hi = min(alpha_max, alpha + width)
    if hi <= lo:
        return alpha_min, alpha_max
    return lo, hi


def align_subjects_grid_scale(
    subjects: Dict[str, Dict[str, np.ndarray]],
    t_grid: np.ndarray,
    traj: np.ndarray,
    grid_min: float,
    grid_max: float,
    dt: float,
    grid_size: int,
    alpha_min: float,
    alpha_max: float,
    n_alpha: int,
    lambda_alpha: float,
    refine_steps: int,
    refine_width: float,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    t0_map: Dict[str, float] = {}
    alpha_map: Dict[str, float] = {}
    alpha_candidates = _alpha_candidates(alpha_min, alpha_max, n_alpha)
    for sid, subj in subjects.items():
        times = subj["times"]
        obs = subj["obs"]
        best_t0 = 0.0
        best_alpha = 1.0
        best_sse = np.inf
        for alpha in alpha_candidates:
            t0_min = grid_min - float(np.max(times)) * alpha
            t0_max = grid_max - float(np.min(times)) * alpha
            if t0_max <= t0_min:
                candidates = np.array([t0_min], dtype=float)
            else:
                candidates = np.linspace(t0_min, t0_max, grid_size, dtype=float)
            for t0 in candidates:
                t_query = t0 + alpha * times
                if np.any(t_query < grid_min) or np.any(t_query > grid_max):
                    continue
                pred = sample_global_numpy(traj, t_query, grid_min, dt)
                diff = pred - obs
                sse = float(np.sum(diff * diff)) + lambda_alpha * float((alpha - 1.0) ** 2)
                if sse < best_sse:
                    best_sse = sse
                    best_t0 = float(t0)
                    best_alpha = float(alpha)
        if refine_steps > 1 and alpha_max > alpha_min:
            r_lo, r_hi = _alpha_refine_bounds(best_alpha, alpha_min, alpha_max, refine_width)
            refine_candidates = _alpha_candidates(r_lo, r_hi, refine_steps)
            for alpha in refine_candidates:
                t0_min = grid_min - float(np.max(times)) * alpha
                t0_max = grid_max - float(np.min(times)) * alpha
                if t0_max <= t0_min:
                    candidates = np.array([t0_min], dtype=float)
                else:
                    candidates = np.linspace(t0_min, t0_max, grid_size, dtype=float)
                for t0 in candidates:
                    t_query = t0 + alpha * times
                    if np.any(t_query < grid_min) or np.any(t_query > grid_max):
                        continue
                    pred = sample_global_numpy(traj, t_query, grid_min, dt)
                    diff = pred - obs
                    sse = float(np.sum(diff * diff)) + lambda_alpha * float((alpha - 1.0) ** 2)
                    if sse < best_sse:
                        best_sse = sse
                        best_t0 = float(t0)
                        best_alpha = float(alpha)
        t0_map[sid] = best_t0
        alpha_map[sid] = best_alpha
    return t0_map, alpha_map


def compute_traj_loss(
    subjects: Dict[str, Dict[str, np.ndarray]],
    t0_map: Dict[str, float],
    alpha_map: Dict[str, float] | None,
    T_grid: torch.Tensor,
    grid_min: float,
    dt: float,
    roi_cap: float,
    ignore_cap_first_n: int,
) -> Tuple[torch.Tensor, int]:
    total_sse = torch.tensor(0.0, dtype=torch.float32)
    total_count = 0
    for sid, subj in subjects.items():
        times = subj["times_t"]
        obs = subj["obs_t"]
        visit_idx = torch.arange(times.shape[0], device=times.device)
        t0 = t0_map[sid]
        alpha = alpha_map[sid] if alpha_map is not None else 1.0
        t_query = t0 + alpha * times
        t_max = grid_min + dt * (T_grid.shape[0] - 1)
        valid = (t_query >= grid_min) & (t_query <= t_max)
        if not torch.any(valid):
            continue
        t_query = t_query[valid]
        obs = obs[valid]
        visit_idx = visit_idx[valid]
        pred = sample_global(T_grid, t_query, grid_min, dt)
        diff = pred - obs
        if ignore_cap_first_n > 0:
            early = visit_idx < ignore_cap_first_n
            mask = torch.ones_like(obs, dtype=torch.bool)
            if torch.any(early):
                mask[early] = obs[early] < roi_cap
        else:
            mask = obs < roi_cap
        if torch.any(mask):
            diff = diff[mask]
            total_sse = total_sse + torch.sum(diff * diff)
            total_count += diff.numel()
    return total_sse, total_count


def eval_metrics(
    subjects: Dict[str, Dict[str, np.ndarray]],
    t0_map: Dict[str, float],
    alpha_map: Dict[str, float] | None,
    traj: np.ndarray,
    grid_min: float,
    dt: float,
    holdout_last: bool,
) -> Dict[str, float]:
    sse_total = 0.0
    r_vals = []
    sse_over_visits = []
    for sid, subj in subjects.items():
        times = subj["times"]
        obs = subj["obs"]
        t0 = t0_map[sid]
        if holdout_last:
            times = times[-1:]
            obs = obs[-1:]
        alpha = alpha_map[sid] if alpha_map is not None else 1.0
        t_query = t0 + alpha * times
        pred = sample_global_numpy(traj, t_query, grid_min, dt)
        for idx in range(pred.shape[0]):
            diff = pred[idx] - obs[idx]
            sse = float(np.sum(diff * diff))
            sse_total += sse
            sse_over_visits.append(sse)
            r_val = np.corrcoef(pred[idx], obs[idx])[0, 1]
            if not np.isnan(r_val):
                r_vals.append(r_val)
    return {
        "SSE_total": float(sse_total),
        "R_mean": float(np.mean(r_vals)) if r_vals else float("nan"),
        "SSE_mean_over_visits": float(np.mean(sse_over_visits)) if sse_over_visits else float("nan"),
    }


def plot_loss_curve(loss_history: List[Tuple[int, int, float, float]], path: str) -> None:
    import matplotlib.pyplot as plt

    steps = list(range(1, len(loss_history) + 1))
    train = [x[2] for x in loss_history]
    val = [x[3] for x in loss_history]
    plt.figure(figsize=(6, 4))
    plt.plot(steps, train, label="train")
    plt.plot(steps, val, label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def alpha_stats(alpha_map: Dict[str, float], alpha_min: float, alpha_max: float, t0_map: Dict[str, float]) -> Dict[str, float]:
    vals = np.array(list(alpha_map.values()), dtype=float)
    if vals.size == 0:
        return {
            "alpha_mean": float("nan"),
            "alpha_std": float("nan"),
            "alpha_p10": float("nan"),
            "alpha_p50": float("nan"),
            "alpha_p90": float("nan"),
            "alpha_on_min_ratio": 0.0,
            "alpha_on_max_ratio": 0.0,
            "alpha_t0_corr": float("nan"),
        }
    t0_vals = np.array([t0_map[k] for k in alpha_map.keys()], dtype=float)
    corr = float(np.corrcoef(vals, t0_vals)[0, 1]) if vals.size > 1 else float("nan")
    return {
        "alpha_mean": float(np.mean(vals)),
        "alpha_std": float(np.std(vals)),
        "alpha_p10": float(np.percentile(vals, 10.0)),
        "alpha_p50": float(np.percentile(vals, 50.0)),
        "alpha_p90": float(np.percentile(vals, 90.0)),
        "alpha_on_min_ratio": float(np.mean(vals <= alpha_min + 1e-6)),
        "alpha_on_max_ratio": float(np.mean(vals >= alpha_max - 1e-6)),
        "alpha_t0_corr": corr,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="MoE-ODE PET model (physical diffusion+reaction + local clearance).")
    parser.add_argument("--config", type=str, default="configs/ours_tau.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["split"]["seed"])

    artifacts_dir = config["data"]["artifacts_dir"]
    os.makedirs(artifacts_dir, exist_ok=True)

    device_cfg = str(config.get("training", {}).get("device", "auto")).lower()
    if device_cfg == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
        print("[WARN] CUDA requested but not available; falling back to CPU.")
    elif device_cfg in {"cuda", "cpu"}:
        device = torch.device(device_cfg)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    roi_subset = config.get("data", {}).get("roi_subset", None)
    subset_after_laplacian = bool(config.get("data", {}).get("roi_subset_after_laplacian", False))
    A, roi_list, _, L, L_norm, L_rw = data_loader.prepare_sc(
        config["data"]["sc_path"],
        roi_subset=roi_subset,
        subset_after_laplacian=subset_after_laplacian,
    )
    np.save(os.path.join(artifacts_dir, "sc_matrix.npy"), A)
    np.save(os.path.join(artifacts_dir, "laplacian.npy"), L)
    np.save(os.path.join(artifacts_dir, "laplacian_norm.npy"), L_norm)

    tau_df = data_loader.load_tau(config["data"]["tau_path"])
    roi_cols = map_roi_columns(tau_df, roi_list, "_SUVR")
    sinf_clip_cfg = config.get("preprocess", {}).get("roi_sinf_clip", {})
    if bool(sinf_clip_cfg.get("enabled", False)):
        params_path = str(sinf_clip_cfg.get("path", "roi_gmm_parameters.csv"))
        eps_clip = float(sinf_clip_cfg.get("eps_clip", 1e-6))
        _, s_inf_vec_clip = data_loader.build_roi_gmm_vectors(roi_list, params_path)
        tau_df = data_loader.apply_roi_sinf_clip(tau_df, roi_cols, s_inf_vec_clip, eps_clip)
    roi_norm_cfg = config.get("preprocess", {}).get("roi_gmm_norm", {})
    if bool(roi_norm_cfg.get("enabled", False)):
        params_path = str(roi_norm_cfg.get("path", "roi_gmm_parameters.csv"))
        eps_num = float(roi_norm_cfg.get("eps_num", 1e-6))
        eps_den = float(roi_norm_cfg.get("eps_den", 1e-6))
        eps_clip = float(roi_norm_cfg.get("eps_clip", 1e-4))
        s0_vec, s_inf_vec = data_loader.build_roi_gmm_vectors(roi_list, params_path)
        tau_df = data_loader.apply_roi_gmm_normalization(
            tau_df, roi_cols, s0_vec, s_inf_vec, eps_num, eps_den, eps_clip
        )
    subject_col = data_loader.subject_id_column(tau_df)
    subjects = parse_tau_visits(tau_df, roi_cols, subject_col)

    subject_ids = list(subjects.keys())
    splits = get_subject_splits(
        subject_ids,
        config["split"]["seed"],
        config["split"]["train_frac"],
        config["split"]["val_frac"],
        config["split"]["test_frac"],
    )

    train_subjects = {k: subjects[k] for k in splits["train"]}
    val_subjects = {k: subjects[k] for k in splits["val"]}
    test_subjects = {k: subjects[k] for k in splits["test"]}

    train_arrays = [subj["obs"] for subj in train_subjects.values()]
    winsorize = config["preprocess"]["winsorize"]
    if winsorize:
        low, high = compute_winsorize_bounds(
            train_arrays,
            config["preprocess"]["winsorize_p1"],
            config["preprocess"]["winsorize_p99"],
        )
    else:
        low, high = -np.inf, np.inf
    if config["preprocess"]["transform"] == "minmax":
        min_v, max_v = compute_minmax([apply_winsorize(arr, low, high) for arr in train_arrays])
    else:
        min_v, max_v = 0.0, 0.0

    def apply_preprocess_to_subjects(subject_dict: Dict[str, Dict[str, np.ndarray]]) -> None:
        for subj in subject_dict.values():
            obs = subj["obs"]
            if winsorize:
                obs = apply_winsorize(obs, low, high)
            obs = transform_values(obs, config["preprocess"]["transform"], (min_v, max_v))
            subj["obs"] = obs

    apply_preprocess_to_subjects(train_subjects)
    apply_preprocess_to_subjects(val_subjects)
    apply_preprocess_to_subjects(test_subjects)

    train_subjects = scale_times(train_subjects, config["model"]["time_unit"])
    val_subjects = scale_times(val_subjects, config["model"]["time_unit"])
    test_subjects = scale_times(test_subjects, config["model"]["time_unit"])

    for subj in {**train_subjects, **val_subjects, **test_subjects}.values():
        subj["times_t"] = torch.tensor(subj["times"], dtype=torch.float32, device=device)
        subj["obs_t"] = torch.tensor(subj["obs"], dtype=torch.float32, device=device)

    lap_mode = str(config["model"].get("laplacian", "norm")).lower()
    if lap_mode == "random_walk":
        lap_mode = "rw"
    L_used = data_loader.select_laplacian(A, lap_mode, L=L, L_norm=L_norm, L_rw=L_rw)
    phys_dmod = bool(config.get("model", {}).get("phys_dmod", False))
    phys_d_hidden = int(config.get("phys_dmod", {}).get("hidden", 32))
    phys_d_eps = float(config.get("phys_dmod", {}).get("eps", 1e-3))
    local_mode = str(config.get("local", {}).get("mode", "per_roi"))
    local_global_hidden = int(config.get("local", {}).get("global_hidden", 64))
    local_global_layers = int(config.get("local", {}).get("global_layers", 2))
    local_layers = int(config.get("local", {}).get("layers", 2))
    disable_gate = bool(config.get("model", {}).get("disable_gate", False))
    fixed_beta_cfg = config.get("model", {}).get("fixed_beta", None)
    fixed_beta_normalize = bool(config.get("model", {}).get("fixed_beta_normalize", True))
    if fixed_beta_cfg is not None and not isinstance(fixed_beta_cfg, (list, tuple)):
        raise ValueError("model.fixed_beta must be null or a list like [beta_phys, beta_local].")
    fixed_beta = [float(x) for x in fixed_beta_cfg] if fixed_beta_cfg is not None else None

    model = MoEODETau(
        n_roi=L_used.shape[0],
        L_sc=torch.tensor(L_used, dtype=torch.float32, device=device),
        reaction=config["model"]["reaction"],
        gate_use_mean=bool(config["model"].get("gate_use_mean", True)),
        enable_local=bool(config["model"].get("enable_local", True)),
        phys_dmod=phys_dmod,
        local_hidden=int(config["local"].get("hidden", 32)),
        local_use_rho=bool(config["local"].get("use_rho", False)),
        local_rho_max=float(config["local"].get("rho_max", 0.1)),
        local_mode=local_mode,
        local_global_hidden=local_global_hidden,
        local_global_layers=local_global_layers,
        local_layers=local_layers,
        phys_d_hidden=phys_d_hidden,
        phys_d_eps=phys_d_eps,
        disable_gate=disable_gate,
        fixed_beta=fixed_beta,
        fixed_beta_normalize=fixed_beta_normalize,
        force_k_zero=bool(config.get("model", {}).get("force_k_zero", False)),
        force_r_zero=bool(config.get("model", {}).get("force_r_zero", False)),
    )
    model.to(device)

    c0_raw = torch.nn.Parameter(torch.zeros((L_used.shape[0],), dtype=torch.float32, device=device))
    params = list(model.parameters()) + [c0_raw]
    optimizer = torch.optim.Adam(params, lr=float(config["training"]["lr"]))

    all_times = np.concatenate([subj["times"] for subj in {**train_subjects, **val_subjects, **test_subjects}.values()])
    min_time = float(np.min(all_times)) if all_times.size else 0.0
    max_time = float(np.max(all_times)) if all_times.size else 1.0
    pad = float(config["model"].get("global_time_pad", 5.0))
    grid_min = float(config["model"].get("global_time_min", min_time - pad))
    grid_max = float(config["model"].get("global_time_max", max_time + pad))
    dt = float(config["model"]["dt"])
    num_steps = int(np.floor((grid_max - grid_min) / dt)) + 1
    t_grid = grid_min + dt * torch.arange(num_steps, dtype=torch.float32, device=device)
    t_grid_np = t_grid.detach().cpu().numpy()

    alignment_iters = int(config["training"].get("alignment_iters", 5))
    epochs_per_iter = int(config["training"].get("epochs_per_iter", 50))
    t0_grid_size = int(config["training"].get("t0_grid_size", 41))
    loss_mode = str(config["training"].get("loss_mode", "mse")).lower()
    if loss_mode not in {"sse", "mse"}:
        raise ValueError(f"Unknown loss_mode: {loss_mode}")
    data_loss_weight = float(config["training"].get("data_loss_weight", 1.0))
    roi_cap = float(config["training"].get("roi_cap", 0.98))
    ignore_cap_first_n = int(config["training"].get("ignore_cap_first_n", 0))
    k_max = config["training"].get("k_max", None)
    if k_max is not None:
        k_max = float(k_max)

    lambda_local = float(config["training"].get("lambda_local", 1e-3))
    lambda_mono = float(config["training"].get("lambda_mono", 1e-2))
    lambda_ent = float(config["training"].get("lambda_ent", 1e-3))
    lambda_d_phys = float(config["training"].get("lambda_d_phys", 1e-2))
    lambda_d_phys_smooth = float(config["training"].get("lambda_d_phys_smooth", 1e-2))
    mono_mode = str(config["training"].get("mono_mode", "roi"))
    mono_eps = float(config["training"].get("mono_eps", 0.0))
    alpha_enabled = bool(config["training"].get("alpha_enabled", False))
    alpha_warmup_iters = int(config["training"].get("alpha_warmup_iters", 0))
    alpha_stage1_iters = int(config["training"].get("alpha_stage1_iters", 0))
    alpha_stage1_min = float(config["training"].get("alpha_stage1_min", 0.8))
    alpha_stage1_max = float(config["training"].get("alpha_stage1_max", 1.2))
    alpha_stage2_min = float(config["training"].get("alpha_stage2_min", 0.5))
    alpha_stage2_max = float(config["training"].get("alpha_stage2_max", 2.0))
    alpha_grid_size = int(config["training"].get("alpha_grid_size", 9))
    alpha_refine_steps = int(config["training"].get("alpha_refine_steps", 5))
    alpha_refine_width = float(config["training"].get("alpha_refine_width", 0.1))
    lambda_alpha = float(config["training"].get("lambda_alpha", 1e-3))

    safety_clip = bool(config["training"].get("safety_clip", True))
    safety_min = float(config["training"].get("safety_min", -0.1))
    safety_max = float(config["training"].get("safety_max", 1.1))

    data_tag = "AMYLOID" if "amyloid" in str(config["data"]["tau_path"]).lower() else "TAU"
    run_dir = os.path.join(
        config["data"]["outputs_dir"],
        f"run_{time.strftime('%Y%m%d_%H%M%S')}_FULL_PHYS_LOCAL_CLEARANCE_DMOD_{data_tag}",
    )
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "train_log.txt")

    def log_line(msg: str) -> None:
        print(msg)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    if fixed_beta is not None:
        log_line(f"[gate] fixed_beta enabled: {fixed_beta}, normalize={fixed_beta_normalize}")
    elif disable_gate:
        log_line("[gate] disabled: using equal weights over active experts")

    loss_history = []
    diagnostics = []
    for align_iter in range(alignment_iters):
        with torch.no_grad():
            c0 = torch.sigmoid(c0_raw)
            outputs = model.integrate(c0, t_grid, dt, k_max=k_max, safety_clip=safety_clip, safety_min=safety_min, safety_max=safety_max)
        traj_np = outputs.traj.detach().cpu().numpy()

        if not alpha_enabled or align_iter < alpha_warmup_iters:
            alpha_min = alpha_max = 1.0
            n_alpha = 1
        elif align_iter < alpha_warmup_iters + alpha_stage1_iters:
            alpha_min, alpha_max = alpha_stage1_min, alpha_stage1_max
            n_alpha = alpha_grid_size
        else:
            alpha_min, alpha_max = alpha_stage2_min, alpha_stage2_max
            n_alpha = alpha_grid_size

        t0_map_train, alpha_map_train = align_subjects_grid_scale(
            train_subjects,
            t_grid_np,
            traj_np,
            grid_min,
            grid_max,
            dt,
            t0_grid_size,
            alpha_min,
            alpha_max,
            n_alpha,
            lambda_alpha,
            alpha_refine_steps,
            alpha_refine_width,
        )
        t0_map_val, alpha_map_val = align_subjects_grid_scale(
            val_subjects,
            t_grid_np,
            traj_np,
            grid_min,
            grid_max,
            dt,
            t0_grid_size,
            alpha_min,
            alpha_max,
            n_alpha,
            lambda_alpha,
            alpha_refine_steps,
            alpha_refine_width,
        )

        for epoch in range(epochs_per_iter):
            optimizer.zero_grad()
            c0 = torch.sigmoid(c0_raw)

            outputs = model.integrate(
                c0, t_grid, dt, k_max=k_max, safety_clip=safety_clip, safety_min=safety_min, safety_max=safety_max
            )
            sse_train, count_train = compute_traj_loss(
                train_subjects, t0_map_train, alpha_map_train, outputs.traj, grid_min, dt, roi_cap, ignore_cap_first_n
            )
            l_local = torch.mean(outputs.f_local * outputs.f_local)
            mono_mask = outputs.traj[1:] < roi_cap
            if mono_mode == "mean":
                mono_val = torch.relu(mono_eps - torch.mean(outputs.dT_total[mono_mask])) if torch.any(mono_mask) else torch.tensor(0.0)
            else:
                mono_val = torch.mean(torch.relu(mono_eps - outputs.dT_total[mono_mask])) if torch.any(mono_mask) else torch.tensor(0.0)
            beta = outputs.beta
            ent = torch.mean(torch.sum(beta * torch.log(beta + 1e-8), dim=-1)) if beta.numel() else torch.tensor(0.0)

            denom_train = max(count_train, 1)
            data_loss = sse_train if loss_mode == "sse" else sse_train / denom_train
            loss = (
                data_loss_weight * data_loss
                + lambda_local * l_local
                + lambda_mono * mono_val
                - lambda_ent * ent
            )
            if phys_dmod:
                loss = loss + lambda_d_phys * outputs.d_reg + lambda_d_phys_smooth * outputs.d_smooth

            loss.backward()
            grad_clip = config["training"].get("grad_clip", None)
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(params, float(grad_clip))
            optimizer.step()

            with torch.no_grad():
                val_sse, val_count = compute_traj_loss(
                    val_subjects, t0_map_val, alpha_map_val, outputs.traj, grid_min, dt, roi_cap, ignore_cap_first_n
                )
                denom_val = max(val_count, 1)
                val_data_loss = val_sse if loss_mode == "sse" else val_sse / denom_val
                val_loss = (
                    data_loss_weight * val_data_loss
                    + lambda_local * l_local
                    + lambda_mono * mono_val
                    - lambda_ent * ent
                )
                if phys_dmod:
                    val_loss = val_loss + lambda_d_phys * outputs.d_reg + lambda_d_phys_smooth * outputs.d_smooth
            loss_history.append((align_iter, epoch, float(loss.item()), float(val_loss.item())))

        with torch.no_grad():
            abs_phys = float(torch.mean(torch.abs(outputs.f_phys)).item())
            abs_local = float(torch.mean(torch.abs(outputs.f_local)).item())
            beta_mean = torch.mean(outputs.beta, dim=0) if outputs.beta.numel() else torch.zeros(2)
            mono_mask_ref = outputs.traj[1:] < roi_cap
            if torch.any(mono_mask_ref):
                mono_violation = float(torch.mean((outputs.dT_total[mono_mask_ref] < 0).to(outputs.dT_total.dtype)).item())
            else:
                mono_violation = 0.0
            param_stats = model.extra_state(k_max)
            gamma_mean = float(torch.mean(outputs.gamma).item()) if outputs.gamma.numel() else float("nan")
            gamma_p90 = float(torch.quantile(outputs.gamma.reshape(-1), 0.9).item()) if outputs.gamma.numel() else float("nan")
            d_mean = float(outputs.d_mean.item()) if phys_dmod else float("nan")
            d_reg = float(outputs.d_reg.item()) if phys_dmod else float("nan")
            d_smooth = float(outputs.d_smooth.item()) if phys_dmod else float("nan")

        alpha_summary = alpha_stats(alpha_map_train, alpha_min, alpha_max, t0_map_train)
        val_sse_noalpha = None
        if alpha_enabled and alpha_min != alpha_max:
            with torch.no_grad():
                val_sse_noalpha, _ = compute_traj_loss(
                    val_subjects, t0_map_val, None, outputs.traj, grid_min, dt, roi_cap, ignore_cap_first_n
                )
        log_line(
            "[align {}/{}] train={:.4f} val={:.4f} | "
            "beta M/L={:.2f}/{:.2f} | "
            "abs f M/L={:.3f}/{:.3f} | mono_violation={:.2%}".format(
                align_iter + 1,
                alignment_iters,
                loss.item(),
                val_loss.item(),
                beta_mean[0].item(),
                beta_mean[1].item(),
                abs_phys,
                abs_local,
                mono_violation,
            )
        )
        if phys_dmod:
            log_line("  d stats: mean={:.3f} reg={:.4f} smooth={:.4f}".format(d_mean, d_reg, d_smooth))
        if alpha_enabled:
            if val_sse_noalpha is None:
                log_line(
                    "  alpha stats: mean={:.3f} std={:.3f} p10/p50/p90={:.3f}/{:.3f}/{:.3f} | "
                    "on_min={:.1%} on_max={:.1%} | corr(t0,alpha)={:.3f}".format(
                        alpha_summary["alpha_mean"],
                        alpha_summary["alpha_std"],
                        alpha_summary["alpha_p10"],
                        alpha_summary["alpha_p50"],
                        alpha_summary["alpha_p90"],
                        alpha_summary["alpha_on_min_ratio"],
                        alpha_summary["alpha_on_max_ratio"],
                        alpha_summary["alpha_t0_corr"],
                    )
                )
            else:
                log_line(
                    "  alpha stats: mean={:.3f} std={:.3f} p10/p50/p90={:.3f}/{:.3f}/{:.3f} | "
                    "on_min={:.1%} on_max={:.1%} | corr(t0,alpha)={:.3f} | "
                    "val_sse delta (alpha-1)={:.4f}".format(
                        alpha_summary["alpha_mean"],
                        alpha_summary["alpha_std"],
                        alpha_summary["alpha_p10"],
                        alpha_summary["alpha_p50"],
                        alpha_summary["alpha_p90"],
                        alpha_summary["alpha_on_min_ratio"],
                        alpha_summary["alpha_on_max_ratio"],
                        alpha_summary["alpha_t0_corr"],
                        float((val_sse.item() - val_sse_noalpha.item())),
                    )
                )
        log_line(
            "  loss_terms: data={:.4f} w={:.2f} L_local={:.4f} "
            "L_mono={:.4f} L_ent={:.4f}".format(
                data_loss.item(),
                data_loss_weight,
                l_local.item(),
                mono_val.item(),
                ent.item(),
            )
        )
        if phys_dmod:
            log_line("  loss_terms_d: L_d={:.4f} L_d_smooth={:.4f}".format(d_reg, d_smooth))
        diagnostics.append(
            {
                "align_iter": int(align_iter + 1),
                "loss": float(loss.item()),
                "val_loss": float(val_loss.item()),
                "abs_f_phys": abs_phys,
                "abs_f_local": abs_local,
                "beta_mean": [float(x) for x in beta_mean.detach().cpu().numpy().tolist()],
                "mono_violation_ratio": mono_violation,
                "k": param_stats["k"],
                "r": param_stats["r"],
                "gamma_mean": gamma_mean,
                "gamma_p90": gamma_p90,
                "d_mean": d_mean if phys_dmod else None,
                "d_reg": d_reg if phys_dmod else None,
                "d_smooth": d_smooth if phys_dmod else None,
                "alpha_stats": alpha_summary if alpha_enabled else None,
                "val_sse_noalpha": float(val_sse_noalpha.item()) if val_sse_noalpha is not None else None,
            }
        )

    save_config(config, os.path.join(run_dir, "config.yaml"))
    with open(os.path.join(run_dir, "split_subjects.json"), "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)
    with open(os.path.join(run_dir, "loss_history.csv"), "w", encoding="utf-8") as f:
        f.write("align_iter,epoch,train_loss,val_loss\n")
        for align_iter, epoch, train_loss, val_loss in loss_history:
            f.write(f"{align_iter},{epoch},{train_loss:.6f},{val_loss:.6f}\n")
    if diagnostics:
        with open(os.path.join(run_dir, "diagnostics.json"), "w", encoding="utf-8") as f:
            json.dump(diagnostics, f, indent=2)
    if loss_history:
        plot_loss_curve(loss_history, os.path.join(run_dir, "loss_curve.png"))

    with torch.no_grad():
        c0 = torch.sigmoid(c0_raw)
        outputs = model.integrate(c0, t_grid, dt, k_max=k_max, safety_clip=safety_clip, safety_min=safety_min, safety_max=safety_max)
    traj_np = outputs.traj.detach().cpu().numpy()
    t0_map_test, alpha_map_test = align_subjects_grid_scale(
        test_subjects,
        t_grid_np,
        traj_np,
        grid_min,
        grid_max,
        dt,
        t0_grid_size,
        alpha_stage2_min if alpha_enabled else 1.0,
        alpha_stage2_max if alpha_enabled else 1.0,
        alpha_grid_size if alpha_enabled else 1,
        lambda_alpha,
        alpha_refine_steps,
        alpha_refine_width,
    )

    metrics_test = eval_metrics(test_subjects, t0_map_test, alpha_map_test, traj_np, grid_min, dt, holdout_last=False)
    metrics_holdout = eval_metrics(test_subjects, t0_map_test, alpha_map_test, traj_np, grid_min, dt, holdout_last=True)
    summary = {
        "test_SSE_total": metrics_test["SSE_total"],
        "test_R_mean": metrics_test["R_mean"],
        "test_SSE_mean_over_visits": metrics_test["SSE_mean_over_visits"],
        "holdout_SSE_total": metrics_holdout["SSE_total"],
        "holdout_R_mean": metrics_holdout["R_mean"],
        "holdout_SSE_mean_over_visits": metrics_holdout["SSE_mean_over_visits"],
    }
    with open(os.path.join(run_dir, "metrics_moe_tau_full.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "c0_raw": c0_raw.detach().cpu(),
            "reaction": config["model"]["reaction"],
            "enable_local": model.enable_local,
        },
        os.path.join(run_dir, "model_state.pt"),
    )
    log_line(f"Outputs saved to {run_dir}")


if __name__ == "__main__":
    main()
