import argparse
import json
import os
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def normalize_roi_name(name: str) -> str:
    if name is None:
        return ""
    name = str(name).strip().lower()
    name = name.replace("-", "").replace("_", "").replace(" ", "")
    return name


def load_roi_gmm_params(params_path: str) -> pd.DataFrame:
    params = pd.read_csv(params_path)
    if "ROI" not in params.columns:
        raise ValueError("roi_gmm_parameters.csv missing ROI column.")
    params = params.copy()
    params["ROI_key"] = params["ROI"].apply(
        lambda x: normalize_roi_name(str(x).replace("_SUVR", ""))
    )
    return params.set_index("ROI_key")


def build_roi_gmm_vectors(roi_list: List[str], params_path: str) -> Tuple[np.ndarray, np.ndarray]:
    params = load_roi_gmm_params(params_path)
    s0 = []
    s_inf = []
    for roi in roi_list:
        key = normalize_roi_name(roi)
        if key not in params.index:
            raise ValueError(f"Missing ROI in params: {roi}")
        s0.append(float(params.loc[key, "s0"]))
        s_inf.append(float(params.loc[key, "s_inf"]))
    return np.array(s0, dtype=float), np.array(s_inf, dtype=float)


def apply_roi_gmm_normalization(
    df: pd.DataFrame,
    roi_cols: List[str],
    s0_vec: np.ndarray,
    s_inf_vec: np.ndarray,
    eps_num: float,
    eps_den: float,
    eps_clip: float,
) -> pd.DataFrame:
    if not roi_cols:
        return df
    if s0_vec.shape[0] != len(roi_cols) or s_inf_vec.shape[0] != len(roi_cols):
        raise ValueError("ROI params length does not match roi_cols length.")
    out = df.copy()
    vals = out[roi_cols].to_numpy(dtype=float)
    s0 = s0_vec.reshape(1, -1)
    s_inf = s_inf_vec.reshape(1, -1)
    denom = np.maximum(s_inf - s0, eps_den)
    vals = np.maximum(vals, s0 + eps_num)
    norm = (vals - s0) / denom
    norm = np.minimum(norm, 1.0 - eps_clip)
    roi_df = pd.DataFrame(norm.astype(float), columns=roi_cols, index=out.index)
    cols = out.columns
    out = out.drop(columns=roi_cols).join(roi_df)
    out = out.loc[:, cols]
    return out


def apply_roi_sinf_clip(
    df: pd.DataFrame,
    roi_cols: List[str],
    s_inf_vec: np.ndarray,
    eps_clip: float,
) -> pd.DataFrame:
    if not roi_cols:
        return df
    if s_inf_vec.shape[0] != len(roi_cols):
        raise ValueError("s_inf length does not match roi_cols length.")
    out = df.copy()
    vals = out[roi_cols].to_numpy(dtype=float)
    s_inf = s_inf_vec.reshape(1, -1)
    vals = np.minimum(vals, s_inf - eps_clip)
    roi_df = pd.DataFrame(vals.astype(float), columns=roi_cols, index=out.index)
    cols = out.columns
    out = out.drop(columns=roi_cols).join(roi_df)
    out = out.loc[:, cols]
    return out


def _filter_sc_rois(
    A: np.ndarray, roi_list: List[str], roi_subset: str | None
) -> Tuple[np.ndarray, List[str]]:
    keep_idx, keep_list = _roi_subset_indices(roi_list, roi_subset)
    if keep_idx is None:
        return A, roi_list
    A_sub = A[np.ix_(keep_idx, keep_idx)]
    return A_sub, keep_list


def _roi_subset_indices(
    roi_list: List[str], roi_subset: str | None
) -> Tuple[List[int] | None, List[str]]:
    if roi_subset is None or str(roi_subset).lower() in {"all", "none"}:
        return None, roi_list
    subset = str(roi_subset).lower()
    if subset not in {"cortex_hippo_amygdala", "roi72", "hippo_amygdala"}:
        raise ValueError(f"Unknown roi_subset: {roi_subset}")

    hip_amy = {
        "lefthippocampus",
        "righthippocampus",
        "leftamygdala",
        "rightamygdala",
    }
    keep_idx = []
    keep_list = []
    for idx, roi in enumerate(roi_list):
        roi_lower = str(roi).lower()
        key = normalize_roi_name(roi)
        is_ctx = roi_lower.startswith("ctx-lh-") or roi_lower.startswith("ctx-rh-")
        if subset == "hippo_amygdala":
            keep = key in hip_amy
        else:
            keep = is_ctx or key in hip_amy
        if keep:
            keep_idx.append(idx)
            keep_list.append(roi)
    if not keep_idx:
        raise ValueError("ROI subset filtering produced empty list.")
    return keep_idx, keep_list


def load_sc(sc_path: str, roi_subset: str | None = None) -> Tuple[np.ndarray, List[str], Dict[str, float]]:
    A, roi_list, stats, _, _, _ = prepare_sc(sc_path, roi_subset=roi_subset, subset_after_laplacian=False)
    return A, roi_list, stats


def prepare_sc(
    sc_path: str,
    roi_subset: str | None = None,
    subset_after_laplacian: bool = False,
) -> Tuple[np.ndarray, List[str], Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(sc_path)
    roi_list = df.columns[1:].tolist()
    df = df.set_index(df.columns[0])
    df = df[roi_list]
    A = df.to_numpy(dtype=float)

    sym_diff = np.abs(A - A.T)
    stats = {
        "sym_max_abs_diff": float(np.max(sym_diff)),
        "sym_mean_abs_diff": float(np.mean(sym_diff)),
        "diag_max_abs": float(np.max(np.abs(np.diag(A)))),
        "neg_count": int((A < 0).sum()),
        "n": int(A.shape[0]),
    }

    A = (A + A.T) / 2.0
    np.fill_diagonal(A, 0.0)
    A = np.maximum(A, 0.0)
    if subset_after_laplacian:
        L_full, L_norm_full, L_rw_full = compute_laplacians(A)
        keep_idx, keep_list = _roi_subset_indices(roi_list, roi_subset)
        if keep_idx is None:
            return A, roi_list, stats, L_full, L_norm_full, L_rw_full
        A_sub = A[np.ix_(keep_idx, keep_idx)]
        L_sub = L_full[np.ix_(keep_idx, keep_idx)]
        L_norm_sub = L_norm_full[np.ix_(keep_idx, keep_idx)]
        L_rw_sub = L_rw_full[np.ix_(keep_idx, keep_idx)]
        return A_sub, keep_list, stats, L_sub, L_norm_sub, L_rw_sub

    A_sub, roi_sub = _filter_sc_rois(A, roi_list, roi_subset)
    L_sub, L_norm_sub, L_rw_sub = compute_laplacians(A_sub)
    return A_sub, roi_sub, stats, L_sub, L_norm_sub, L_rw_sub


def compute_laplacians(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    deg = A.sum(axis=1)
    L = np.diag(deg) - A

    with np.errstate(divide="ignore"):
        inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    D_inv_sqrt = np.diag(inv_sqrt)
    L_norm = np.eye(A.shape[0]) - D_inv_sqrt @ A @ D_inv_sqrt
    with np.errstate(divide="ignore"):
        inv_deg = np.where(deg > 0, 1.0 / deg, 0.0)
    D_inv = np.diag(inv_deg)
    L_rw = np.eye(A.shape[0]) - D_inv @ A
    return L, L_norm, L_rw


def compute_laplacian_variant(A: np.ndarray, mode: str) -> np.ndarray:
    mode = str(mode).lower()
    if mode in {"raw", "laplacian", ""}:
        deg = A.sum(axis=1)
        return np.diag(deg) - A
    if mode in {"amax", "amax_lnorm"}:
        scale = float(np.max(A))
        if not np.isfinite(scale) or scale <= 0.0:
            scale = 1.0
        A_scaled = A / scale
        deg = A_scaled.sum(axis=1)
        L = np.diag(deg) - A_scaled
        if mode == "amax_lnorm":
            denom = float(np.max(np.abs(L)))
            if not np.isfinite(denom) or denom <= 0.0:
                denom = 1.0
            L = L / denom
        return L
    raise ValueError(f"Unknown laplacian variant: {mode}")


def select_laplacian(
    A: np.ndarray,
    laplacian: str,
    L: np.ndarray | None = None,
    L_norm: np.ndarray | None = None,
    L_rw: np.ndarray | None = None,
) -> np.ndarray:
    mode = str(laplacian).lower()
    if mode == "random_walk":
        mode = "rw"
    if mode == "norm":
        if L_norm is None:
            _, L_norm, _ = compute_laplacians(A)
        return L_norm
    if mode == "rw":
        if L_rw is None:
            _, _, L_rw = compute_laplacians(A)
        return L_rw
    if mode in {"raw", "laplacian", "", "amax", "amax_lnorm"}:
        if mode in {"raw", "laplacian", ""} and L is not None:
            return L
        return compute_laplacian_variant(A, mode)
    raise ValueError(f"Unknown laplacian mode: {mode}")


def load_tau(tau_path: str) -> pd.DataFrame:
    df = pd.read_csv(tau_path)
    return df


def build_tau_roi_mapping(tau_df: pd.DataFrame) -> Dict[str, str]:
    suvr_cols = [c for c in tau_df.columns if c.endswith("_SUVR")]
    mapping = {}
    for col in suvr_cols:
        base = col[: -len("_SUVR")]
        key = normalize_roi_name(base)
        if key in mapping:
            mapping[key].append(col)
        else:
            mapping[key] = [col]
    return mapping


def align_tau_to_sc(
    tau_df: pd.DataFrame, roi_list: List[str]
) -> Tuple[List[str], Dict[str, List[str]]]:
    mapping = build_tau_roi_mapping(tau_df)
    ordered_cols = []
    missing = {}
    for roi in roi_list:
        key = normalize_roi_name(roi)
        cols = mapping.get(key, [])
        if len(cols) == 1:
            ordered_cols.append(cols[0])
        else:
            missing[roi] = cols
    return ordered_cols, missing


def parse_viscode2(viscode: str) -> float:
    if pd.isna(viscode):
        return np.nan
    s = str(viscode).strip().lower()
    match = re.match(r"m(\d+)$", s)
    if match:
        return float(match.group(1))
    if s.isdigit():
        return float(s)
    return np.nan


def subject_id_column(df: pd.DataFrame) -> str:
    if "RID" in df.columns:
        return "RID"
    if "PTID" in df.columns:
        return "PTID"
    raise ValueError("No RID/PTID column found in tau CSV.")


def run_sanity_checks(
    sc_path: str,
    tau_path: str,
    output_dir: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    A, roi_list, sc_stats = load_sc(sc_path)
    L, L_norm, L_rw = compute_laplacians(A)

    roi_txt = os.path.join(output_dir, "roi_list.txt")
    with open(roi_txt, "w", encoding="utf-8") as f:
        for roi in roi_list:
            f.write(f"{roi}\n")

    np.save(os.path.join(output_dir, "sc_matrix.npy"), A)
    np.save(os.path.join(output_dir, "laplacian.npy"), L)
    np.save(os.path.join(output_dir, "laplacian_norm.npy"), L_norm)

    tau_df = load_tau(tau_path)
    ordered_cols, missing = align_tau_to_sc(tau_df, roi_list)

    tau_df["t_month"] = tau_df["VISCODE2"].apply(parse_viscode2)
    bad_time = int(tau_df["t_month"].isna().sum())

    subj_col = subject_id_column(tau_df)
    counts = (
        tau_df.dropna(subset=["t_month"])
        .groupby(subj_col)["t_month"]
        .nunique()
        .sort_values()
    )
    counts_desc = counts.describe()
    at_least_2 = int((counts >= 2).sum())

    summary = {
        "sc_n": sc_stats["n"],
        "tau_suvr_columns": int(len([c for c in tau_df.columns if c.endswith('_SUVR')])),
        "roi_match_total": int(len(roi_list)),
        "roi_match_found": int(len(ordered_cols)),
        "roi_match_missing": int(len(missing)),
        "sc_sym_max_abs_diff": sc_stats["sym_max_abs_diff"],
        "sc_sym_mean_abs_diff": sc_stats["sym_mean_abs_diff"],
        "sc_diag_max_abs": sc_stats["diag_max_abs"],
        "sc_neg_count": sc_stats["neg_count"],
        "viscode2_bad_count": bad_time,
        "subjects_total": int(counts.shape[0]),
        "subjects_with_ge_2_timepoints": at_least_2,
        "timepoints_min": float(counts_desc["min"]) if counts.shape[0] else 0.0,
        "timepoints_median": float(counts_desc["50%"]) if counts.shape[0] else 0.0,
        "timepoints_mean": float(counts_desc["mean"]) if counts.shape[0] else 0.0,
        "timepoints_max": float(counts_desc["max"]) if counts.shape[0] else 0.0,
    }

    with open(os.path.join(output_dir, "sanity_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=== Sanity Checks ===")
    print(f"N (SC) = {summary['sc_n']}")
    print(
        f"ROI match: {summary['roi_match_found']}/{summary['roi_match_total']} "
        f"(missing {summary['roi_match_missing']})"
    )
    if missing:
        missing_preview = list(missing.items())[:10]
        print("Missing/multi-match ROIs (first 10):")
        for roi, cols in missing_preview:
            print(f"  {roi}: {cols}")
    print(
        "SC stats: sym_max_abs_diff={:.6f}, sym_mean_abs_diff={:.6f}, diag_max_abs={:.6f}, neg_count={}".format(
            summary["sc_sym_max_abs_diff"],
            summary["sc_sym_mean_abs_diff"],
            summary["sc_diag_max_abs"],
            summary["sc_neg_count"],
        )
    )
    print(f"VISCODE2 parse failures: {summary['viscode2_bad_count']}")
    print(
        "Timepoints per subject: min={:.1f}, median={:.1f}, mean={:.1f}, max={:.1f}".format(
            summary["timepoints_min"],
            summary["timepoints_median"],
            summary["timepoints_mean"],
            summary["timepoints_max"],
        )
    )
    print(f"Subjects with >=2 timepoints: {summary['subjects_with_ge_2_timepoints']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Load and sanity-check SC and Tau data.")
    parser.add_argument(
        "--sc_path",
        type=str,
        default="micapipe_sc_avg_HC001-050_clean82.csv",
        help="Path to SC CSV.",
    )
    parser.add_argument(
        "--tau_path",
        type=str,
        default="TauPVC_intersection_minROI_LUTordered.csv",
        help="Path to Tau CSV.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="artifacts",
        help="Directory to save intermediate outputs.",
    )
    args = parser.parse_args()
    run_sanity_checks(args.sc_path, args.tau_path, args.output_dir)


if __name__ == "__main__":
    main()
