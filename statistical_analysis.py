#!/usr/bin/env python3
"""Statistical analysis for decay-vs-CONST and family-level experiments.

The script performs a primary paired decay-vs-CONST analysis and a secondary
family-level analysis with descriptive statistics, ANOVA, Tukey HSD, rankings, and
summaries. Paired inference can use either a median-collapsed or mean-collapsed
CONST baseline via --paired_delta_basis.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, f_oneway, wilcoxon

try:
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
except Exception:  # pragma: no cover - handled gracefully at runtime
    pairwise_tukeyhsd = None


METRIC_CHOICES = ["MCC", "Accuracy", "F1_Score", "ROC_AUC", "PR_AUC"]

FAMILY_MAP = {
    "DAE_CONST": "CONST",
    "DAE_LINEAR": "LINEAR",
    "DAE_EXP": "EXPONENTIAL",
    "DAE_COSINE": "COSINE",
    "DAE_FIB": "FIBONACCI",
    "DAE_SIGMOID": "SIGMOID",
    "DAE_CAUCHY": "CAUCHY",
    "DAE_LAPLACE": "LAPLACE",
    "DAE_LOGISTIC": "LOGISTIC",
    "DVAE_CONST": "CONST",
    "DVAE_LINEAR": "LINEAR",
    "DVAE_EXP": "EXPONENTIAL",
    "DVAE_COSINE": "COSINE",
    "DVAE_FIB": "FIBONACCI",
    "DVAE_SIGMOID": "SIGMOID",
    "DVAE_CAUCHY": "CAUCHY",
    "DVAE_LAPLACE": "LAPLACE",
    "DVAE_LOGISTIC": "LOGISTIC",
    "RES_DAE_CONST": "CONST",
    "RES_DAE_LINEAR": "LINEAR",
    "RES_DAE_EXP": "EXPONENTIAL",
    "RES_DAE_COSINE": "COSINE",
    "RES_DAE_FIB": "FIBONACCI",
    "RES_DAE_SIGMOID": "SIGMOID",
    "RES_DAE_CAUCHY": "CAUCHY",
    "RES_DAE_LAPLACE": "LAPLACE",
    "RES_DAE_LOGISTIC": "LOGISTIC",
    "SPARSE_DAE_CONST": "CONST",
    "SPARSE_DAE_LINEAR": "LINEAR",
    "SPARSE_DAE_EXP": "EXPONENTIAL",
    "SPARSE_DAE_COSINE": "COSINE",
    "SPARSE_DAE_FIB": "FIBONACCI",
    "SPARSE_DAE_SIGMOID": "SIGMOID",
    "SPARSE_DAE_CAUCHY": "CAUCHY",
    "SPARSE_DAE_LAPLACE": "LAPLACE",
    "SPARSE_DAE_LOGISTIC": "LOGISTIC",
}

# Runtime columns exported by the experiment runner when available.
RUNTIME_SCHEMA_COL_ALIASES = {
    "Epochs": ("EpochsCfg",),
}

RUNTIME_SCHEMA_COLS = [
    "Dataset",
    "Seed",
    "AEVariant",
    "ConfigID",
    "ConfigInitSeed",
    "Model",
    "RangeName",
    "Epochs",
    "BatchSize",
    "LearningRate",
    "EncUnits",
    "DecUnits",
    "Latent",
    "NoiseType",
    "SelectionMetric",
    "AnomalyScoreType",
    "VAEBeta",
    "SparsityL1",
    "SelectedPercentile",
    "SelectedThreshold",
    "SigmaStart",
    "SigmaEnd",
    "SigmaMin",
    "DecayRate",
    "Accuracy",
    "Precision",
    "Recall",
    "F1_Score",
    "MCC",
    "ROC_AUC",
    "PR_AUC",
]

PARAM_COLS = ["SigmaStart", "SigmaEnd", "SigmaMin", "DecayRate"]
RUN_TRACE_COLS = [
    "Seed",
    "AEVariant",
    "ConfigID",
    "ConfigInitSeed",
    "RuntimeConfigID",
    "BatchSize",
    "LearningRate",
    "EncUnits",
    "DecUnits",
    "Latent",
    "Epochs",
    "NoiseType",
    "Architecture",
    "SelectionMetric",
    "AnomalyScoreType",
    "VAEBeta",
    "SparsityL1",
    "SelectedPercentile",
    "SelectedThreshold",
    "RangeName",
] + PARAM_COLS
NOISE_FAMILIES = set(FAMILY_MAP.values())
EMPTY_TUKEY_COLUMNS = ["group1", "group2", "meandiff", "p_adj", "lower", "upper", "reject"]


def empty_tukey_df(ae_variant=None, family_unit_used=None):
    cols = []
    data = {}
    if family_unit_used is not None:
        cols.append("family_unit_used")
        data["family_unit_used"] = []
    if ae_variant is not None:
        cols.append("AEVariant")
        data["AEVariant"] = []
    for c in EMPTY_TUKEY_COLUMNS:
        cols.append(c)
        data[c] = []
    return pd.DataFrame(data, columns=cols)


def str2bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return True
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "t"}:
        return True
    if text in {"false", "0", "no", "n", "f"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_int_csv_list(raw: Optional[str]) -> Optional[List[int]]:
    if raw is None:
        return None
    items = [x.strip() for x in str(raw).split(",") if x.strip() != ""]
    if not items:
        return []
    try:
        return [int(x) for x in items]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid integer CSV list: {raw}") from exc


def parse_str_csv_list(raw: Optional[str]) -> Optional[List[str]]:
    if raw is None:
        return None
    items = [x.strip() for x in str(raw).split(",") if x.strip() != ""]
    return items


def normalize_ae_variant(value: object, default: str = "FF_DAE") -> str:
    if value is None or pd.isna(value):
        return default
    text = str(value).strip().upper()
    if text in {"", "NAN", "NONE", "NULL"}:
        return default
    aliases = {
        "FF": "FF_DAE",
        "DAE": "FF_DAE",
        "FF_DAE": "FF_DAE",
        "DVAE": "DVAE",
        "RES": "RES_DAE",
        "RES_DAE": "RES_DAE",
        "SPARSE": "SPARSE_DAE",
        "SPARSE_DAE": "SPARSE_DAE",
    }
    return aliases.get(text, text)


def normalize_ae_variant_list(values: Optional[Sequence[str]]) -> Optional[List[str]]:
    if values is None:
        return None
    normalized = []
    for value in values:
        variant = normalize_ae_variant(value)
        if variant not in normalized:
            normalized.append(variant)
    return normalized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run paired (primary inferential) and/or family-level (secondary ANOVA/Tukey) analysis on IDS experiment metrics."
        )
    )
    parser.add_argument("--input_dir", default="generated_outputs")
    parser.add_argument("--output_dir", default="generated_outputs/statistical_outputs")
    parser.add_argument("--analysis_mode", default="both", choices=["paired", "family", "both"])
    parser.add_argument(
        "--paired_delta_basis",
        default="median",
        choices=["median", "mean"],
        help=(
            "Controls paired-mode delta inference baseline collapse: "
            "Default is 'median'. "
            "'median' uses delta_*_vs_const_median and 'mean' uses delta_*_vs_const_mean."
        ),
    )
    parser.add_argument(
        "--family_unit",
        default="seed_median",
        choices=["raw", "seed_mean", "seed_median"],
        help=(
            "Family-mode unit: raw pooled rows or seed-collapsed Dataset×Seed×Family summaries. "
            "Default is 'seed_median'."
        ),
    )
    parser.add_argument("--metric", default="MCC", choices=METRIC_CHOICES)
    parser.add_argument("--top_n", type=int, default=10)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--min_group_size", type=int, default=2)
    parser.add_argument("--no_plots", action="store_true")
    parser.add_argument("--noise_only", action="store_true", default=False)
    parser.add_argument("--deduplicate", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--noise_types",
        type=str,
        default=None,
        help="Optional comma-separated noise types to retain, e.g. gaussian,masking,none",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Optional comma-separated integer seeds to retain, e.g. 42,7,123",
    )
    parser.add_argument(
        "--ae_variants",
        type=str,
        default=None,
        help="Optional comma-separated AE variants to retain, e.g. FF_DAE,DVAE,RES_DAE,SPARSE_DAE",
    )
    args = parser.parse_args()
    args.noise_types = parse_str_csv_list(args.noise_types)
    args.seeds = parse_int_csv_list(args.seeds)
    args.ae_variants = normalize_ae_variant_list(parse_str_csv_list(args.ae_variants))
    return args


def _parse_seed_from_path(path: Path) -> Optional[int]:
    m = re.search(r"_seed_?(\d+)(?:\.csv)?$", path.name)
    return int(m.group(1)) if m else None


def _parse_seed_from_metric_path(path: Path) -> Optional[int]:
    seed_from_file = _parse_seed_from_path(path)
    if seed_from_file is not None:
        return seed_from_file
    for ancestor in path.parents:
        seed_from_ancestor = _parse_seed_from_path(ancestor)
        if seed_from_ancestor is not None:
            return seed_from_ancestor
    return None


def find_run_metric_files(
    output_root: Path,
    metric_filename: str,
    dataset_name: Optional[str] = None,
    seeds: Optional[Sequence[int]] = None,
) -> List[Path]:
    runs_root = output_root / "runs"
    if not runs_root.exists():
        return []
    dataset_glob = dataset_name if dataset_name else "*"
    patterns = [
        f"{dataset_glob}/*/*/metrics/{metric_filename}",
    ]
    if metric_filename.startswith("final_test_metrics"):
        patterns.extend([
            f"{dataset_glob}/*/*/metrics/*_final_test_metrics_seed*.csv",
        ])
    if metric_filename.startswith("train_val_metrics"):
        patterns.extend([
            f"{dataset_glob}/*/*/metrics/*_train_val_metrics_seed*.csv",
        ])
    if metric_filename.startswith("selected_threshold_rows"):
        patterns.extend([
            f"{dataset_glob}/*/*/metrics/*_selected_threshold_rows_seed*.csv",
        ])
    if metric_filename.endswith("_run.csv"):
        seed_metric_filename = metric_filename.replace("_run.csv", "_seed*.csv")
        patterns.extend([
            f"{dataset_glob}/*/*/metrics/{seed_metric_filename}",
        ])
    files = sorted({path for pattern in patterns for path in runs_root.glob(pattern)})
    if seeds is None:
        return files
    seed_set = {int(s) for s in seeds}
    filtered: List[Path] = []
    for f in files:
        inferred_seed = _parse_seed_from_metric_path(f)
        # Keep files with unknown seed (e.g., *_run.csv) and defer filtering to row-level "Seed".
        if inferred_seed is None or inferred_seed in seed_set:
            filtered.append(f)
    return filtered


def load_run_metric_files(output_root: Path, metric_filename: str, dataset_name: Optional[str] = None, seeds: Optional[Sequence[int]] = None) -> pd.DataFrame:
    files = find_run_metric_files(output_root, metric_filename, dataset_name=dataset_name, seeds=seeds)
    if not files:
        raise FileNotFoundError(f"No matching metric files found for '{metric_filename}' under {output_root / 'runs'}.")
    df = load_and_combine(files)
    if df.empty:
        raise ValueError(f"Matched files for '{metric_filename}', but no rows were loaded.")
    if seeds is not None and "Seed" in df.columns:
        df = df[df["Seed"].isin([int(s) for s in seeds])]
    return df.drop_duplicates().reset_index(drop=True)




def make_pair_core_id(row: pd.Series) -> str:
    ae_variant = normalize_ae_variant(row.get("AEVariant"))
    anomaly_score_type = row.get("AnomalyScoreType")
    if pd.isna(anomaly_score_type) or str(anomaly_score_type).strip() == "":
        anomaly_score_type = "elbo" if ae_variant == "DVAE" else "recon"
    vae_beta = row.get("VAEBeta")
    if pd.isna(vae_beta):
        vae_beta = 0.001 if ae_variant == "DVAE" else np.nan
    beta_token = "NA" if pd.isna(vae_beta) else str(float(vae_beta))
    sparsity_l1 = row.get("SparsityL1")
    sparsity_token = "NA" if pd.isna(sparsity_l1) else str(float(sparsity_l1))

    return (
        f"Dataset={row.get('Dataset')}|Seed={int(row.get('Seed'))}|AEVariant={ae_variant}|"
        f"NoiseType={str(row.get('NoiseType'))}|SelectionMetric={str(row.get('SelectionMetric'))}|"
        f"BatchSize={int(row.get('BatchSize'))}|LearningRate={float(row.get('LearningRate'))}|"
        f"EncUnits={str(row.get('EncUnits'))}|DecUnits={str(row.get('DecUnits'))}|"
        f"Latent={int(row.get('Latent'))}|Epochs={int(row.get('Epochs'))}|"
        f"SparsityL1={sparsity_token}|AnomalyScoreType={str(anomaly_score_type)}|VAEBeta={beta_token}"
    )


def ensure_pair_core_id(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "AEVariant" not in out.columns:
        out["AEVariant"] = "FF_DAE"
    out["AEVariant"] = out["AEVariant"].apply(normalize_ae_variant)
    if "AnomalyScoreType" not in out.columns:
        out["AnomalyScoreType"] = np.nan
    missing_score_type = out["AnomalyScoreType"].isna() | out["AnomalyScoreType"].astype(str).str.strip().eq("")
    out.loc[missing_score_type, "AnomalyScoreType"] = np.where(
        out.loc[missing_score_type, "AEVariant"].eq("DVAE"),
        "elbo",
        "recon",
    )
    if "VAEBeta" not in out.columns:
        out["VAEBeta"] = np.nan
    out["VAEBeta"] = pd.to_numeric(out["VAEBeta"], errors="coerce")
    missing_beta = out["VAEBeta"].isna() & out["AEVariant"].eq("DVAE")
    out.loc[missing_beta, "VAEBeta"] = 0.001
    if "SparsityL1" not in out.columns:
        out["SparsityL1"] = np.nan

    if "PairCoreID" in out.columns:
        pair_core = out["PairCoreID"].astype("object")
        pair_core_str = pair_core.astype(str)
        non_empty = pair_core.notna() & ~pair_core_str.str.strip().str.lower().isin({"", "nan", "none", "null"})
        has_required_context = (
            pair_core_str.str.contains("AnomalyScoreType=", regex=False)
            & pair_core_str.str.contains("VAEBeta=", regex=False)
            & pair_core_str.str.contains("SparsityL1=", regex=False)
        )
        if (non_empty & has_required_context).all():
            out["PairCoreID"] = pair_core_str
            return out
    req = [
        "Dataset",
        "Seed",
        "NoiseType",
        "SelectionMetric",
        "BatchSize",
        "LearningRate",
        "EncUnits",
        "DecUnits",
        "Latent",
        "Epochs",
        "SparsityL1",
    ]
    missing = [c for c in req if c not in out.columns]
    if missing:
        raise ValueError(f"Cannot recreate PairCoreID; missing columns: {missing}")
    invalid = [c for c in ["Dataset", "Seed", "BatchSize", "LearningRate", "Latent", "Epochs"] if out[c].isna().any()]
    if invalid:
        raise ValueError(f"Cannot recreate PairCoreID; missing values in columns: {invalid}")
    out["PairCoreID"] = out.apply(make_pair_core_id, axis=1)
    return out


def build_independent_paired_decay_vs_const_df(
    final_rows_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build independent decay-vs-CONST pairs using collapsed CONST baselines."""
    diag_cols = [
        "PairCoreID",
        "AEVariant",
        "const_rows",
        "decay_rows",
        "const_collapse_method",
        "independent_pairs_created",
    ]
    empty_diag = pd.DataFrame(columns=diag_cols)
    if final_rows_df is None or final_rows_df.empty:
        return pd.DataFrame(), pd.DataFrame(), empty_diag

    work = ensure_pair_core_id(final_rows_df)
    if "Family" not in work.columns:
        return pd.DataFrame(), pd.DataFrame(), empty_diag

    work = work.copy()
    work["AEVariant"] = work.get("AEVariant", "FF_DAE")
    work["AEVariant"] = work["AEVariant"].apply(normalize_ae_variant)

    metric_cols = ["MCC", "F1_Score", "PR_AUC", "ROC_AUC", "Accuracy"]
    for metric_col in metric_cols:
        if metric_col not in work.columns:
            work[metric_col] = np.nan
        work[metric_col] = pd.to_numeric(work[metric_col], errors="coerce")

    const_df = work[work["Family"] == "CONST"].copy()
    decay_df = work[work["Family"] != "CONST"].copy()

    const_counts = const_df.groupby("PairCoreID").size() if not const_df.empty else pd.Series(dtype="int64")
    decay_counts = decay_df.groupby("PairCoreID").size() if not decay_df.empty else pd.Series(dtype="int64")
    all_paircores = sorted(set(const_counts.index.astype(str)).union(set(decay_counts.index.astype(str))))

    ae_variant_by_paircore = (
        work.drop_duplicates("PairCoreID").set_index("PairCoreID")["AEVariant"].to_dict()
        if "PairCoreID" in work.columns
        else {}
    )
    diag_rows = []
    for paircore_id in all_paircores:
        const_count = int(const_counts.get(paircore_id, 0))
        decay_count = int(decay_counts.get(paircore_id, 0))
        diag_rows.append(
            {
                "PairCoreID": paircore_id,
                "AEVariant": ae_variant_by_paircore.get(paircore_id, "FF_DAE"),
                "const_rows": const_count,
                "decay_rows": decay_count,
                "const_collapse_method": "median_and_mean_available",
                "independent_pairs_created": int(decay_count if const_count > 0 else 0),
            }
        )
    diag_df = pd.DataFrame(diag_rows, columns=diag_cols)

    def _empty_summary() -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "total_const_rows": int(const_df.shape[0]),
                    "total_decay_rows": int(decay_df.shape[0]),
                    "independent_pairs_created": 0,
                    "paircore_with_const": int(const_df["PairCoreID"].nunique()) if not const_df.empty else 0,
                    "paircore_with_decay": int(decay_df["PairCoreID"].nunique()) if not decay_df.empty else 0,
                    "const_collapse_method_primary": "paired_delta_basis_configurable",
                }
            ]
        )

    if const_df.empty or decay_df.empty:
        return pd.DataFrame(), _empty_summary(), diag_df

    const_agg = const_df.groupby("PairCoreID").agg({m: ["median", "mean"] for m in metric_cols})
    const_agg.columns = [f"{metric}_const_{stat}" for metric, stat in const_agg.columns]
    const_agg["const_rows_collapsed"] = const_df.groupby("PairCoreID").size()
    const_agg = const_agg.reset_index()

    merged = decay_df.merge(const_agg, on="PairCoreID", how="inner")
    if merged.empty:
        return pd.DataFrame(), _empty_summary(), diag_df

    paired = pd.DataFrame(
        {
            "Dataset": merged["Dataset"],
            "Seed": merged["Seed"],
            "AEVariant": merged["AEVariant"],
            "PairCoreID": merged["PairCoreID"],
            "Family": merged["Family"],
            "DecayConfigID": merged["ConfigID"],
            "DecayRangeName": merged["RangeName"],
            "NoiseType": merged["NoiseType"],
            "SelectionMetric": merged["SelectionMetric"],
            "SigmaStart": merged["SigmaStart"],
            "SigmaEnd": merged["SigmaEnd"],
            "SigmaMin": merged["SigmaMin"],
            "DecayRate": merged["DecayRate"],
            "const_rows_collapsed": merged["const_rows_collapsed"],
            "const_collapse_method_primary": "paired_delta_basis_configurable",
        }
    )

    for metric in metric_cols:
        paired[f"{metric}_decay"] = merged[metric]
        paired[f"{metric}_const_median"] = merged[f"{metric}_const_median"]
        paired[f"{metric}_const_mean"] = merged[f"{metric}_const_mean"]
        paired[f"delta_{metric}_vs_const_median"] = merged[metric] - merged[f"{metric}_const_median"]
        paired[f"delta_{metric}_vs_const_mean"] = merged[metric] - merged[f"{metric}_const_mean"]

    ordered_cols = [
        "Dataset",
        "Seed",
        "AEVariant",
        "PairCoreID",
        "Family",
        "DecayConfigID",
        "DecayRangeName",
        "NoiseType",
        "SelectionMetric",
        "SigmaStart",
        "SigmaEnd",
        "SigmaMin",
        "DecayRate",
    ]
    for metric in metric_cols:
        ordered_cols.extend(
            [
                f"{metric}_decay",
                f"{metric}_const_median",
                f"{metric}_const_mean",
                f"delta_{metric}_vs_const_median",
                f"delta_{metric}_vs_const_mean",
            ]
        )
    ordered_cols.extend(["const_rows_collapsed", "const_collapse_method_primary"])
    paired = paired.reindex(columns=ordered_cols)

    summary = pd.DataFrame(
        [
            {
                "total_const_rows": int(const_df.shape[0]),
                "total_decay_rows": int(decay_df.shape[0]),
                "independent_pairs_created": int(paired.shape[0]),
                "paircore_with_const": int(const_df["PairCoreID"].nunique()),
                "paircore_with_decay": int(decay_df["PairCoreID"].nunique()),
                "const_collapse_method_primary": "paired_delta_basis_configurable",
            }
        ]
    )
    return paired, summary, diag_df


def infer_dataset_from_filename(path: Path) -> str:
    if path.parent.name == "metrics":
        ancestors = path.parents
        for idx, parent in enumerate(ancestors):
            if parent.name == "runs" and idx > 0:
                return ancestors[idx - 1].name
    name = path.stem
    m = re.match(r"^test_metrics_(.+)_\d{4}-\d{2}-\d{2}$", name)
    if m:
        return m.group(1)

    parts = name.split("_")
    if len(parts) >= 4:
        return "_".join(parts[2:-1])
    if len(parts) >= 3:
        return parts[2]
    return "UNKNOWN"


def map_family(model: object) -> str:
    model_str = "" if pd.isna(model) else str(model).strip().upper()
    if model_str in FAMILY_MAP:
        return FAMILY_MAP[model_str]
    suffix_map = {
        "_CONST": "CONST",
        "_LINEAR": "LINEAR",
        "_EXP": "EXPONENTIAL",
        "_COSINE": "COSINE",
        "_FIB": "FIBONACCI",
        "_SIGMOID": "SIGMOID",
        "_CAUCHY": "CAUCHY",
        "_LAPLACE": "LAPLACE",
        "_LOGISTIC": "LOGISTIC",
    }
    for suffix, family in suffix_map.items():
        if model_str.endswith(suffix):
            return family
    return model_str if model_str else "UNKNOWN"


def _is_missing_scalar(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, (list, tuple, set, dict, np.ndarray, pd.Series)):
        return False
    return bool(pd.isna(value))


def _stringify_units(value: object) -> str:
    if _is_missing_scalar(value):
        return "?"
    txt = str(value).strip()
    if txt == "":
        return "?"
    return txt


def build_architecture_label(row: pd.Series) -> str:
    enc = _stringify_units(row.get("EncUnits"))
    lat = _stringify_units(row.get("Latent"))
    dec = _stringify_units(row.get("DecUnits"))
    ae_variant_raw = row.get("AEVariant")
    if ae_variant_raw is None or pd.isna(ae_variant_raw) or str(ae_variant_raw).strip() == "":
        prefix = "AE"
    else:
        prefix = normalize_ae_variant(ae_variant_raw)
    return f"{prefix}[{enc}->{lat}->{dec}]"


def _safe_str_for_label(value: object) -> str:
    if _is_missing_scalar(value):
        return "NA"
    text = str(value).strip()
    if text == "":
        return "NA"
    return text


def build_runtime_config_label(row: pd.Series) -> str:
    ae_variant = normalize_ae_variant(row.get("AEVariant"))
    fam = _safe_str_for_label(row.get("Family"))
    rng = _safe_str_for_label(row.get("RangeName"))
    bs = _safe_str_for_label(row.get("BatchSize"))
    lr = _safe_str_for_label(row.get("LearningRate"))
    ep = _safe_str_for_label(row.get("Epochs"))
    noise = _safe_str_for_label(row.get("NoiseType"))
    arch = _safe_str_for_label(row.get("Architecture"))
    return f"AEV={ae_variant}|{fam}|R={rng}|B={bs}|LR={lr}|E={ep}|N={noise}|{arch}"



def _normalize_noise_type(series: pd.Series) -> pd.Series:
    out = series.copy()
    out = out.astype("object")
    out = out.where(~out.isna(), "none")
    out = out.astype(str).str.strip()
    out = out.replace({"": "none", "None": "none", "nan": "none", "NaN": "none", "NULL": "none", "null": "none"})
    return out.str.lower()


def load_and_combine(files: Sequence[Path]) -> pd.DataFrame:
    frames = []
    for fp in files:
        try:
            df = pd.read_csv(fp)
        except Exception as exc:
            print(f"[WARN] Failed reading {fp}: {exc}")
            continue
        if df.empty:
            print(f"[WARN] Empty CSV skipped: {fp}")
            continue

        if "Dataset" not in df.columns:
            df["Dataset"] = infer_dataset_from_filename(fp)
        else:
            df["Dataset"] = df["Dataset"].fillna(infer_dataset_from_filename(fp)).astype(str)

        # Ensure expected schema columns are present to avoid silent drops downstream.
        for canonical_col, aliases in RUNTIME_SCHEMA_COL_ALIASES.items():
            if canonical_col not in df.columns:
                for alias in aliases:
                    if alias in df.columns:
                        df[canonical_col] = df[alias]
                        break
        for col in RUNTIME_SCHEMA_COLS:
            if col not in df.columns:
                df[col] = np.nan

        if "AEVariant" not in df.columns:
            df["AEVariant"] = "FF_DAE"
        df["AEVariant"] = df["AEVariant"].apply(normalize_ae_variant)
        missing_score_type = df["AnomalyScoreType"].isna() | df["AnomalyScoreType"].astype(str).str.strip().eq("")
        df.loc[missing_score_type, "AnomalyScoreType"] = np.where(
            df.loc[missing_score_type, "AEVariant"].eq("DVAE"),
            "elbo",
            "recon",
        )
        df["VAEBeta"] = pd.to_numeric(df["VAEBeta"], errors="coerce")
        missing_beta = df["VAEBeta"].isna() & df["AEVariant"].eq("DVAE")
        df.loc[missing_beta, "VAEBeta"] = 0.001

        if "NoiseType" in df.columns:
            df["NoiseType"] = _normalize_noise_type(df["NoiseType"])

        df["SourceFile"] = fp.name
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    if "Model" not in combined.columns:
        combined["Model"] = "UNKNOWN"
    combined["Family"] = combined["Model"].apply(map_family)
    combined["Architecture"] = combined.apply(build_architecture_label, axis=1)

    if "ConfigID" in combined.columns:
        combined["RuntimeConfigID"] = combined["ConfigID"].astype("object")
        combined["RuntimeConfigID"] = combined["RuntimeConfigID"].where(~combined["RuntimeConfigID"].isna(), "")
        combined["RuntimeConfigID"] = combined["RuntimeConfigID"].astype(str).str.strip()
    else:
        combined["RuntimeConfigID"] = ""

    missing_runtime = combined["RuntimeConfigID"].eq("") | combined["RuntimeConfigID"].str.lower().isin({"nan", "none", "null"})
    combined.loc[missing_runtime, "RuntimeConfigID"] = combined.loc[missing_runtime].apply(build_runtime_config_label, axis=1)

    return combined


def resolve_deduplication_keys(df: pd.DataFrame) -> Tuple[Optional[List[str]], str]:
    priority_1 = ["RunID", "ConfigID"]
    priority_2 = ["Dataset", "Seed", "AEVariant", "ConfigID"]
    fallback = [
        "Dataset",
        "Seed",
        "AEVariant",
        "Model",
        "RangeName",
        "NoiseType",
        "BatchSize",
        "LearningRate",
        "EncUnits",
        "DecUnits",
        "Latent",
        "Epochs",
        "SelectionMetric",
        "SelectedPercentile",
        "SelectedThreshold",
    ]

    if all(c in df.columns for c in priority_1):
        return priority_1, "RunID+ConfigID"
    if all(c in df.columns for c in priority_2):
        return priority_2, "Dataset+Seed+AEVariant+ConfigID"

    if "Seed" not in df.columns:
        fallback = [c for c in fallback if c != "Seed"]

    existing_fallback = [c for c in fallback if c in df.columns]
    if not existing_fallback:
        return None, "no_applicable_key"
    return existing_fallback, "fallback_broad_key"


def apply_deduplication(df: pd.DataFrame, enabled: bool) -> Tuple[pd.DataFrame, str, int]:
    if not enabled:
        return df.copy(), "disabled", 0
    keys, mode = resolve_deduplication_keys(df)
    if keys is None:
        print("[WARN] Deduplication requested but no applicable key columns were found; skipping deduplication.")
        return df.copy(), mode, 0
    before = len(df)
    deduped = df.drop_duplicates(subset=keys, keep="last").copy()
    removed = before - len(deduped)
    print(f"[INFO] Deduplication enabled: mode={mode}, keys={keys}, rows_removed={removed}")
    return deduped, mode, removed


def _family_group_cols(df: pd.DataFrame) -> List[str]:
    """Return family-analysis grouping columns, preserving AE variants when present."""
    return ["AEVariant", "Family"] if "AEVariant" in df.columns else ["Family"]


def descriptive_stats(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    group_cols = _family_group_cols(df)
    grouped = df.groupby(group_cols, dropna=False)[metric]
    out = grouped.agg(["count", "mean", "median", "std", "min", "max"]).reset_index()
    q1 = grouped.quantile(0.25).rename("q1").reset_index()
    q3 = grouped.quantile(0.75).rename("q3").reset_index()
    out = out.merge(q1, on=group_cols, how="left").merge(q3, on=group_cols, how="left")
    out["iqr"] = out["q3"] - out["q1"]
    out["best"] = out["max"]
    return out[group_cols + ["count", "mean", "median", "std", "min", "max", "q1", "q3", "iqr", "best"]]


def best_by_family(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    group_cols = _family_group_cols(df)
    cols_preferred = ["Dataset", "AEVariant", "Family", "Model"] + [c for c in RUN_TRACE_COLS if c in df.columns] + [
        m for m in METRIC_CHOICES if m in df.columns
    ]
    idx = df.groupby(group_cols, dropna=False)[metric].idxmax()
    best = df.loc[idx].copy().sort_values(group_cols + [metric], ascending=[True] * len(group_cols) + [False])

    cols_existing = []
    seen = set()
    for c in cols_preferred:
        if c in best.columns and c not in seen:
            cols_existing.append(c)
            seen.add(c)
    remaining = [c for c in best.columns if c not in seen]
    return best[cols_existing + remaining]


def top_family_rows(df: pd.DataFrame, metric: str, top_n: int) -> pd.DataFrame:
    return df.sort_values(metric, ascending=False).head(top_n).copy()


def prepare_anova_groups(df: pd.DataFrame, metric: str, min_group_size: int) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
    if "AEVariant" in df.columns and df["AEVariant"].dropna().astype(str).nunique() > 1:
        raise ValueError("prepare_anova_groups received multiple AEVariant values; run ANOVA separately per AEVariant.")

    counts = df.groupby("Family")[metric].apply(lambda x: x.notna().sum()).rename("count")
    eligible_families = counts[counts >= min_group_size].index.tolist()
    eligible_df = df[df["Family"].isin(eligible_families)].copy()

    groups = {
        fam: g[metric].dropna().values
        for fam, g in eligible_df.groupby("Family")
        if g[metric].dropna().shape[0] >= min_group_size
    }
    return groups, eligible_df


def filter_by_family_mode(df: pd.DataFrame, noise_only: bool) -> Tuple[pd.DataFrame, str]:
    if noise_only:
        filtered = df[df["Family"].isin(NOISE_FAMILIES)].copy()
        return filtered, "noise_families_only"
    return df.copy(), "all_dae_families"


def filter_by_seed_noise_and_ae_variant(
    df: pd.DataFrame,
    seeds: Optional[List[int]],
    noise_types: Optional[List[str]],
    ae_variants: Optional[List[str]],
) -> Tuple[pd.DataFrame, str]:
    out = df.copy()
    filter_summary = []

    if seeds is not None and len(seeds) > 0:
        if "Seed" not in out.columns:
            out = out.iloc[0:0].copy()
            filter_summary.append("seed_filter_requested_but_seed_column_missing")
        else:
            seed_vals = pd.to_numeric(out["Seed"], errors="coerce")
            out = out[seed_vals.isin(seeds)].copy()
            filter_summary.append(f"seeds={seeds}")

    if noise_types is not None and len(noise_types) > 0:
        if "NoiseType" not in out.columns:
            out = out.iloc[0:0].copy()
            filter_summary.append("noise_filter_requested_but_noise_column_missing")
        else:
            wanted = {n.strip().lower() for n in noise_types if n.strip() != ""}
            out = out[out["NoiseType"].astype(str).str.lower().isin(wanted)].copy()
            filter_summary.append(f"noise_types={sorted(wanted)}")

    if ae_variants is not None and len(ae_variants) > 0:
        if "AEVariant" not in out.columns:
            out = out.iloc[0:0].copy()
            filter_summary.append("ae_variant_filter_requested_but_aevariant_column_missing")
        else:
            wanted_ae = {normalize_ae_variant(v) for v in ae_variants if str(v).strip() != ""}
            out = out[out["AEVariant"].apply(normalize_ae_variant).isin(wanted_ae)].copy()
            filter_summary.append(f"ae_variants={sorted(wanted_ae)}")

    return out, "; ".join(filter_summary) if filter_summary else "no_seed_noise_or_ae_variant_filter"


def filter_by_seed_and_noise(df: pd.DataFrame, seeds: Optional[List[int]], noise_types: Optional[List[str]]) -> Tuple[pd.DataFrame, str]:
    return filter_by_seed_noise_and_ae_variant(df, seeds=seeds, noise_types=noise_types, ae_variants=None)


def run_anova(
    df: pd.DataFrame,
    dataset: str,
    metric: str,
    min_group_size: int,
    ae_variant: Optional[str] = None,
) -> Tuple[pd.DataFrame, Optional[str], pd.DataFrame, bool]:
    groups, eligible_df = prepare_anova_groups(df, metric, min_group_size)
    variant_label = normalize_ae_variant(ae_variant) if ae_variant is not None else None

    def base_row() -> Dict[str, object]:
        row: Dict[str, object] = {
            "metric": metric,
            "dataset": dataset,
            "num_groups": len(groups),
            "num_observations": int(sum(len(v) for v in groups.values())),
            "f_statistic": np.nan,
            "p_value": np.nan,
        }
        if variant_label is not None:
            row["AEVariant"] = variant_label
        return row

    if len(groups) < 2:
        variant_part = f", AEVariant={variant_label}" if variant_label is not None else ""
        note = (
            f"ANOVA skipped for dataset={dataset}{variant_part}, metric={metric}: "
            f"need >=2 eligible families with at least min_group_size={min_group_size} observations. "
            f"Found {len(groups)} eligible families."
        )
        out = pd.DataFrame([base_row()])
        return out, note, eligible_df, False

    f_stat, p_val = f_oneway(*groups.values())
    row = base_row()
    row["f_statistic"] = float(f_stat)
    row["p_value"] = float(p_val)
    out = pd.DataFrame([row])
    return out, None, eligible_df, True


def run_tukey(df_eligible: pd.DataFrame, metric: str, alpha: float) -> Tuple[pd.DataFrame, Optional[str]]:
    if pairwise_tukeyhsd is None:
        return pd.DataFrame(), "statsmodels is unavailable; Tukey HSD skipped."
    if df_eligible.empty:
        return pd.DataFrame(), "No eligible rows for Tukey HSD."
    if df_eligible["Family"].nunique() < 2:
        return pd.DataFrame(), "Fewer than 2 eligible families for Tukey HSD."

    tukey = pairwise_tukeyhsd(endog=df_eligible[metric].astype(float), groups=df_eligible["Family"].astype(str), alpha=alpha)
    res = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])
    res = res.rename(
        columns={
            "group1": "group1",
            "group2": "group2",
            "meandiff": "meandiff",
            "p-adj": "p_adj",
            "lower": "lower",
            "upper": "upper",
            "reject": "reject",
        }
    )
    keep = ["group1", "group2", "meandiff", "p_adj", "lower", "upper", "reject"]
    return res[keep], None


def build_seed_collapsed_family_df(df: pd.DataFrame, metric: str, agg: str = "mean") -> pd.DataFrame:
    """Collapse rows to one Dataset×Seed×AEVariant×Family summary for family ANOVA/Tukey."""
    out_cols = [
        "Dataset",
        "Seed",
        "AEVariant",
        "Family",
        "FamilyMetricCollapsed",
        "CollapsedMetricValue",
        "FamilyRowCountWithinSeed",
    ]
    if df.empty:
        return pd.DataFrame(columns=out_cols)
    if agg not in {"mean", "median"}:
        raise ValueError(f"Unsupported aggregation '{agg}'. Expected one of ['mean', 'median'].")
    required = {"Dataset", "Seed", "Family", metric}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Cannot build seed-collapsed family table; missing columns: {missing}")

    work = df.copy()
    if "AEVariant" not in work.columns:
        work["AEVariant"] = "FF_DAE"
    work["AEVariant"] = work["AEVariant"].apply(normalize_ae_variant)
    work[metric] = pd.to_numeric(work[metric], errors="coerce")
    work = work.dropna(subset=[metric])
    if work.empty:
        return pd.DataFrame(columns=out_cols)

    grouped = work.groupby(["Dataset", "Seed", "AEVariant", "Family"], dropna=False)[metric]
    agg_vals = grouped.mean() if agg == "mean" else grouped.median()
    counts = grouped.size()
    out = pd.DataFrame(
        {
            "FamilyMetricCollapsed": agg_vals,
            "FamilyRowCountWithinSeed": counts.astype(int),
        }
    ).reset_index()
    # Backward-compatible alias used by earlier family-mode code paths.
    out["CollapsedMetricValue"] = out["FamilyMetricCollapsed"]
    return out[out_cols]


def extract_top_tukey_families(tukey_df: pd.DataFrame, desc_df: pd.DataFrame) -> Set[str]:
    if tukey_df.empty or desc_df.empty:
        return set()
    means = desc_df.set_index("Family")["mean"].to_dict()

    non_sig_edges: Dict[str, Set[str]] = {}
    families = set(desc_df["Family"].unique())
    for fam in families:
        non_sig_edges[fam] = {fam}

    for _, row in tukey_df.iterrows():
        g1, g2, reject = row["group1"], row["group2"], bool(row["reject"])
        if not reject:
            non_sig_edges.setdefault(g1, {g1}).add(g2)
            non_sig_edges.setdefault(g2, {g2}).add(g1)

    if not means:
        return set()
    top_family = max(means, key=means.get)
    return non_sig_edges.get(top_family, {top_family})


def _annotate_top_tukey_group(merged: pd.DataFrame, desc_df: pd.DataFrame, tukey_df: pd.DataFrame) -> pd.Series:
    if tukey_df.empty:
        return pd.Series(False, index=merged.index)
    if "AEVariant" not in merged.columns:
        top_group = extract_top_tukey_families(tukey_df, desc_df)
        return merged["Family"].isin(top_group) if top_group else pd.Series(False, index=merged.index)

    flags = pd.Series(False, index=merged.index)
    for ae_variant, desc_part in desc_df.groupby("AEVariant", dropna=False):
        tukey_part = tukey_df
        if "AEVariant" in tukey_df.columns:
            tukey_part = tukey_df[tukey_df["AEVariant"].astype(str) == str(ae_variant)]
        top_group = extract_top_tukey_families(tukey_part, desc_part)
        if top_group:
            mask = (merged["AEVariant"].astype(str) == str(ae_variant)) & merged["Family"].isin(top_group)
            flags.loc[mask] = True
    return flags


def _annotate_anova_significance(merged: pd.DataFrame, anova_df: pd.DataFrame, alpha: float) -> pd.Series:
    if anova_df.empty:
        return pd.Series(False, index=merged.index)
    if "AEVariant" not in merged.columns:
        anova_p = float(anova_df["p_value"].iloc[0]) if "p_value" in anova_df.columns else np.nan
        return pd.Series(bool(not pd.isna(anova_p) and anova_p < alpha), index=merged.index)

    flags = pd.Series(False, index=merged.index)
    if "AEVariant" not in anova_df.columns or "p_value" not in anova_df.columns:
        return flags
    for _, row in anova_df.iterrows():
        anova_p = float(row["p_value"]) if not pd.isna(row["p_value"]) else np.nan
        if not pd.isna(anova_p) and anova_p < alpha:
            mask = merged["AEVariant"].astype(str) == str(row["AEVariant"])
            flags.loc[mask] = True
    return flags

def best_by_family_median(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    group_cols = _family_group_cols(df)
    out_rows: List[pd.Series] = []
    for _, group in df.groupby(group_cols, dropna=False):
        values = pd.to_numeric(group[metric], errors="coerce").dropna()
        if values.empty:
            continue
        family_median = float(values.median())
        working = group.copy()
        working["_distance_to_family_median"] = (pd.to_numeric(working[metric], errors="coerce") - family_median).abs()
        picked = working.sort_values(["_distance_to_family_median", metric], ascending=[True, False]).iloc[0].copy()
        picked["family_median_metric"] = family_median
        picked["abs_distance_to_family_median"] = float(picked["_distance_to_family_median"])
        out_rows.append(picked.drop(labels=["_distance_to_family_median"]))
    if not out_rows:
        return pd.DataFrame()
    out = pd.DataFrame(out_rows).sort_values(group_cols).reset_index(drop=True)
    return out


def family_rank_table(desc_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["AEVariant"] if "AEVariant" in desc_df.columns else []
    if group_cols:
        rank_df = desc_df.sort_values(group_cols + ["mean", "median"], ascending=[True, False, False]).copy()
        rank_df["rank"] = rank_df.groupby(group_cols, dropna=False).cumcount() + 1
    else:
        rank_df = desc_df.sort_values(["mean", "median"], ascending=[False, False]).copy().reset_index(drop=True)
        rank_df.insert(0, "rank", np.arange(1, len(rank_df) + 1))
    return rank_df[["rank"] + group_cols + ["Family", "mean", "median", "std", "best", "count"]]


def build_family_summary(
    desc_df: pd.DataFrame,
    rank_df: pd.DataFrame,
    tukey_df: pd.DataFrame,
    anova_df: pd.DataFrame,
    alpha: float,
) -> pd.DataFrame:
    group_cols = ["AEVariant", "Family"] if "AEVariant" in desc_df.columns else ["Family"]
    empty_cols = group_cols + [
        "count",
        "mean",
        "median",
        "std",
        "best",
        "rank",
        "is_top_mean_above_dataset_median",
        "is_in_top_tukey_group",
        "anova_significant",
    ]
    if desc_df.empty:
        return pd.DataFrame(columns=empty_cols)

    merged = desc_df[group_cols + ["count", "mean", "median", "std", "best"]].merge(
        rank_df[group_cols + ["rank"]], on=group_cols, how="left"
    )
    if "AEVariant" in desc_df.columns:
        median_of_means = desc_df.groupby("AEVariant", dropna=False)["mean"].transform("median")
        desc_thresholds = desc_df[group_cols].copy()
        desc_thresholds["_variant_median_of_means"] = median_of_means
        merged = merged.merge(desc_thresholds, on=group_cols, how="left")
        merged["is_top_mean_above_dataset_median"] = merged["mean"] > merged["_variant_median_of_means"]
        merged = merged.drop(columns=["_variant_median_of_means"])
    else:
        dataset_median_of_means = float(desc_df["mean"].median())
        merged["is_top_mean_above_dataset_median"] = merged["mean"] > dataset_median_of_means

    merged["is_in_top_tukey_group"] = _annotate_top_tukey_group(merged, desc_df, tukey_df)
    merged["anova_significant"] = _annotate_anova_significance(merged, anova_df, alpha)
    return merged.sort_values([c for c in ["AEVariant", "rank"] if c in merged.columns] or ["rank"])

def plot_boxplot(df: pd.DataFrame, dataset: str, metric: str, output_dir: Path, suffix: str = "") -> None:
    fams = sorted(df["Family"].dropna().unique())
    data = [df.loc[df["Family"] == f, metric].dropna().values for f in fams]
    if not fams or all(len(v) == 0 for v in data):
        return
    plt.figure(figsize=(12, 6))
    plt.boxplot(data, tick_labels=fams, showfliers=True)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(metric)
    plt.title(f"Boxplot by Family - {dataset} ({metric})")
    plt.tight_layout()
    suffix_part = f"_{suffix}" if suffix else ""
    plt.savefig(output_dir / f"boxplot_{dataset}_{metric}{suffix_part}.jpg", dpi=150)
    plt.close()


def plot_violin(df: pd.DataFrame, dataset: str, metric: str, output_dir: Path, suffix: str = "") -> None:
    preferred_order = [
        "CONST",
        "LINEAR",
        "EXPONENTIAL",
        "COSINE",
        "SIGMOID",
        "LOGISTIC",
        "FIBONACCI",
        "LAPLACE",
        "CAUCHY",
    ]

    available = set(df["Family"].dropna().astype(str).unique())
    fams = [f for f in preferred_order if f in available]
    fams += sorted([f for f in available if f not in preferred_order])

    data = [
        df.loc[df["Family"].astype(str) == f, metric].dropna().values
        for f in fams
    ]

    if not fams or all(len(v) == 0 for v in data):
        return

    plt.figure(figsize=(16, 8))
    plt.violinplot(data, showmeans=True, showmedians=True)

    plt.xticks(
        np.arange(1, len(fams) + 1),
        fams,
        rotation=30,
        ha="right",
        fontsize=13,
    )
    plt.yticks(fontsize=13)
    plt.xlabel("Schedule family", fontsize=14)
    plt.ylabel(metric, fontsize=14)
    plt.title(f"{dataset}: {metric} distribution by schedule family", fontsize=15)

    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    suffix_part = f"_{suffix}" if suffix else ""
    plt.savefig(
        output_dir / f"violin_{dataset}_{metric}{suffix_part}.jpg",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def safe_to_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def sanitize_token(text: str) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text).strip())
    return token.strip("_") or "UNKNOWN"


def ae_variant_values(df: pd.DataFrame) -> List[str]:
    if "AEVariant" not in df.columns:
        return []
    vals = (
        df["AEVariant"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )
    return sorted(vals)


def plot_family_figures_by_variant(
    df: pd.DataFrame,
    dataset: str,
    metric: str,
    output_dir: Path,
    suffix: str,
) -> List[Path]:
    generated: List[Path] = []
    variants = ae_variant_values(df)
    suffix_part = f"_{suffix}" if suffix else ""

    if len(variants) <= 1:
        plot_boxplot(df, dataset, metric, output_dir, suffix=suffix)
        generated.append(output_dir / f"boxplot_{dataset}_{metric}{suffix_part}.jpg")
        plot_violin(df, dataset, metric, output_dir, suffix=suffix)
        generated.append(output_dir / f"violin_{dataset}_{metric}{suffix_part}.jpg")
        return generated

    print(
        "[Plot] Multiple AEVariant values detected. "
        "Generating separate family plots per AEVariant to avoid mixed distributions."
    )
    variant_series = df["AEVariant"].astype(str).str.strip()
    for variant in variants:
        part = df[variant_series == str(variant)].copy()
        if part.empty:
            continue

        variant_token = sanitize_token(variant)
        dataset_variant = f"{dataset}_{variant_token}"

        plot_boxplot(
            part,
            dataset=dataset_variant,
            metric=metric,
            output_dir=output_dir,
            suffix=suffix,
        )
        generated.append(output_dir / f"boxplot_{dataset_variant}_{metric}{suffix_part}.jpg")

        plot_violin(
            part,
            dataset=dataset_variant,
            metric=metric,
            output_dir=output_dir,
            suffix=suffix,
        )
        generated.append(output_dir / f"violin_{dataset_variant}_{metric}{suffix_part}.jpg")


    return generated


def summarize_seed_coverage(df: pd.DataFrame) -> Dict[str, Dict[str, object]]:
    out: Dict[str, Dict[str, object]] = {}
    if "Seed" not in df.columns:
        return out
    for ds, g in df.groupby("Dataset"):
        seeds = pd.to_numeric(g["Seed"], errors="coerce")
        seed_values = sorted(seeds.dropna().astype(int).unique().tolist())
        rows_per_seed = g.assign(_seed=seeds).dropna(subset=["_seed"]).groupby("_seed").size().to_dict()
        rows_per_seed = {str(int(k)): int(v) for k, v in rows_per_seed.items()}
        out[str(ds)] = {
            "unique_seed_count": len(seed_values),
            "seeds": seed_values,
            "rows_per_seed": rows_per_seed,
        }
    return out


def summarize_noise_type_coverage(df: pd.DataFrame) -> Dict[str, Dict[str, object]]:
    out: Dict[str, Dict[str, object]] = {}
    if "NoiseType" not in df.columns:
        return out
    for ds, g in df.groupby("Dataset"):
        types = sorted(g["NoiseType"].astype(str).str.lower().unique().tolist())
        out[str(ds)] = {
            "unique_noise_type_count": len(types),
            "noise_types": types,
        }
    return out


def summarize_architecture_coverage(df: pd.DataFrame) -> Dict[str, Dict[str, object]]:
    out: Dict[str, Dict[str, object]] = {}
    if "Architecture" not in df.columns:
        return out
    for ds, g in df.groupby("Dataset"):
        vals = sorted(g["Architecture"].astype(str).unique().tolist())
        out[str(ds)] = {
            "unique_architecture_count": len(vals),
            "architectures": vals,
        }
    return out


def summarize_runtime_config_coverage(df: pd.DataFrame) -> Dict[str, int]:
    out: Dict[str, int] = {}
    if "RuntimeConfigID" not in df.columns:
        return out
    for ds, g in df.groupby("Dataset"):
        out[str(ds)] = int(g["RuntimeConfigID"].nunique())
    return out



def holm_correction(pvals: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=pvals.index, dtype=float)
    valid = pvals.dropna()
    m = len(valid)
    if m == 0:
        return out
    ordered = valid.sort_values()
    running_max = 0.0
    for i, (idx, p) in enumerate(ordered.items(), start=1):
        adj = min(1.0, (m - i + 1) * float(p))
        running_max = max(running_max, adj)
        out.loc[idx] = running_max
    return out


def bh_correction(pvals: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=pvals.index, dtype=float)
    valid = pvals.dropna()
    m = len(valid)
    if m == 0:
        return out
    ordered = valid.sort_values(ascending=False)
    running_min = 1.0
    for rank_desc, (idx, p) in enumerate(ordered.items(), start=1):
        rank = m - rank_desc + 1
        adj = min(1.0, (m / rank) * float(p))
        running_min = min(running_min, adj)
        out.loc[idx] = running_min
    return out


def bootstrap_ci(values: np.ndarray, stat_fn, n_boot: int = 1000, alpha: float = 0.05, seed: int = 42) -> Tuple[float, float]:
    vals = pd.to_numeric(pd.Series(values), errors="coerce").dropna().values
    if vals.size == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot, dtype=float)
    n = vals.size
    for i in range(n_boot):
        sample = vals[rng.integers(0, n, n)]
        stats[i] = float(stat_fn(sample))
    lo = float(np.quantile(stats, alpha / 2.0))
    hi = float(np.quantile(stats, 1.0 - alpha / 2.0))
    return lo, hi


def _paired_inference_stats(delta: pd.Series, min_pairs: int = 6) -> Dict[str, object]:
    d = pd.to_numeric(delta, errors="coerce").dropna()
    n = int(d.shape[0])
    if n < min_pairs:
        return {
            "test_name": np.nan,
            "p_value": np.nan,
            "effect_size": np.nan,
            "improves_vs_const": np.nan,
            "inferential_note": f"insufficient_pairs(n={n}, min_required={min_pairs})",
        }
    non_zero = d[d != 0]
    if non_zero.empty:
        return {
            "test_name": "all_zero_deltas",
            "p_value": 1.0,
            "effect_size": 0.0,
            "improves_vs_const": False,
            "inferential_note": "all_pairwise_deltas_equal_zero",
        }
    try:
        w = wilcoxon(d, zero_method="wilcox", alternative="greater", mode="auto")
        effect = float(non_zero.mean() / (non_zero.std(ddof=1) + 1e-12))
        return {
            "test_name": "wilcoxon_signed_rank_greater",
            "p_value": float(w.pvalue),
            "effect_size": effect,
            "improves_vs_const": bool((w.pvalue < 0.05) and (d.mean() > 0)),
            "inferential_note": "",
        }
    except Exception:
        pos = int((d > 0).sum())
        neg = int((d < 0).sum())
        n_eff = pos + neg
        p_value = 1.0 if n_eff == 0 else float(binomtest(pos, n=n_eff, p=0.5, alternative="greater").pvalue)
        effect = float((pos - neg) / max(1, n_eff))
        return {
            "test_name": "sign_test_greater",
            "p_value": p_value,
            "effect_size": effect,
            "improves_vs_const": bool((p_value < 0.05) and (d.mean() > 0)),
            "inferential_note": "wilcoxon_unavailable_or_invalid_used_sign_test",
        }


def paired_family_summary(
    df: pd.DataFrame,
    delta_metric_col: str,
    metric_name: str,
    paired_delta_basis: str,
    min_pairs: int = 6,
) -> pd.DataFrame:
    rows = []
    mean_col = f"mean_delta_{metric_name}"
    median_col = f"median_delta_{metric_name}"
    std_col = f"std_delta_{metric_name}"
    min_col = f"min_delta_{metric_name}"
    max_col = f"max_delta_{metric_name}"
    mean_ci_low_col = f"mean_delta_{metric_name}_ci_low"
    mean_ci_high_col = f"mean_delta_{metric_name}_ci_high"
    median_ci_low_col = f"median_delta_{metric_name}_ci_low"
    median_ci_high_col = f"median_delta_{metric_name}_ci_high"

    if "AEVariant" in df.columns:
        grouped = df.groupby(["AEVariant", "Family"])
    else:
        grouped = df.groupby("Family")
    for group_key, g in grouped:
        if "AEVariant" in df.columns:
            ae_variant, family = group_key
        else:
            ae_variant, family = None, group_key
        d = pd.to_numeric(g[delta_metric_col], errors="coerce").dropna()
        n = int(d.shape[0])
        if n == 0:
            continue
        infer = _paired_inference_stats(d, min_pairs=min_pairs)
        mean_ci_low, mean_ci_high = bootstrap_ci(d.values, np.mean)
        median_ci_low, median_ci_high = bootstrap_ci(d.values, np.median)
        row = {
            "Family": family,
            "AEVariant": ae_variant,
            "metric": metric_name,
            "delta_column": delta_metric_col,
            "paired_delta_basis_used": paired_delta_basis,
            "n_pairs": n,
            mean_col: float(d.mean()),
            median_col: float(d.median()),
            std_col: float(d.std(ddof=1)) if n > 1 else 0.0,
            min_col: float(d.min()),
            max_col: float(d.max()),
            "positive_delta_count": int((d > 0).sum()),
            "negative_delta_count": int((d < 0).sum()),
            "zero_delta_count": int((d == 0).sum()),
            "win_rate_vs_const": float((d > 0).mean()),
            "test_name": infer["test_name"],
            "p_value": infer["p_value"],
            "effect_size": infer["effect_size"],
            "improves_vs_const": infer["improves_vs_const"],
            "inferential_note": infer["inferential_note"],
            mean_ci_low_col: mean_ci_low,
            mean_ci_high_col: mean_ci_high,
            median_ci_low_col: median_ci_low,
            median_ci_high_col: median_ci_high,
        }
        for extra_metric in METRIC_CHOICES:
            extra_col = f"delta_{extra_metric}_vs_const_{paired_delta_basis}"
            if extra_col in g.columns:
                row[f"mean_delta_{extra_metric}"] = float(pd.to_numeric(g[extra_col], errors="coerce").mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(mean_col, ascending=False) if rows else pd.DataFrame()


def run_paired_mode(args: argparse.Namespace, input_dir: Path, output_dir: Path) -> None:
    try:
        all_test_df = load_run_metric_files(
            output_root=input_dir,
            metric_filename="final_test_metrics_seed*.csv",
            seeds=args.seeds,
        )
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return

    print(f"Loaded {len(find_run_metric_files(input_dir, 'final_test_metrics_seed*.csv'))} final_test_metrics_seed*.csv files")
    print(f"Loaded {len(all_test_df)} total test metric rows")

    all_test_df, filter_note = filter_by_seed_noise_and_ae_variant(
        all_test_df, seeds=args.seeds, noise_types=args.noise_types, ae_variants=args.ae_variants
    )
    print(f"[INFO] paired_extra_filter={filter_note}")
    if args.datasets:
        all_test_df = all_test_df[all_test_df["Dataset"].astype(str).isin([str(d) for d in args.datasets])].copy()

    paired_df, _, _ = build_independent_paired_decay_vs_const_df(all_test_df)
    print(f"Built {len(paired_df)} paired CONST vs decay rows")
    if paired_df.empty:
        print('[ERROR] No paired rows produced from final_test_metrics_seed*.csv inputs.')
        return

    independent_outputs = []
    for dataset, ds_paired in paired_df.groupby("Dataset"):
        dataset_token = sanitize_token(dataset)
        dataset_paired_dir = output_dir / dataset_token / "paired"
        dataset_paired_dir.mkdir(parents=True, exist_ok=True)
        dataset_paired_out = dataset_paired_dir / f"paired_decay_vs_const_independent_{dataset_token}.csv"
        safe_to_csv(ds_paired, dataset_paired_out)
        independent_outputs.append(dataset_paired_out)
    if not (args.datasets and len(args.datasets) == 1):
        all_paired_dir = output_dir / "ALL" / "paired"
        all_paired_dir.mkdir(parents=True, exist_ok=True)
        all_paired_out = all_paired_dir / "paired_decay_vs_const_independent_ALL.csv"
        safe_to_csv(paired_df, all_paired_out)
        independent_outputs.append(all_paired_out)
    print('Saved paired CONST vs decay output to:')
    for paired_out in independent_outputs:
        print(f'  {paired_out}')

    all_df = paired_df
    selected_delta_col = f"delta_{args.metric}_vs_const_{args.paired_delta_basis}"
    if selected_delta_col not in all_df.columns:
        raise ValueError(
            f"Requested paired metric '{args.metric}' with delta basis '{args.paired_delta_basis}' "
            f"is unavailable in loaded paired files: missing '{selected_delta_col}'."
        )

    per_dataset_summary_frames = []
    for dataset, ds in all_df.groupby("Dataset"):
        summary = paired_family_summary(
            ds,
            delta_metric_col=selected_delta_col,
            metric_name=args.metric,
            paired_delta_basis=args.paired_delta_basis,
            min_pairs=max(2, int(args.min_group_size)),
        )
        if summary.empty:
            continue
        summary["holm_p_value"] = holm_correction(summary["p_value"])
        summary["bh_p_value"] = bh_correction(summary["p_value"])
        summary["significant_holm"] = summary["holm_p_value"] < args.alpha
        summary["significant_bh"] = summary["bh_p_value"] < args.alpha
        dataset_paired_dir = output_dir / sanitize_token(dataset) / "paired"
        dataset_paired_dir.mkdir(parents=True, exist_ok=True)
        out_path = dataset_paired_dir / f"paired_decay_vs_const_summary_{sanitize_token(dataset)}_{args.metric}.csv"
        safe_to_csv(summary, out_path)
        per_dataset_summary_frames.append(summary.assign(Dataset=str(dataset)))

    if not per_dataset_summary_frames:
        print("[ERROR] No per-dataset paired summaries produced.")
        return
    combined_summary = pd.concat(per_dataset_summary_frames, ignore_index=True)


    print(
        f"[INFO] Paired mode complete. Output directory: {output_dir}. "
        f"Paired delta basis used: {args.paired_delta_basis} ({selected_delta_col})."
    )


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_root_dir = Path(args.output_dir)
    output_root_dir.mkdir(parents=True, exist_ok=True)
    paired_output_dir = output_root_dir
    family_output_dir = output_root_dir

    print(f"[INFO] Statistical analysis mode: {args.analysis_mode}")
    print(
        "[INFO] Paired mode is primary; family mode is secondary. "
        "Paired inference baseline collapse is configurable via --paired_delta_basis (median|mean)."
    )

    if args.analysis_mode in {"paired", "both"}:
        paired_output_dir.mkdir(parents=True, exist_ok=True)
        run_paired_mode(args=args, input_dir=input_dir, output_dir=paired_output_dir)
    if args.analysis_mode in {"paired", "family", "both"}:
        manifest_path = output_root_dir / "analysis_manifest.json"
        manifest = {
            "analysis_mode": args.analysis_mode,
            "output_layout": {
                "dataset_paired_dir": str(output_root_dir / "<DATASET>" / "paired"),
                "dataset_family_dir": str(output_root_dir / "<DATASET>" / "family"),
                "combined_dir": str(output_root_dir / "ALL"),
                "manifest": str(manifest_path),
            },
            "paired_output_dir": str(paired_output_dir),
            "family_output_dir": str(family_output_dir),
            "selected_ae_variants": args.ae_variants,
            "paired_delta_basis": args.paired_delta_basis,
            "family_unit": args.family_unit,
            "metric": args.metric,
            "datasets": args.datasets,
            "seeds": args.seeds,
            "noise_types": args.noise_types,
            "alpha": args.alpha,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if args.analysis_mode == "paired":
        return

    output_dir = family_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        all_df = load_run_metric_files(
            output_root=input_dir,
            metric_filename="final_test_metrics_seed*.csv",
            seeds=args.seeds,
        )
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return
    total_rows_loaded = len(all_df)

    metric = args.metric
    if metric not in all_df.columns:
        print(f"[ERROR] Metric column '{metric}' not found. Available columns: {list(all_df.columns)}")
        return

    all_df[metric] = pd.to_numeric(all_df[metric], errors="coerce")
    all_df = all_df.dropna(subset=[metric])
    if all_df.empty:
        print(f"[ERROR] No valid numeric rows for metric '{metric}' after coercion.")
        return
    rows_after_metric_cleanup = len(all_df)

    all_df, filter_note = filter_by_seed_noise_and_ae_variant(
        all_df, seeds=args.seeds, noise_types=args.noise_types, ae_variants=args.ae_variants
    )
    print(f"[INFO] extra_filter={filter_note}")
    if all_df.empty:
        print("[ERROR] No rows remained after seed/noise filtering.")
        return
    rows_after_seed_noise_filter = len(all_df)

    all_df, dedup_mode, dedup_removed = apply_deduplication(all_df, enabled=args.deduplicate)
    if all_df.empty:
        print("[ERROR] No rows remained after deduplication.")
        return
    rows_after_deduplication = len(all_df)

    detected_datasets = sorted(all_df["Dataset"].astype(str).unique().tolist())
    selected_datasets = args.datasets if args.datasets else detected_datasets

    generated_files: List[Path] = []
    processed_datasets: List[str] = []
    analysis_mode_global: Optional[str] = None
    analyzed_frames: List[pd.DataFrame] = []
    dataset_row_counts: Dict[str, int] = {}
    dataset_analyzed_counts: Dict[str, int] = {}
    dataset_family_counts: Dict[str, int] = {}
    family_level_tests_run = 0
    filtered_out_datasets = 0

    for dataset in selected_datasets:
        ds_df = all_df[all_df["Dataset"].astype(str) == str(dataset)].copy()
        if ds_df.empty:
            print(f"[WARN] Dataset '{dataset}' not found in loaded rows. Skipping.")
            continue

        dataset_row_counts[dataset] = len(ds_df)
        ds_df, analysis_mode = filter_by_family_mode(ds_df, noise_only=args.noise_only)
        analysis_mode_global = analysis_mode
        if ds_df.empty:
            family_unit_token = sanitize_token(str(args.family_unit))
            filtered_out_datasets += 1
            dataset_analyzed_counts[dataset] = 0
            dataset_family_counts[dataset] = 0
            print(
                f"[WARN] Dataset '{dataset}' has no eligible rows after filtering "
                f"(analysis mode: {analysis_mode}). Skipping."
            )
            continue

        processed_datasets.append(dataset)
        dataset_token = sanitize_token(dataset)
        dataset_family_dir = output_dir / dataset_token / "family"
        dataset_family_dir.mkdir(parents=True, exist_ok=True)
        ds_df = ds_df.sort_values(metric, ascending=False)
        analyzed_frames.append(ds_df.copy())
        dataset_analyzed_counts[dataset] = len(ds_df)
        dataset_family_counts[dataset] = int(ds_df["Family"].nunique())

        family_unit = str(args.family_unit)
        # family_work_df is used for inferential tests; family_summary_input_df drives tables/plots.
        family_work_df = ds_df
        family_metric = metric
        family_unit_used = family_unit
        family_unit_token = sanitize_token(family_unit_used)
        family_summary_input_df = ds_df.copy()
        if family_unit in {"seed_mean", "seed_median"}:
            agg = "mean" if family_unit == "seed_mean" else "median"
            collapsed = build_seed_collapsed_family_df(ds_df, metric=metric, agg=agg)
            family_work_df = collapsed.copy()
            family_work_df[metric] = family_work_df["FamilyMetricCollapsed"]
            family_summary_input_df = family_work_df.copy()
            family_summary_input_df[metric] = pd.to_numeric(
                family_summary_input_df["FamilyMetricCollapsed"], errors="coerce"
            )
            family_metric = metric
        # Keep summaries/plots aligned to the same family input table.

        analysis_input_path = dataset_family_dir / f"analysis_input_{dataset}_{metric}_{family_unit_token}.csv"
        safe_to_csv(family_summary_input_df, analysis_input_path)
        generated_files.append(analysis_input_path)

        desc = descriptive_stats(family_summary_input_df, metric)


        anova_frames: List[pd.DataFrame] = []
        tukey_frames: List[pd.DataFrame] = []
        if "AEVariant" in family_work_df.columns:
            anova_variant_items = [
                (str(ae_variant), part.copy())
                for ae_variant, part in family_work_df.groupby("AEVariant", dropna=False)
            ]
        else:
            anova_variant_items = [(None, family_work_df.copy())]

        for ae_variant, variant_work_df in anova_variant_items:
            variant_token = sanitize_token(ae_variant) if ae_variant is not None else None
            file_variant_part = f"_{variant_token}" if variant_token else ""
            anova_df, _, eligible_df, anova_ran = run_anova(
                variant_work_df,
                dataset,
                family_metric,
                args.min_group_size,
                ae_variant=ae_variant,
            )
            if not anova_df.empty:
                anova_df.insert(0, "family_unit_used", family_unit_used)
            anova_frames.append(anova_df)
            if anova_ran:
                family_level_tests_run += 1
            anova_path = dataset_family_dir / f"anova_{dataset}{file_variant_part}_{metric}_{family_unit_token}.csv"
            safe_to_csv(anova_df, anova_path)
            generated_files.append(anova_path)

            tukey_path = dataset_family_dir / f"tukey_{dataset}{file_variant_part}_{metric}_{family_unit_token}.csv"
            if not anova_ran:
                tukey_df = empty_tukey_df(ae_variant=ae_variant, family_unit_used=family_unit_used)
                print("[INFO] Tukey HSD skipped because ANOVA was not run.")
            elif pairwise_tukeyhsd is None:
                tukey_df = empty_tukey_df(ae_variant=ae_variant, family_unit_used=family_unit_used)
                print("[INFO] Tukey HSD skipped because statsmodels is unavailable.")
            elif eligible_df["Family"].nunique() < 2:
                tukey_df = empty_tukey_df(ae_variant=ae_variant, family_unit_used=family_unit_used)
                print("[INFO] Tukey HSD skipped because fewer than 2 eligible families remained.")
            else:
                tukey_df, tukey_note = run_tukey(eligible_df, family_metric, alpha=args.alpha)
                if tukey_note:
                    print(f"[INFO] {tukey_note}")
                if tukey_df.empty:
                    tukey_df = empty_tukey_df(ae_variant=ae_variant, family_unit_used=family_unit_used)
                    if not tukey_note:
                        print("[INFO] Tukey HSD produced no comparisons.")
            if ae_variant is not None and "AEVariant" not in tukey_df.columns:
                tukey_df.insert(0, "AEVariant", normalize_ae_variant(ae_variant))
            if not tukey_df.empty and "family_unit_used" not in tukey_df.columns:
                tukey_df.insert(0, "family_unit_used", family_unit_used)
            tukey_frames.append(tukey_df)
            safe_to_csv(tukey_df, tukey_path)
            generated_files.append(tukey_path)

        anova_df = pd.concat(anova_frames, ignore_index=True) if anova_frames else pd.DataFrame()
        tukey_df = pd.concat(tukey_frames, ignore_index=True) if tukey_frames else empty_tukey_df(family_unit_used=family_unit_used)
        rank_df = family_rank_table(desc)

        family_summary = build_family_summary(
            desc_df=desc,
            rank_df=rank_df,
            tukey_df=tukey_df,
            anova_df=anova_df,
            alpha=args.alpha,
        )
        if not family_summary.empty:
            family_summary.insert(0, "family_unit_used", family_unit_used)
        if not args.no_plots:
            try:
                generated_files.extend(
                    plot_family_figures_by_variant(
                        df=family_summary_input_df,
                        dataset=dataset_token,
                        metric=metric,
                        output_dir=dataset_family_dir,
                        suffix=family_unit_token,
                    )
                )
            except Exception as exc:
                print(f"[WARN] Plot generation failed for dataset={dataset}: {exc}")

    all_filtered_analyzed_df = pd.concat(analyzed_frames, ignore_index=True) if analyzed_frames else pd.DataFrame()
    rows_after_dataset_family_filter = len(all_filtered_analyzed_df)

    seed_cov = summarize_seed_coverage(all_filtered_analyzed_df)
    noise_cov = summarize_noise_type_coverage(all_filtered_analyzed_df)
    arch_cov = summarize_architecture_coverage(all_filtered_analyzed_df)
    runtime_cov = summarize_runtime_config_coverage(all_filtered_analyzed_df)

    print("-" * 88)
    print("[INFO] Analysis summary (compact):")
    print(f"[INFO]   analysis_mode={analysis_mode_global or 'N/A'}")
    print(f"[INFO]   metric={metric}")
    print(f"[INFO]   noise_only={args.noise_only}")
    print(f"[INFO]   selected_noise_types={args.noise_types if args.noise_types is not None else 'ALL'}")
    print(f"[INFO]   selected_ae_variants={args.ae_variants if args.ae_variants is not None else 'ALL'}")
    print(f"[INFO]   selected_seeds={args.seeds if args.seeds is not None else 'ALL'}")
    print(f"[INFO]   deduplication_enabled={args.deduplicate}")
    print(f"[INFO]   deduplication_mode={dedup_mode}")
    print(f"[INFO]   rows_removed_by_deduplication={dedup_removed}")
    print(f"[INFO]   total_rows_loaded={total_rows_loaded}")
    print(f"[INFO]   rows_after_metric_cleanup={rows_after_metric_cleanup}")
    print(f"[INFO]   rows_after_seed_noise_filter={rows_after_seed_noise_filter}")
    print(f"[INFO]   rows_after_deduplication={rows_after_deduplication}")
    print(f"[INFO]   rows_after_dataset_family_filter={rows_after_dataset_family_filter}")
    print(f"[INFO]   total_rows_analyzed={len(all_filtered_analyzed_df)}")
    print(f"[INFO]   family_level_tests_run={family_level_tests_run}")
    print(f"[INFO]   datasets_processed={processed_datasets}")
    for ds in processed_datasets:
        print(
            f"[INFO]   dataset={ds}: rows_before_family_filter={dataset_row_counts.get(ds, 0)}, "
            f"rows_after_filter={dataset_analyzed_counts.get(ds, 0)}, families={dataset_family_counts.get(ds, 0)}, "
            f"unique_seeds={seed_cov.get(ds, {}).get('unique_seed_count', 'N/A')}, "
            f"rows_per_seed={seed_cov.get(ds, {}).get('rows_per_seed', 'N/A')}, "
            f"noise_types={noise_cov.get(ds, {}).get('noise_types', 'N/A')}, "
            f"unique_architectures={arch_cov.get(ds, {}).get('unique_architecture_count', 'N/A')}, "
            f"unique_runtime_configs={runtime_cov.get(ds, 'N/A')}"
        )
    if filtered_out_datasets > 0:
        print(f"[INFO]   datasets_filtered_out_after_family_mode={filtered_out_datasets}")
    print(f"[INFO] Output directory: {output_dir}")
    print(f"[INFO] Generated file count: {len(generated_files)}")
    for path in generated_files:
        print(f"  - {path}")


if __name__ == "__main__":
    main()




