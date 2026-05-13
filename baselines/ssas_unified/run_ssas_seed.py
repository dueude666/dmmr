import argparse
import json
import os
import random
import shutil
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_DIR = REPO_ROOT / "baselines" / "ssas_official" / "SSAS"
UNIFIED_DIR = REPO_ROOT / "baselines" / "ssas_unified"
DEFAULT_DATA_ROOT = REPO_ROOT.parents[0] / "data" / "seed3_repo_layout" / "ExtractedFeatures"


def install_pyriemann_import_stub() -> None:
    """Official SSAS imports pyriemann utilities even when the active path does not use them."""
    if "pyriemann" in sys.modules:
        return

    pyriemann = types.ModuleType("pyriemann")
    tangentspace = types.ModuleType("pyriemann.tangentspace")
    utils_pkg = types.ModuleType("pyriemann.utils")
    covariance_mod = types.ModuleType("pyriemann.utils.covariance")
    mean_mod = types.ModuleType("pyriemann.utils.mean")

    class TangentSpace:
        def fit(self, x):
            return self

        def transform(self, x):
            return np.asarray(x).reshape(np.asarray(x).shape[0], -1)

    def covariances(x, estimator="cov"):
        arr = np.asarray(x)
        covs = []
        for sample in arr:
            cov = np.cov(sample)
            covs.append(cov + np.eye(cov.shape[0]) * 1e-6)
        return np.asarray(covs)

    def mean_covariance(covs, metric="riemann"):
        return np.mean(np.asarray(covs), axis=0)

    tangentspace.TangentSpace = TangentSpace
    covariance_mod.covariances = covariances
    mean_mod.mean_covariance = mean_covariance
    utils_pkg.covariance = covariance_mod
    utils_pkg.mean = mean_mod
    pyriemann.tangentspace = tangentspace
    pyriemann.utils = utils_pkg

    sys.modules["pyriemann"] = pyriemann
    sys.modules["pyriemann.tangentspace"] = tangentspace
    sys.modules["pyriemann.utils"] = utils_pkg
    sys.modules["pyriemann.utils.covariance"] = covariance_mod
    sys.modules["pyriemann.utils.mean"] = mean_mod


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_flat_seed_session(data_root: Path, session: int, cache_root: Path) -> Path:
    data_root = data_root.resolve()
    session_dir = data_root / str(session)
    if not data_root.exists():
        raise FileNotFoundError(f"SEED data root not found: {data_root}")
    if not session_dir.exists():
        raise FileNotFoundError(f"SEED session directory not found: {session_dir}")
    label_file = data_root / "label.mat"
    if not label_file.exists():
        raise FileNotFoundError(f"SEED label.mat not found: {label_file}")

    flat_dir = (cache_root / f"session_{session}").resolve()
    flat_dir.mkdir(parents=True, exist_ok=True)
    _link_or_copy(label_file, flat_dir / "label.mat")
    for mat_file in session_dir.glob("*.mat"):
        _link_or_copy(mat_file, flat_dir / mat_file.name)
    return flat_dir


def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def build_official_args(cli_args, fold_output_dir: Path, fold_log_file):
    return SimpleNamespace(
        baseline="MSTN",
        device=cli_args.device,
        gpu_id=cli_args.gpu_id,
        dataset="seed",
        source="seed_sources",
        target=1,
        iteration=1,
        test_interval=1,
        snapshot_interval=1000,
        output_dir=str(fold_output_dir),
        mixed_sessions="per_session",
        lr_a=cli_args.lr_a,
        lr_b=cli_args.lr_b,
        radius=cli_args.radius,
        num_class=3,
        num_class2=14,
        stages=1,
        max_iter1=cli_args.max_iter1,
        max_iter2=cli_args.max_iter2,
        max_iter3=cli_args.max_iter3,
        batch_size=cli_args.batch_size,
        seed=cli_args.seed,
        bottleneck_dim=cli_args.bottleneck_dim,
        session=cli_args.session,
        gamma=cli_args.gamma,
        use_feature_mixstyle=cli_args.use_feature_mixstyle,
        mixstyle_p=cli_args.mixstyle_p,
        mixstyle_alpha=cli_args.mixstyle_alpha,
        file_path=str(cli_args.flat_session_dir) + os.sep,
        log_file=fold_log_file,
        ila_switch_iter=1,
        n_samples=2,
        mu=80,
        k=3,
        msc_coeff=1.0,
        count_dir=str(fold_output_dir / "count"),
    )


def run_fold(cli_args, target_subject_zero_based: int):
    for path in (str(UNIFIED_DIR), str(OFFICIAL_DIR)):
        if path not in sys.path:
            sys.path.insert(0, path)

    install_pyriemann_import_stub()
    if not hasattr(np, "float"):
        np.float = float  # compatibility with official SSAS utils.py on new NumPy

    from selection_domain_new import new_SSDA
    from solvers import NEW_DGMA

    fold_id = target_subject_zero_based + 1
    fold_output_dir = Path(cli_args.output_dir) / f"target_{fold_id:02d}"
    fold_output_dir.mkdir(parents=True, exist_ok=True)
    fold_log_path = fold_output_dir / "official_train_log.txt"

    with fold_log_path.open("w", encoding="utf-8", errors="replace") as fold_log_file:
        official_args = build_official_args(cli_args, fold_output_dir, fold_log_file)
        official_args.target = fold_id
        print(f"[SSAS] target_subject={fold_id} max_iter1={official_args.max_iter1} max_iter2={official_args.max_iter2}")
        x, y, domain_labels, _, _, count_num, net_f = new_SSDA(official_args)
        if cli_args.use_source_weight_calibration:
            count_num = calibrate_source_weights(
                count_num=count_num,
                target_subject_zero_based=target_subject_zero_based,
                blend=cli_args.source_weight_blend,
                temperature=cli_args.source_weight_temperature,
            )
            print("[SSAS][source_weight_calibration]", np.array2string(count_num, precision=4))
        _, _, final_acc, final_f1, final_auc, final_mat, _, last_acc = NEW_DGMA(
            x, y, domain_labels, count_num, net_f, official_args
        )

    result = {
        "target_subject": fold_id,
        "final_acc": float(final_acc),
        "last_acc": float(last_acc),
        "final_f1": float(final_f1),
        "final_auc": float(final_auc),
        "log_path": str(fold_log_path),
    }
    print("[SSAS][fold_result]", json.dumps(result, ensure_ascii=False))
    return result


def calibrate_source_weights(count_num, target_subject_zero_based: int, blend: float, temperature: float):
    """Smooth SSAS source-selection weights to reduce pseudo-label overconfidence."""
    weights = np.asarray(count_num, dtype=np.float64).copy()
    source_mask = np.ones_like(weights, dtype=bool)
    source_mask[target_subject_zero_based] = False
    source_weights = np.clip(weights[source_mask], 1e-6, None)
    log_w = np.log(source_weights) / max(temperature, 1e-6)
    exp_w = np.exp(log_w - log_w.max())
    calibrated = exp_w / exp_w.mean()
    weights[source_mask] = (1.0 - blend) + blend * calibrated
    weights[target_subject_zero_based] = 1.0
    return weights.astype(np.float32)


def parse_args():
    parser = argparse.ArgumentParser(description="Unified SSAS SEED runner")
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--session", type=int, default=1)
    parser.add_argument("--subject_start", type=int, default=0)
    parser.add_argument("--subject_end", type=int, default=3)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--exp_name", type=str, default="quick_ssas_seed")
    parser.add_argument("--output_dir", type=Path, default=REPO_ROOT / "outputs" / "ssas" / "quick_ssas_seed")
    parser.add_argument("--log_dir", type=Path, default=REPO_ROOT / "logs")
    parser.add_argument("--flat_cache", type=Path, default=REPO_ROOT / "baselines" / "ssas_unified" / ".seed_flat_cache")
    parser.add_argument("--max_iter1", type=int, default=20)
    parser.add_argument("--max_iter2", type=int, default=35)
    parser.add_argument("--max_iter3", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--bottleneck_dim", type=int, default=128)
    parser.add_argument("--lr_a", type=float, default=0.1)
    parser.add_argument("--lr_b", type=float, default=0.1)
    parser.add_argument("--radius", type=float, default=10.0)
    parser.add_argument("--gamma", type=int, default=1)
    parser.add_argument("--use_source_weight_calibration", action="store_true")
    parser.add_argument("--source_weight_blend", type=float, default=0.3)
    parser.add_argument("--source_weight_temperature", type=float, default=2.0)
    parser.add_argument("--use_feature_mixstyle", action="store_true")
    parser.add_argument("--mixstyle_p", type=float, default=0.5)
    parser.add_argument("--mixstyle_alpha", type=float, default=0.1)
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available() and args.device == "cuda":
        raise RuntimeError("Official SSAS uses .cuda() directly, but CUDA is not available in this Python environment.")
    if args.subject_start < 0 or args.subject_end > 15 or args.subject_start >= args.subject_end:
        raise ValueError("--subject_start/--subject_end must define a non-empty 0-based range within [0, 15].")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    set_seed(args.seed)
    args.output_dir = (args.output_dir / args.exp_name).resolve()
    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.flat_session_dir = ensure_flat_seed_session(args.data_root, args.session, args.flat_cache)

    print("[SSAS] repo_root:", REPO_ROOT)
    print("[SSAS] official_dir:", OFFICIAL_DIR)
    print("[SSAS] data_root:", args.data_root.resolve())
    print("[SSAS] flat_session_dir:", args.flat_session_dir)
    print("[SSAS] output_dir:", args.output_dir)
    print("[SSAS] protocol:", {
        "session": args.session,
        "subject_start": args.subject_start,
        "subject_end": args.subject_end,
        "seed": args.seed,
        "max_iter1": args.max_iter1,
        "max_iter2": args.max_iter2,
        "batch_size": args.batch_size,
    })

    results = [run_fold(args, subject) for subject in range(args.subject_start, args.subject_end)]
    acc = [item["final_acc"] for item in results]
    summary = {
        "exp_name": args.exp_name,
        "each_acc": acc,
        "avg_acc": float(np.mean(acc)) if acc else None,
        "std_acc": float(np.std(acc)) if acc else None,
        "results": results,
    }
    summary_path = args.log_dir / f"{args.exp_name}_summary.json"
    summary_txt = args.log_dir / f"{args.exp_name}_summary.txt"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    summary_txt.write_text(
        "final each acc: " + ",".join(f"{x:.10f}" for x in acc) + "\n"
        + f"final acc avg: {summary['avg_acc']:.10f}\n"
        + f"final acc std: {summary['std_acc']:.10f}\n",
        encoding="utf-8",
    )
    print("[SSAS][summary]", json.dumps(summary, ensure_ascii=False))
    print("[SSAS] summary_path:", summary_path)
    print("[SSAS] summary_txt:", summary_txt)


if __name__ == "__main__":
    main()
