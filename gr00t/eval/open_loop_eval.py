from copy import deepcopy
from dataclasses import dataclass, field
import logging
from pathlib import Path
import re
from typing import Any
import warnings

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy import BasePolicy
from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.policy.server_client import PolicyClient
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import tyro


warnings.simplefilter("ignore", category=FutureWarning)

"""
Example commands:

NOTE: provide --model_path to load up the model checkpoint in this script,
        else it will use the default host and port via RobotInferenceClient

"""


def plot_trajectory_results(
    state_joints_across_time: np.ndarray,
    gt_action_across_time: np.ndarray,
    pred_action_across_time: np.ndarray,
    traj_id: int,
    state_keys: list[str],
    action_keys: list[str],
    action_horizon: int,
    save_plot_path: str,
) -> None:
    """
    Plot and save trajectory results comparing ground truth and predicted actions.

    Args:
        state_joints_across_time: Array of state joints over time
        gt_action_across_time: Ground truth actions over time
        pred_action_across_time: Predicted actions over time
        traj_id: Trajectory ID
        state_keys: List of state modality keys
        action_keys: List of action modality keys
        action_horizon: Action horizon used for inference
        save_plot_path: Path to save the plot
    """
    actual_steps = len(gt_action_across_time)
    action_dim = gt_action_across_time.shape[1]

    indices_to_plot = list(range(action_dim))

    num_plots = len(indices_to_plot)
    if num_plots == 0:
        logging.warning("No valid indices to plot")
        return

    # Always plot and save
    fig, axes = plt.subplots(nrows=num_plots, ncols=1, figsize=(8, 4 * num_plots))

    # Handle case where there's only one subplot
    if num_plots == 1:
        axes = [axes]

    # Add a global title showing the modality keys
    fig.suptitle(
        f"Trajectory {traj_id} - State: {', '.join(state_keys)} | Action: {', '.join(action_keys)}",
        fontsize=16,
        color="blue",
    )

    for plot_idx, action_idx in enumerate(indices_to_plot):
        ax = axes[plot_idx]

        # The dimensions of state_joints and action are the same
        # only when the robot uses actions directly as joint commands.
        # Therefore, do not plot them if this is not the case.
        if state_joints_across_time.shape == gt_action_across_time.shape:
            ax.plot(state_joints_across_time[:, action_idx], label="state joints")
        ax.plot(gt_action_across_time[:, action_idx], label="gt action")
        ax.plot(pred_action_across_time[:, action_idx], label="pred action")

        # put a dot every ACTION_HORIZON
        for j in range(0, actual_steps, action_horizon):
            if j == 0:
                ax.plot(
                    j,
                    gt_action_across_time[j, action_idx],
                    "ro",
                    label="inference point",
                )
            else:
                ax.plot(j, gt_action_across_time[j, action_idx], "ro")

        ax.set_title(f"Action {action_idx}")
        ax.legend()

    plt.tight_layout()

    # Create filename with trajectory ID
    Path(save_plot_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_plot_path)

    plt.close()  # Close the figure to free memory


def parse_observation_gr00t(
    obs: dict[str, Any], modality_configs: dict[str, Any]
) -> dict[str, Any]:
    new_obs = {}
    for modality in ["video", "state", "language"]:
        new_obs[modality] = {}
        for key in modality_configs[modality].modality_keys:
            if modality == "language":
                parsed_key = key
            else:
                parsed_key = f"{modality}.{key}"
            arr = obs[parsed_key]
            # Add batch dimension
            if isinstance(arr, str):
                new_obs[modality][key] = [[arr]]
            else:
                new_obs[modality][key] = arr[None, :]
    return new_obs


def parse_action_gr00t(action: dict[str, Any]) -> dict[str, Any]:
    # Unbatch and add prefix
    return {f"action.{key}": action[key][0] for key in action}


def evaluate_single_trajectory(
    policy: BasePolicy,
    loader: LeRobotEpisodeLoader,
    traj_id: int,
    embodiment_tag: EmbodimentTag,
    modality_keys: list[str] | None = None,
    steps=300,
    action_horizon=16,
    save_plot_path=None,
):
    # Ensure steps doesn't exceed trajectory length
    traj = loader[traj_id]
    traj_length = len(traj)
    actual_steps = min(steps, traj_length)
    logging.info(
        f"Using {actual_steps} steps (requested: {steps}, trajectory length: {traj_length})"
    )

    pred_action_across_time = []

    # Extract state and action keys separately and sort for consistent order
    state_keys = loader.modality_configs["state"].modality_keys
    action_keys = (
        loader.modality_configs["action"].modality_keys if modality_keys is None else modality_keys
    )

    modality_configs = deepcopy(loader.modality_configs)
    modality_configs.pop("action")
    for step_count in range(0, actual_steps, action_horizon):
        data_point = extract_step_data(traj, step_count, modality_configs, embodiment_tag)
        logging.info(f"inferencing at step: {step_count}")
        obs = {}
        for k, v in data_point.states.items():
            obs[f"state.{k}"] = v  # (T, D)
        for k, v in data_point.images.items():
            obs[f"video.{k}"] = np.array(v)  # (T, H, W, C)
        for language_key in loader.modality_configs["language"].modality_keys:
            obs[language_key] = data_point.text
        parsed_obs = parse_observation_gr00t(obs, loader.modality_configs)
        _action_chunk, _ = policy.get_action(parsed_obs)
        action_chunk = parse_action_gr00t(_action_chunk)
        for j in range(action_horizon):
            # NOTE: concat_pred_action = action[f"action.{modality_keys[0]}"][j]
            # the np.atleast_1d is to ensure the action is a 1D array, handle where single value is returned
            concat_pred_action = np.concatenate(
                [
                    np.atleast_1d(np.atleast_1d(action_chunk[f"action.{key}"])[j])
                    for key in action_keys
                ],
                axis=0,
            )
            pred_action_across_time.append(concat_pred_action)

    def extract_state_joints(traj: pd.DataFrame, columns: list[str]):
        np_dict = {}
        for column in columns:
            np_dict[column] = np.vstack([arr for arr in traj[column]])
        return np.concatenate([np_dict[column] for column in columns], axis=-1)

    def extract_with_slices(traj: pd.DataFrame, columns: list[str]):
        """Like extract_state_joints, but also returns {column: (start, end)} slices."""
        np_dict = {}
        for column in columns:
            np_dict[column] = np.vstack([arr for arr in traj[column]])
        slices = {}
        offset = 0
        for column in columns:
            d = np_dict[column].shape[1]
            slices[column] = (offset, offset + d)
            offset += d
        return np.concatenate([np_dict[column] for column in columns], axis=-1), slices

    # plot the joints
    state_joints_across_time = extract_state_joints(traj, [f"state.{key}" for key in state_keys])
    gt_action_across_time, action_slices = extract_with_slices(
        traj, [f"action.{key}" for key in action_keys]
    )
    gt_action_across_time = gt_action_across_time[:actual_steps]
    pred_action_across_time = np.array(pred_action_across_time)[:actual_steps]
    assert gt_action_across_time.shape == pred_action_across_time.shape, (
        f"gt_action: {gt_action_across_time.shape}, pred_action: {pred_action_across_time.shape}"
    )

    # ── Overall MSE/MAE (averaged over all action dims, including any zero-padded) ──
    err = gt_action_across_time - pred_action_across_time
    mse = float(np.mean(err ** 2))
    mae = float(np.mean(np.abs(err)))
    logging.info(f"Unnormalized Action MSE across single traj: {mse}")
    logging.info(f"Unnormalized Action MAE across single traj: {mae}")

    # ── Per-key MSE/MAE ──
    # Useful when arms or modalities have different scales (xyz in m, quat ~unit, jaw radians).
    # Also flags zero-padded dims (e.g. PSM2 in monomanual setup) that trivially have 0 error.
    per_key_metrics: dict[str, dict[str, float]] = {}
    for col, (s, e) in action_slices.items():
        key = col[len("action."):] if col.startswith("action.") else col
        gt_slice = gt_action_across_time[:, s:e]
        pr_slice = pred_action_across_time[:, s:e]
        per_key_metrics[key] = {
            "mse": float(np.mean((gt_slice - pr_slice) ** 2)),
            "mae": float(np.mean(np.abs(gt_slice - pr_slice))),
            "gt_range": float(np.ptp(gt_slice)),
            "gt_min": float(np.min(gt_slice)),
            "gt_max": float(np.max(gt_slice)),
            "gt_mean": float(np.mean(gt_slice)),
            "gt_std": float(np.std(gt_slice)),
            "pred_min": float(np.min(pr_slice)),
            "pred_max": float(np.max(pr_slice)),
            "pred_mean": float(np.mean(pr_slice)),
            "pred_std": float(np.std(pr_slice)),
            "is_constant_zero": bool(np.allclose(gt_slice, 0.0)),
        }
        m = per_key_metrics[key]
        logging.info(
            f"  {key:20s}  MAE={m['mae']:.4f}  "
            f"gt[min={m['gt_min']:+.3f} max={m['gt_max']:+.3f} mean={m['gt_mean']:+.3f} std={m['gt_std']:.3f}]  "
            f"pred[min={m['pred_min']:+.3f} max={m['pred_max']:+.3f} mean={m['pred_mean']:+.3f} std={m['pred_std']:.3f}]"
            f"{'  (gt-zero-padded)' if m['is_constant_zero'] else ''}"
        )

    # ── Active-dim aggregate: drop trivially-zero columns (e.g. unused PSM2 in mono setup) ──
    mse_active = mae_active = None
    active_cols = [
        slice(s, e) for col, (s, e) in action_slices.items()
        if not per_key_metrics[col[len("action."):]]["is_constant_zero"]
    ]
    if active_cols and len(active_cols) < len(action_slices):
        active_idx = np.concatenate([np.arange(sl.start, sl.stop) for sl in active_cols])
        gt_active = gt_action_across_time[:, active_idx]
        pr_active = pred_action_across_time[:, active_idx]
        mse_active = float(np.mean((gt_active - pr_active) ** 2))
        mae_active = float(np.mean(np.abs(gt_active - pr_active)))
        logging.info(
            f"  Active-dim aggregate (excludes zero-padded): MSE={mse_active:.6f}  "
            f"MAE={mae_active:.6f}"
        )

    logging.info(f"state_joints vs time {state_joints_across_time.shape}")
    logging.info(f"gt_action_joints vs time {gt_action_across_time.shape}")
    logging.info(f"pred_action_joints vs time {pred_action_across_time.shape}")

    # Plot trajectory results
    plot_trajectory_results(
        state_joints_across_time=state_joints_across_time,
        gt_action_across_time=gt_action_across_time,
        pred_action_across_time=pred_action_across_time,
        traj_id=traj_id,
        state_keys=state_keys,
        action_keys=action_keys,
        action_horizon=action_horizon,
        save_plot_path=save_plot_path or f"/tmp/open_loop_eval/traj_{traj_id}.jpeg",
    )

    # Build a per-trajectory result dict for downstream aggregation/plotting.
    task_index = int(traj["task_index"].iloc[0]) if "task_index" in traj.columns else None
    result = {
        "traj_id": int(traj_id),
        "task_index": task_index,
        "n_steps": int(actual_steps),
        "mse": float(mse),
        "mae": float(mae),
        "mse_active_dim": mse_active,
        "mae_active_dim": mae_active,
        "per_key": per_key_metrics,
    }
    return result


@dataclass
class ArgsConfig:
    """Configuration for evaluating a policy."""

    host: str = "127.0.0.1"
    """Host to connect to."""

    port: int = 5555
    """Port to connect to."""

    steps: int = 200
    """Maximum number of steps to evaluate (will be capped by trajectory length)."""

    traj_ids: list[int] = field(default_factory=lambda: [0])
    """List of trajectory IDs to evaluate."""

    action_horizon: int = 16
    """Action horizon to evaluate."""

    dataset_path: str = "demo_data/cube_to_bowl_5/"
    """Path to the dataset."""

    embodiment_tag: EmbodimentTag = EmbodimentTag.NEW_EMBODIMENT
    """Embodiment tag to use."""

    model_path: str | None = None
    """Path to the model checkpoint."""

    denoising_steps: int = 4
    """Number of denoising steps to use."""

    save_plot_path: str | None = None
    """Path to save the plot to."""

    modality_keys: list[str] | None = None
    """List of modality keys to plot. If None, plot all keys."""

    results_json: str | None = None
    """Path to dump per-trajectory results (JSON) for downstream plotting."""

    modality_config_path: str | None = None
    """Path to a Python file defining the modality config for the embodiment.
    Required for NEW_EMBODIMENT — mirrors launch_finetune.py's load_modality_config."""


def main(args: ArgsConfig):
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Register NEW_EMBODIMENT modality config (side-effect import) before policy load.
    if args.modality_config_path:
        import importlib, sys as _sys
        from pathlib import Path as _Path
        p = _Path(args.modality_config_path)
        if not (p.exists() and p.suffix == ".py"):
            raise FileNotFoundError(f"Modality config not found: {p}")
        _sys.path.insert(0, str(p.parent))
        importlib.import_module(p.stem)
        logging.info(f"Loaded modality config: {p}")

    # Download model checkpoint if it's an S3 path
    local_model_path = args.model_path

    # Extract global_step and checkpoint directory name from checkpoint path
    global_step = None
    if local_model_path:
        # Search for pattern "checkpoint-{number}" anywhere in the path
        match = re.search(r"checkpoint-(\d+)", local_model_path)
        if match:
            try:
                global_step = int(match.group(1))
                logging.info(f"Extracted global_step {global_step} from checkpoint path")
            except ValueError:
                logging.warning(
                    f"Could not parse step number from checkpoint path: {local_model_path}"
                )
        else:
            logging.warning(f"Could not find checkpoint-<step> pattern in path: {local_model_path}")

    if local_model_path is not None:
        import torch

        policy = Gr00tPolicy(
            embodiment_tag=args.embodiment_tag,
            model_path=local_model_path,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
    else:
        policy = PolicyClient(host=args.host, port=args.port)

    # Get the supported modalities for the policy
    modality = policy.get_modality_config()
    logging.info(f"Current modality config: \n{modality}")

    # Create the dataset
    dataset = LeRobotEpisodeLoader(
        dataset_path=args.dataset_path,
        modality_configs=modality,
        video_backend="torchcodec",
        video_backend_kwargs=None,
    )

    logging.info(f"Dataset length: {len(dataset)}")

    # Resolve task_index -> task description for the results JSON, if available.
    tasks_jsonl = Path(args.dataset_path) / "meta" / "tasks.jsonl"
    task_index_to_desc: dict[int, str] = {}
    if tasks_jsonl.exists():
        import json as _json
        with open(tasks_jsonl) as f:
            for line in f:
                row = _json.loads(line)
                task_index_to_desc[int(row["task_index"])] = row["task"]

    # If traj_ids is empty or contains a sentinel "-1", evaluate all trajectories.
    traj_ids = list(args.traj_ids)
    if not traj_ids or traj_ids == [-1]:
        traj_ids = list(range(len(dataset)))
    logging.info(f"Running evaluation on trajectories: {traj_ids}")

    per_traj_results = []

    for traj_id in traj_ids:
        if traj_id >= len(dataset):
            logging.warning(f"Trajectory ID {traj_id} is out of range. Skipping.")
            continue

        logging.info(f"Running trajectory: {traj_id}")
        result = evaluate_single_trajectory(
            policy,
            dataset,
            traj_id,
            args.embodiment_tag,
            args.modality_keys,
            steps=args.steps,
            action_horizon=args.action_horizon,
            save_plot_path=args.save_plot_path,
        )
        result["task_description"] = task_index_to_desc.get(result["task_index"])
        logging.info(f"MSE for trajectory {traj_id}: {result['mse']}, MAE: {result['mae']}")
        per_traj_results.append(result)

    if per_traj_results:
        avg_mse = float(np.mean([r["mse"] for r in per_traj_results]))
        avg_mae = float(np.mean([r["mae"] for r in per_traj_results]))
        logging.info(f"Average MSE across all trajs: {avg_mse}")
        logging.info(f"Average MAE across all trajs: {avg_mae}")
    else:
        avg_mse = avg_mae = None
        logging.info("No valid trajectories were evaluated.")

    if args.results_json:
        import json as _json
        out = {
            "model_path": args.model_path,
            "global_step": global_step,
            "dataset_path": args.dataset_path,
            "embodiment_tag": args.embodiment_tag.value,
            "steps": args.steps,
            "action_horizon": args.action_horizon,
            "avg_mse": avg_mse,
            "avg_mae": avg_mae,
            "per_traj": per_traj_results,
        }
        Path(args.results_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.results_json, "w") as f:
            _json.dump(out, f, indent=2)
        logging.info(f"Wrote results JSON to {args.results_json}")

    logging.info("Done")


if __name__ == "__main__":
    # Parse arguments using tyro
    config = tyro.cli(ArgsConfig)
    main(config)
