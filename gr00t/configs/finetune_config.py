# Finetune config used for single node post-training.
from dataclasses import dataclass

from gr00t.data.embodiment_tags import EmbodimentTag


@dataclass
class FinetuneConfig:
    """
    Configuration for fine-tuning a Vision-Language-Action (VLA) model.

    This dataclass defines all parameters needed to launch a fine-tuning job
    on a pretrained base model using a custom dataset and embodiment-specific
    modality configuration. It controls model tuning options, data augmentation,
    and training hyperparameters.
    """

    # --- Data and Model Paths ---
    base_model_path: str
    """Path to the pretrained base model checkpoint (e.g., Hugging Face model hub or local directory)."""

    dataset_path: str
    """Path to the dataset root directory containing trajectory data for fine-tuning."""

    embodiment_tag: EmbodimentTag
    """Identifier specifying which embodiment (robot configuration) this fine-tuning run targets."""

    modality_config_path: str | None = None
    """
    Path to a Python file defining the modality configuration for the given embodiment. 
    If None, use the pre-registered modality config in `gr00t/configs/data/embodiment_configs.py`. 
    """

    include_splits: list[str] | None = None
    """
    Optional allowlist of dataset splits (from meta/info.json) to include.
    If provided, only these splits are used for training and stats.
    """

    exclude_splits: list[str] | None = None
    """
    Optional denylist of dataset splits (from meta/info.json) to exclude.
    Applied after include_splits (if set). Useful for skipping fail episodes.
    """

    # --- Model Tuning Flags ---
    tune_llm: bool = False
    """If True, fine-tune the language model (LLM) backbone during training."""

    tune_visual: bool = False
    """If True, fine-tune the visual encoder (e.g., ViT or CNN backbone)."""

    tune_projector: bool = True
    """If True, fine-tune the multimodal projector layers that map vision/language features to a shared space."""

    tune_diffusion_model: bool = True
    """If True, fine-tune the diffusion-based action decoder (if present in the model)."""

    action_horizon: int = 50
    """Number of action timesteps the model predicts per forward pass. Must match
    the data pipeline's window length (set via the modality config's
    delta_indices). Default 50 matches peg-transfer; sofa_env / Scheikl tasks
    use 16."""

    state_dropout_prob: float = 0.0
    """
    Dropout probability applied to state inputs for regularization during training.
    """

    state_dropout_prob_per_embodiment: dict[str, float] | None = None
    """
    Per-embodiment state dropout overrides. Keys are embodiment tag strings,
    values are dropout probabilities in [0.0, 1.0].
    """

    state_dropout_prob_per_embodiment_json: str | None = None
    """
    JSON string form of state_dropout_prob_per_embodiment, for shell-friendly
    invocation. Example: '{"jhu_imerse_dvrk_mono": 1.0}'. Takes precedence over
    state_dropout_prob_per_embodiment when set.
    """

    # --- BEAST (B-spline action tokenizer) ---
    use_bspline: bool = False
    """If True, predict K B-spline control points per action channel instead of
    T raw timesteps; control points are decoded to a T-step trajectory via a
    fixed B-spline basis. See beast-groot-h-integration plan."""

    bspline_num_basis: int = 10
    """Number of B-spline control points (K). DiT predicts a length-K sequence."""

    bspline_degree: int = 4
    """B-spline polynomial degree. Degree 4 → C³-continuous → bounded jerk."""

    bspline_init_cond_order: int = 0
    """B-spline initial-condition order: 0 = free-floating chunk start (default),
    2 = clamp the chunk start position AND velocity to the current robot state.
    Order 2 removes inter-chunk discontinuities (the boundary jerk spikes). BEAST
    only implements orders 0 and 2. Requires state delta_indices=[-1,0]."""

    # --- Data Augmentation ---
    random_rotation_angle: int | None = None
    """Maximum rotation angle (in degrees) for random rotation augmentation of input images."""

    color_jitter_params: dict[str, float] | None = None
    """
    Parameters for color jitter augmentation on images.

    Expected keys include:
      - "brightness": float
      - "contrast": float
      - "saturation": float
      - "hue": float
    Example: {"brightness": 0.4, "contrast": 0.4, "saturation": 0.4, "hue": 0.1}

    If None, applying the default color jitter augmentation from the pretrained model.
    """
    extra_augmentation_config: str | None = None
    """
    JSON string for extra image augmentations (mask-based and others).

    Expected keys include:
      - "background_noise_transforms": list of dicts for noise on mask regions
          - "target_mask_values": list of int (e.g., [0])
          - "p": float (probability of applying)
      - "masked_region_transforms": list of dicts for color tint on mask regions
          - "target_mask_values": list of int (e.g., [4] or [5])
          - "p": float (probability of applying)
          - "alpha_range": [min, max] for random_tint intensity

    Example: {"background_noise_transforms": [{"target_mask_values": [0], "p": 0.9}],
              "masked_region_transforms": [{"target_mask_values": [4], "p": 1.0, "alpha_range": [0, 1]}]}

    If None, no extra augmentations are applied.
    """

    image_size: tuple[int, int] | None = None
    """
    Intermediate padded size as (height, width) for resize with padding.
    Images are resized (preserving aspect ratio) and padded to this size.
    Should be >= image_crop_size to allow for cropping augmentation.
    Example: (540, 720) to pad all images to 540×720 before cropping.
    If None, uses shortest_image_edge with aspect-preserving crops (default albumentations).
    """

    image_crop_size: tuple[int, int] | None = None
    """
    Final output size as (height, width) after cropping. Only used when image_size is set.
    This is the resolution the model actually sees.
    Example: (480, 640) means final images are 480×640.
    If None, defaults to image_size (no cropping, padded size is final size).
    """

    # --- Training Configuration ---
    global_batch_size: int = 64
    """Total effective batch size across all GPUs and accumulation steps."""

    dataloader_num_workers: int = 2
    """Number of parallel worker processes used for data loading."""

    learning_rate: float = 1e-4
    """Initial learning rate for optimizer."""

    gradient_accumulation_steps: int = 1
    """Number of forward passes to accumulate before performing a backward/update step."""

    output_dir: str = "./outputs"
    """Directory where model checkpoints, logs, and outputs are saved."""

    save_steps: int = 1000
    """Frequency (in training steps) at which to save checkpoints."""

    save_total_limit: int = 5
    """Maximum number of checkpoints to keep before older ones are deleted."""

    num_gpus: int = 1
    """Number of GPUs available for distributed or single-node training."""

    use_wandb: bool = False
    """
    If True, log metrics and artifacts to Weights & Biases (wandb).
    The project is `finetune-gr00t-n1d6`.
    You need to login to wandb to view the logs.
    """

    max_steps: int = 10000
    """Total number of training steps to run before stopping."""

    weight_decay: float = 1e-5
    """Weight decay coefficient for optimizer (L2 regularization)."""

    warmup_ratio: float = 0.05
    """Proportion of total training steps used for learning rate warm-up."""

    shard_size: int = 2**10
    """Size of the shard to use for the dataset during preloading."""

    episode_sampling_rate: float = 0.1
    """Sampling rate for the episodes."""

    num_shards_per_epoch: int = int(1e5)
    """Number of shards to use for the dataset. reduce this number if vram is limited."""

    # --- Statistics Override ---
    override_pretraining_statistics: bool = False
    """
    If True, override the pretrained model's normalization statistics with
    statistics computed from the fine-tuning dataset. Required when the
    fine-tuning data has a different distribution than the pretraining data
    (e.g., different coordinate frame, workspace size, or gripper units).
    """

    # --- Statistics Calculation Flags ---
    calculate_norm_stats: bool = False
    """
    If True, only calculate normalization statistics and exit without training.
    Uses skip_video=True for fast iteration. Statistics will be saved to
    the norm_stats_output_path or the dataset's meta directory.
    """

    norm_stats_output_path: str | None = None
    """
    Path to save calculated normalization statistics. If None, saves to
    the dataset's meta/temporal_stats.json file. Unlike stats.json (raw
    parquet data), temporal stats cover actions after REL_XYZ_ROT6D
    conversion and include a temporal dimension for the action chunk.
    """

    stats_num_workers: int | None = None
    """
    Number of parallel workers for statistics calculation. If None, uses CPU count.
    Set to 1 to disable parallelism. Only used when calculate_norm_stats=True.
    """
