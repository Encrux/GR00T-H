"""
Stereo dVRK (JHU) modality configuration for GR00T N1.6.

Same dual-arm REL_XYZ_ROT6D action representation as the monocular variant,
but exposes both endoscope_left and endoscope_right as video inputs to give
the model a depth cue from parallel-axis stereo (5.5 mm baseline, matching
the real da Vinci endoscope).
"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


ACTION_HORIZON = 50

jhu_dvrk_mono_stereo_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "endoscope_left",
            "endoscope_right",
        ],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "psm1_pose",
            "psm1_gripper",
            "psm2_pose",
            "psm2_gripper",
        ],
        mean_std_embedding_keys=[
            "psm1_pose",
            "psm1_gripper",
            "psm2_pose",
            "psm2_gripper",
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(ACTION_HORIZON)),
        modality_keys=[
            "psm1_pose",
            "psm1_gripper",
            "psm2_pose",
            "psm2_gripper",
        ],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.REL_XYZ_ROT6D,
                type=ActionType.EEF,
                format=ActionFormat.XYZ_ROT6D,
                state_key="psm1_pose",
                normalization_type="temporal_meanstd",
                input_rotation_format="quat",
                reference_rotation_format="quat",
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
                state_key=None,
                normalization_type="temporal_meanstd",
            ),
            ActionConfig(
                rep=ActionRepresentation.REL_XYZ_ROT6D,
                type=ActionType.EEF,
                format=ActionFormat.XYZ_ROT6D,
                state_key="psm2_pose",
                normalization_type="temporal_meanstd",
                input_rotation_format="quat",
                reference_rotation_format="quat",
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
                state_key=None,
                normalization_type="temporal_meanstd",
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

register_modality_config(jhu_dvrk_mono_stereo_config, embodiment_tag=EmbodimentTag.JHU_IMERSE_DVRK_MONO_STEREO)
