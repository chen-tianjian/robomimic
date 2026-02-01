"""
Test script for Diffusion Policy algorithm. Each test trains a variant of
Diffusion Policy for a handful of gradient steps and tries one rollout with
the model. Excludes stdout output by default (pass --verbose to see stdout output).
"""
import argparse
from collections import OrderedDict

import robomimic
from robomimic.config import Config
import robomimic.utils.test_utils as TestUtils
from robomimic.utils.log_utils import silence_stdout
from robomimic.utils.torch_utils import dummy_context_mgr


def get_algo_base_config():
    """
    Base config for testing Diffusion Policy algorithm.
    """

    # config with basic settings for quick training run
    config = TestUtils.get_base_config(algo_name="diffusion_policy")

    # low-level obs (no images by default)
    config.observation.modalities.obs.low_dim = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"]
    config.observation.modalities.obs.rgb = []

    # Diffusion Policy requires normalized actions in [-1, 1]
    # Enable min_max normalization for actions
    config.train.action_config.actions.normalization = "min_max"

    # set shorter horizons for faster testing
    config.algo.horizon.observation_horizon = 2
    config.algo.horizon.action_horizon = 4
    config.algo.horizon.prediction_horizon = 8
    config.train.seq_length = 8  # should match prediction_horizon
    config.train.frame_stack = 2  # should match observation_horizon

    # reduce timesteps for faster testing
    config.algo.ddpm.num_train_timesteps = 10
    config.algo.ddpm.num_inference_timesteps = 10
    config.algo.ddim.num_train_timesteps = 10
    config.algo.ddim.num_inference_timesteps = 5

    # by default, use DDPM with EMA
    config.algo.ddpm.enabled = True
    config.algo.ddim.enabled = False
    config.algo.ema.enabled = True

    return config


def convert_config_for_images(config):
    """
    Modify config to use image observations.
    """

    # using high-dimensional images - don't load entire dataset into memory, and smaller batch size
    config.train.hdf5_cache_mode = "low_dim"
    config.train.num_data_workers = 0
    config.train.batch_size = 16

    # replace object with rgb modality
    config.observation.modalities.obs.low_dim = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
    config.observation.modalities.obs.rgb = ["agentview_image"]

    # set up visual encoders
    config.observation.encoder.rgb.core_class = "VisualCore"
    config.observation.encoder.rgb.core_kwargs.feature_dimension = 64
    config.observation.encoder.rgb.core_kwargs.backbone_class = 'ResNet18Conv'
    config.observation.encoder.rgb.core_kwargs.backbone_kwargs.pretrained = False
    config.observation.encoder.rgb.core_kwargs.backbone_kwargs.input_coord_conv = False
    config.observation.encoder.rgb.core_kwargs.pool_class = "SpatialSoftmax"
    config.observation.encoder.rgb.core_kwargs.pool_kwargs.num_kp = 32
    config.observation.encoder.rgb.core_kwargs.pool_kwargs.learnable_temperature = False
    config.observation.encoder.rgb.core_kwargs.pool_kwargs.temperature = 1.0
    config.observation.encoder.rgb.core_kwargs.pool_kwargs.noise_std = 0.0

    # observation randomizer class - set to None to use no randomization
    config.observation.encoder.rgb.obs_randomizer_class = None

    return config


def make_image_modifier(config_modifier):
    """
    Turn a config modifier into its image version. Note that
    this explicit function definition is needed for proper
    scoping of @config_modifier.
    """
    return lambda x: config_modifier(convert_config_for_images(x))


# mapping from test name to config modifier functions
MODIFIERS = OrderedDict()
def register_mod(test_name):
    def decorator(config_modifier):
        MODIFIERS[test_name] = config_modifier
    return decorator


@register_mod("diffusion_policy-ddpm")
def diffusion_policy_ddpm_modifier(config):
    # default DDPM scheduler
    config.algo.ddpm.enabled = True
    config.algo.ddim.enabled = False
    return config


@register_mod("diffusion_policy-ddim")
def diffusion_policy_ddim_modifier(config):
    # DDIM scheduler (faster inference)
    config.algo.ddpm.enabled = False
    config.algo.ddim.enabled = True
    return config


@register_mod("diffusion_policy-ddpm-no-ema")
def diffusion_policy_ddpm_no_ema_modifier(config):
    # DDPM without EMA
    config.algo.ddpm.enabled = True
    config.algo.ddim.enabled = False
    config.algo.ema.enabled = False
    return config


@register_mod("diffusion_policy-ddim-no-ema")
def diffusion_policy_ddim_no_ema_modifier(config):
    # DDIM without EMA
    config.algo.ddpm.enabled = False
    config.algo.ddim.enabled = True
    config.algo.ema.enabled = False
    return config


@register_mod("diffusion_policy-ddpm-linear-schedule")
def diffusion_policy_ddpm_linear_schedule_modifier(config):
    # DDPM with linear beta schedule
    config.algo.ddpm.enabled = True
    config.algo.ddim.enabled = False
    config.algo.ddpm.beta_schedule = "linear"
    return config


# add image version of all tests
image_modifiers = OrderedDict()
for test_name in MODIFIERS:
    lst = test_name.split("-")
    name = "-".join(lst[:1] + ["rgb"] + lst[1:])
    image_modifiers[name] = make_image_modifier(MODIFIERS[test_name])
MODIFIERS.update(image_modifiers)


# test for image crop randomization
@register_mod("diffusion_policy-image-crop")
def diffusion_policy_image_crop_modifier(config):
    config = convert_config_for_images(config)

    # observation randomizer class - using Crop randomizer
    config.observation.encoder.rgb.obs_randomizer_class = "CropRandomizer"

    # kwargs for observation randomizers
    config.observation.encoder.rgb.obs_randomizer_kwargs.crop_height = 76
    config.observation.encoder.rgb.obs_randomizer_kwargs.crop_width = 76
    config.observation.encoder.rgb.obs_randomizer_kwargs.num_crops = 1
    config.observation.encoder.rgb.obs_randomizer_kwargs.pos_enc = False
    return config


def test_diffusion_policy(silence=True):
    for test_name in MODIFIERS:
        context = silence_stdout() if silence else dummy_context_mgr()
        with context:
            base_config = get_algo_base_config()
            res_str = TestUtils.test_run(base_config=base_config, config_modifier=MODIFIERS[test_name])
        print("{}: {}".format(test_name, res_str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verbose",
        action='store_true',
        help="don't suppress stdout during tests",
    )
    args = parser.parse_args()

    test_diffusion_policy(silence=(not args.verbose))
