"""
Test script for ACT (Action Chunking with Transformers) algorithm. Each test
trains a variant of ACT for a handful of gradient steps and tries one rollout
with the model. Excludes stdout output by default (pass --verbose to see stdout output).

Note: ACT requires the external 'act' package to be installed. If the package
is not available, tests will fail with an import error.

Note: ACT requires image observations, so all tests use RGB inputs.
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
    Base config for testing ACT algorithm.

    ACT requires image observations (camera inputs), so we always include RGB modalities.
    """

    # config with basic settings for quick training run
    config = TestUtils.get_base_config(algo_name="act")

    # ACT requires images - set up observation modalities
    config.observation.modalities.obs.low_dim = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
    config.observation.modalities.obs.rgb = ["agentview_image"]

    # using high-dimensional images - don't load entire dataset into memory, and smaller batch size
    config.train.hdf5_cache_mode = "low_dim"
    config.train.num_data_workers = 0
    config.train.batch_size = 8  # smaller batch for faster testing

    # shorter sequence length for faster testing
    config.train.seq_length = 5

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

    # observation randomizer - use CropRandomizer by default (as in template)
    config.observation.encoder.rgb.obs_randomizer_class = "CropRandomizer"
    config.observation.encoder.rgb.obs_randomizer_kwargs.crop_height = 76
    config.observation.encoder.rgb.obs_randomizer_kwargs.crop_width = 76
    config.observation.encoder.rgb.obs_randomizer_kwargs.num_crops = 1
    config.observation.encoder.rgb.obs_randomizer_kwargs.pos_enc = False

    # smaller architecture for faster testing
    config.algo.act.hidden_dim = 128
    config.algo.act.dim_feedforward = 512
    config.algo.act.enc_layers = 2
    config.algo.act.dec_layers = 2
    config.algo.act.nheads = 4
    config.algo.act.latent_dim = 16

    # default loss weights
    config.algo.loss.l1_weight = 1.0
    config.algo.loss.l2_weight = 0.0
    config.algo.loss.cos_weight = 0.0
    config.algo.act.kl_weight = 20

    return config


# mapping from test name to config modifier functions
MODIFIERS = OrderedDict()
def register_mod(test_name):
    def decorator(config_modifier):
        MODIFIERS[test_name] = config_modifier
    return decorator


@register_mod("act")
def act_modifier(config):
    # default ACT - L1 loss with KL regularization
    return config


@register_mod("act-l2-loss")
def act_l2_loss_modifier(config):
    # ACT with L2 loss instead of L1
    config.algo.loss.l1_weight = 0.0
    config.algo.loss.l2_weight = 1.0
    return config


@register_mod("act-combined-loss")
def act_combined_loss_modifier(config):
    # ACT with combined L1 and L2 loss
    config.algo.loss.l1_weight = 0.5
    config.algo.loss.l2_weight = 0.5
    return config


@register_mod("act-low-kl-weight")
def act_low_kl_weight_modifier(config):
    # ACT with lower KL weight (more diverse actions)
    config.algo.act.kl_weight = 1
    return config


@register_mod("act-high-kl-weight")
def act_high_kl_weight_modifier(config):
    # ACT with higher KL weight (more regularized latent space)
    config.algo.act.kl_weight = 100
    return config


@register_mod("act-larger-latent")
def act_larger_latent_modifier(config):
    # ACT with larger latent dimension
    config.algo.act.latent_dim = 64
    return config


@register_mod("act-no-crop-randomizer")
def act_no_crop_randomizer_modifier(config):
    # ACT without crop randomization
    config.observation.encoder.rgb.obs_randomizer_class = None
    return config


def test_act(silence=True):
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

    test_act(silence=(not args.verbose))
