# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

robomimic is a framework for robot learning from demonstration, providing standardized datasets, algorithms, and workflows for imitation learning research. Version 0.5.0 adds Diffusion Policy, multi-dataset training, and language-conditioned policies.

## Common Commands

### Installation
```bash
pip install -e .
```

### Training
```bash
# Main training script
python robomimic/scripts/train.py --config <config.json>

# Quick sanity check
python examples/train_bc_rnn.py --debug
```

### Testing
```bash
cd tests
bash test.sh

# Run individual test
python tests/test_bc.py
```

### Evaluation
```bash
# Run trained policy in environment
python robomimic/scripts/run_trained_agent.py --agent <model_path>

# Visualize demonstration data
python robomimic/scripts/playback_dataset.py --dataset <hdf5_path>
```

### Dataset Management
```bash
# Download datasets from HuggingFace
python robomimic/scripts/download_datasets.py

# Extract observations from raw datasets
python robomimic/scripts/dataset_states_to_obs.py --dataset <path>

# Get dataset info
python robomimic/scripts/get_dataset_info.py --dataset <path>
```

### Documentation
```bash
pip install -r requirements-docs.txt
cd docs && make clean && make apidoc && make html && make prep
```

## Architecture

### Training Pipeline
Config → SequenceDataset (HDF5) → DataLoader → Algo (training/eval) → Models → Environment (rollout)

### Key Design Patterns

**Algorithm Factory**: Algorithms register via `@register_algo_factory_func("name")` decorator in `robomimic/algo/algo.py`. Each algorithm implements `train_on_batch()`, `get_action()`, `serialize()`/`deserialize()`.

**Config Factory**: Config classes inherit from `BaseConfig` with metaclass registration. Configs use a locking mechanism for key access control.

**Model Composition**: Network modules in `robomimic/models/` implement `output_shape()` for automated shape propagation. Place sub-modules in `self.nets` (use ModuleDict for multiple).

### Directory Layout
- `robomimic/algo/` - Algorithm implementations (BC, BCQ, CQL, TD3-BC, HBC, IRIS, IQL, Diffusion Policy)
- `robomimic/config/` - Configuration classes with `base_config.py` as foundation
- `robomimic/models/` - Neural network architectures (base_nets, obs_nets, policy_nets, etc.)
- `robomimic/utils/` - Utilities (dataset.py for HDF5 loading, train_utils.py, obs_utils.py)
- `robomimic/envs/` - Environment wrappers (RoboSuite, Gym)
- `robomimic/scripts/` - Entry points for training, evaluation, data processing
- `robomimic/exps/templates/` - JSON config templates for each algorithm
- `examples/` - Example scripts demonstrating usage
- `tests/` - Test suite with per-algorithm test files

### Configuration Hierarchy
```python
config.experiment.name           # Experiment metadata
config.train.batch_size          # Training parameters
config.observation.modalities    # Input specification
config.observation.encoder       # Encoder architecture
config.algo.*                    # Algorithm-specific parameters
```

## Coding Conventions

- 4-space indentation (soft tabs)
- Google Python Style docstrings
- Use `None` or `()` for default arguments, not mutable types
- Prefer `torch.expand` over `torch.repeat` for memory efficiency
- Specify empty kwargs in base configs to allow overrides
- Top-level docstrings in scripts describing purpose and usage

## Dependencies

Core: PyTorch, h5py, numpy, tensorboard, huggingface_hub, transformers, diffusers

Optional: robosuite (v1.5.1 for released datasets), wandb
