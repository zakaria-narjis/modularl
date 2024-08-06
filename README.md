# ModulaRL
<div align="center">
  <img src="assets/modulaRL_logo.svg" alt="ModulaRL Logo">
</div>

<div align="center">
  🚧 This library is still under construction. 🚧
</div>

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![pytest](https://img.shields.io/badge/tested%20with-pytest-46a2f1.svg)](https://docs.pytest.org/en/stable/)
[![Documentation Status](https://readthedocs.org/projects/modularl/badge/?version=latest)](https://modularl.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

ModulaRL is a highly modular and extensible reinforcement learning library built on PyTorch. It aims to provide researchers and developers with a flexible framework for implementing, experimenting with, and extending various RL algorithms.

## Features

- Modular architecture allowing easy component swapping and extension
- Efficient implementations leveraging PyTorch's capabilities
- Integration with TorchRL for optimized replay buffers
- Clear documentation and examples for quick start
- Designed for both research and practical applications in reinforcement learning

## Installation

```bash
pip install modularl
```
## Algorithms Implemented

## Algorithms Implemented

| Algorithm                  | Type       | Paper                                               | Continuous Action | Discrete Action      |
|----------------------------|------------|-----------------------------------------------------|-------------------|----------------------|
| SAC (Soft Actor-Critic)    | Off-policy | [Haarnoja et al. 2018](https://arxiv.org/abs/1801.01290) | ✅                 | Not implemented YET  |
| TD3 (Twin Delayed DDPG)    | Off-policy | [Fujimoto et al. 2018](https://arxiv.org/abs/1802.09477) | ✅                 | Not implemented YET  |



## Citation
```
@software{modularl2024,
  author = {zakaria narjis},
  title = {ModulaRL: A Modular Reinforcement Learning Library},
  year = {2024},
  url = {https://github.com/zakaria-narjis/modularl}
}
```
