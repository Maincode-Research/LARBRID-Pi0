<div align="center">

# LARBRID Fine Tuning of π₀: Diffusion Policy Fine Tuning with Latent Noise Representation Bridging

<p>
  <a href="https://arxiv.org/abs/2611.07275"><img src="https://img.shields.io/badge/arXiv-2611.07275-b31b1b.svg" alt="arXiv"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT"></a>
</p>

</div>


## Overview
This repository provides the official implementation of fine tuning π₀ using LARBRID, Efficient Diffusion Policy Fine Tuning with Latent Noise Representation Bridging (Neurips 2026).

 it keeps the same design as DSRL for those shared components but replace raw diffusion steering with our latent noise representation bridging technique.  

We fine tune the VLA model [π₀](https://github.com/Physical-Intelligence/openpi) across various environments from [Libero](https://libero-project.github.io/main.html) and [AlohaTransferCube](https://arxiv.org/abs/2401.02117?_mdAttr=%7B%22target%22%3A%22_blank%22%7D).

For the other code base that fine tunes pre-trained models for single tasks in Gymnasium, Robomimic and OGBench, please refer to this [LARBRID repo](https://github.com/Maincode-Research/LARBRID).


## Insallation

We recommend using [UV](https://docs.astral.sh/uv/) for managing python version and virtual environments,

1. Clone the repo first:
```
git clone --recurse-submodules git@github.com:Maincode-Research/LARBRID-Pi0.git
cd LARBRID-Pi0
```

2. Set up virtual environment, plus uv inside it
```
python3 -m venv .venv && source .venv/bin/activate
pip install uv
```

3. Install everything except LIBERO in one command. 
```
uv pip install \
  --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match \
  -e . -r requirements.txt \
  -e openpi -e openpi/packages/openpi-client
```

4. Install LIBERO on its own (it has no extra dependencies):
```
uv pip install --config-settings editable_mode=compat -e LIBERO
```

## Training
Libero
```
bash examples/scripts/run_libero_larbrid_psi.sh
```
Aloha
```
bash examples/scripts/run_aloha_larbrid_psi.sh
```


## Credits
This repository is built upon [DSRL](https://diffusion-steering.github.io) which in turn relies on [jaxrl2](https://github.com/ikostrikov/jaxrl2) and [PTR](https://github.com/Asap7772/PTR).

## License
This repository is under the MIT license.

## Citations
If you found this repository helpful, please cite our work,
```
@inproceedings{liang2026larbrid,
  author    = {Liang, Yuanchu and Yang, Yue and He, An-Chi and Kurniawati, Hanna},
  title     = {Efficient Diffusion Policy Fine Tuning with Latent Noise Representation Bridging},
  journal   = {Conference on Neural Information Processing Systems (NeurIPS)},
  year      = {2026},
}
```