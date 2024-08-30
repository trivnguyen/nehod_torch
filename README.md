# NeHOD: Neural Halo Occupation Distribution

NeHOD is a machine learning-based model for painting galaxies onto dark matter (DM) halos, similar to the Halo Occupation Distribution (HOD) model.
NeHOD models galaxies as point clouds using a Variational Diffusion Model with a Transformer-based noise model.
Point clouds allow NeHOD to resolve small spatial scales down to the resolution of the simulations.

The diffusion model is implemented using `torch` and follows quite closely the implementation in Cuesta-Lazaro & Mishra-Sharma 2023 (CM23).
Additionally, NeHOD also uses the Neural Spline Flows implementation from the `zuko` library to model central galaxies using neural spline flows.
Training is managed using `pytorch-lightning`, and configuration is handled using `ml_collections`.

For more details, please refer to the following papers:
- NeHOD: [arXiv:xxxx.xxxx](https://arxiv.org/abs/xxxx.xxxx)
- CM23: [arXiv:2311.17141](https://arxiv.org/abs/2311.17141)

## Requirements
Core ML requirements are:
- `Python >= 3.11`
- `torch >= 2.0.1`
- `pytorch-lightning >= 2.1.3`
- `ml_collections >= 0.1.1`
- `zuko >= 1.1.0`
- `numpy >= 2.1.0`
- `scipy >= 1.14.1`

Additional requirements are (mostly for input/output and logging):
- `absl-py >= 2.0.0`
- `pandas >= 2.2.2`
- `PyYAML >= 6.0.2`
- `tqdm >= 4.66.1`


Full requirements can be found in `requirements.txt`. To install all requirements, run:
```bash
pip install -r requirements.txt
```

## Simple training example
The syntax for running the training script is simple:
```bash
python train.py --config config/example_vdm.py  # for the VDM
python train_flows.py --config config/example_flows.py  # for the NSF
```
The VDM and flows are trained independently, so this can be done in parallel.
The example config files are found in the `config` directory.

Most of the configuration is handled in the `config/*.py` files, which are Python files that return a dictionary with the configuration parameters.
See `ml_collections` [(repo)](https://github.com/google/ml_collections) for more details on how to define configuration files.


## Citation
If you are using this code, please cite the following paper:

CM23 paper:
```bibtex
@article{Cuesta-Lazaro:2023zuk,
    author = "Cuesta-Lazaro, Carolina and Mishra-Sharma, Siddharth",
    title = "{A point cloud approach to generative modeling for galaxy surveys at the field level}",
    eprint = "2311.17141",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.CO",
    reportNumber = "MIT-CTP/5651",
    month = "11",
    year = "2023"
}
```

NeHOD paper (placeholder):
```bibtex
```