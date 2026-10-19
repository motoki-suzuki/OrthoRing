# OrthoRing

**OrthoRing** is a Python package for Bayesian inference of black-hole ringdown using orthonormalized quasinormal-mode (QNM) templates.

The package implements an analysis framework based on the Gram–Schmidt orthonormalization of damped-sinusoid templates. By adopting orthonormal modes, correlations among mode amplitudes can be reduced, and the likelihood can be analytically marginalized over these amplitudes, substantially reducing the computational cost of parameter estimation.

OrthoRing is based on the method developed in:

S. Morisaki, H. Motohashi, M. Suzuki, and D. Watarai, *Analyzing black-hole ringdowns with orthonormal modes*, [Phys. Rev. D **112**, 124083 (2025)](https://journals.aps.org/prd/abstract/10.1103/lhkp-w6hm), [arXiv:2507.12376 [gr-qc]](http://arxiv.org/abs/2507.12376).


## Features

- **Orthonormalized templates:** Construct orthonormal damped-sinusoid templates using the Gram–Schmidt procedure.
- **Analytic marginalization:** Analytically marginalize the likelihood over orthonormal-mode coefficients.
- **Kerr and agnostic models:** Perform parameter estimation assuming the Kerr QNM spectrum or treating the frequencies and damping rates as independent parameters.
- **Bayesian inference:** Interface with [Bilby](https://bilby-dev.github.io/bilby/) for posterior sampling and evidence calculation.
- **Posterior reconstruction:** Recover marginalized mode coefficients, transform between orthonormal and nonorthogonal bases, and reconstruct the corresponding waveforms.
- **Multiple detectors:** Analyze ringdown signals from a network of gravitational-wave detectors using time-domain noise covariance matrices.


## Installation

OrthoRing requires Python 3.10 or later and depends on `numpy`, `scipy`, `bilby`, `lalsuite`, and `tqdm`.

OrthoRing is available on [PyPI](https://pypi.org/project/orthoring/) and can be installed using `pip`:

```bash
pip install orthoring
```

For the latest development version, you can install directly from [GitHub](https://github.com/motoki-suzuki/OrthoRing):

```bash
pip install git+https://github.com/motoki-suzuki/OrthoRing.git
```


### Complete Environment

A Conda environment can be created using the `environment.yml` file included in this repository:

```bash
git clone https://github.com/motoki-suzuki/OrthoRing.git
cd OrthoRing
conda env create -f environment.yml
conda activate orthoring
```


## Usage

OrthoRing supports two frequency models:

- `KerrFrequencyModel`: QNM frequencies and damping rates determined by the remnant black-hole mass and spin.
- `AgnosticFrequencyModel`: Frequencies and damping rates treated as independent parameters without assuming the Kerr QNM spectrum.


### Kerr QNM Model

```python
from orthoring import KerrFrequencyModel

model = KerrFrequencyModel(
    mode_array=[(2, 2, 0), (2, 2, 1)]
)

parameters = {
    "mf": 70.0,
    "chif": 0.7,
}

frequencies, damping_rates = model.get_frequencies(parameters)

print("Frequencies [Hz]:", frequencies)
print("Damping rates [Hz]:", damping_rates)
```


### Agnostic Model

```python
from orthoring import AgnosticFrequencyModel

model = AgnosticFrequencyModel(
    n_modes=2,
    f_min=0.0,
    f_max=1000.0,
    g_min=0.0,
    g_max=10000.0,
    hierarchy="g",
)

parameters = {
    "g_0": 2000.0,
    "dg_1": 500.0,
    "f_0": 250.0,
    "f_1": 280.0,
}

frequencies, damping_rates = model.get_frequencies(parameters)

print("Frequencies [Hz]:", frequencies)
print("Damping rates [Hz]:", damping_rates)
```

The `hierarchy` argument specifies the ordering condition imposed on the modes: `"f"` for decreasing frequencies or `"g"` for increasing damping rates.


### Bayesian Inference

The `Likelihood` class implements time-domain ringdown likelihoods compatible with Bilby.


```python
from orthoring import Likelihood
```

Both explicit coefficient sampling and analytic marginalization over orthonormal-mode coefficients are supported.

- `marginalized=True`: Perform orthonormal analysis with analytic marginalization over the mode coefficients.
- `marginalized=False`: Perform conventional nonorthogonal analysis by jointly sampling the mode amplitudes, phases along with the remnant mass and spin for the Kerr model, or frequencies and damping rates for the agnostic model.


## Repository Structure

```text
OrthoRing/
├── README.md
├── LICENSE
├── THIRD_PARTY_LICENSES.md
├── pyproject.toml
├── src/
│   └── orthoring/
│       ├── __init__.py
│       ├── likelihood.py
│       ├── kerrqnms.py
│       └── KerrQNMEFs-2/
└── examples/
```

- `likelihood.py`: Likelihood evaluation, orthonormalization, Bayesian inference, coefficient sampling, and waveform reconstruction.
- `kerrqnms.py`: Interpolation of Kerr QNM frequencies and excitation factors.
- `KerrQNMEFs-2/`: Tabulated Kerr QNM data obtained from the external dataset described below.
- `examples/`: Example analysis scripts.


## Kerr QNM Data

OrthoRing includes tabulated Kerr QNM frequencies and excitation factors computed by Hayato Motohashi.

The original dataset is available on [Zenodo](https://zenodo.org/records/14380191) and is redistributed without modification under the [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) license.

Users of these data should cite the original dataset and its associated publication:

- H. Motohashi, *Kerr quasinormal mode frequencies and excitation factors*, [Zenodo](https://zenodo.org/records/14380191).
- H. Motohashi, *Resonant Excitation of Quasinormal Modes of Black Holes*, [Phys. Rev. Lett. **134**, 141401 (2025)](https://doi.org/10.1103/PhysRevLett.134.141401), [arXiv:2407.15191 [gr-qc]](http://arxiv.org/abs/2407.15191).

See [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) for details.


## Citation

If you use OrthoRing in your research, please cite:

- S. Morisaki, H. Motohashi, M. Suzuki, and D. Watarai, *Analyzing black-hole ringdowns with orthonormal modes*, [**Phys. Rev. D 112**, 124083 (2025)](https://journals.aps.org/prd/abstract/10.1103/lhkp-w6hm), [arXiv:2507.12376 [gr-qc]](http://arxiv.org/abs/2507.12376).

For applications to the ringdown of GW250114, see also:

- M. Suzuki, K. Kubota, S. Morisaki, H. Motohashi, and D. Watarai, *Ringdown analysis of GW250114 with orthonormal modes*, [**Phys. Rev. D 114**, 044099 (2026)](https://journals.aps.org/prd/abstract/10.1103/6x5z-q9ml), [arXiv:2605.03576 [gr-qc]](http://arxiv.org/abs/2605.03576).

<!-- For applications to QNMs near avoided crossings, see also:

- M. Suzuki, K. Kubota, S. Morisaki, H. Motohashi, and D. Watarai, *TBD*. -->


## License

The OrthoRing source code is distributed under the MIT License. See [LICENSE](LICENSE).

The included Kerr QNM dataset is distributed under the CC BY 4.0 license. See [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).