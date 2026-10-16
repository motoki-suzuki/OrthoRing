from pathlib import Path
import lal
import numpy as np
from scipy.interpolate import CubicSpline

HERE = Path(__file__).resolve().parent
T_MSUN_SECONDS = lal.G_SI * lal.MSUN_SI / lal.C_SI**3

modes = []
ls = [2, 3, 4, 5, 6]
ns = [0, 1, 2, 3, 4, 5, 6, 7]

for l in ls:
    for m in range(-l, l + 1):
        for n in ns:
            modes.append((l, m, n))

data_dir = HERE / "KerrQNMEFs-2"

if not data_dir.is_dir():
    raise FileNotFoundError(
        f"Kerr QNM data directory not found: {data_dir}"
    )

data = {}
for l, m, n in modes:
    file_path = data_dir / f"l{l}/s-2l{l}m{m}n{n}.dat"
    data[(l, m, n)] = np.loadtxt(file_path, delimiter=",")

interp_omega_dict = {}
interp_excitation_factor_dict = {}

for l, m, n in modes:
    spin = data[(l, m, n)][:,0]
    omega = data[(l, m, n)][:,1] + 1j * data[(l, m, n)][:,2]
    excitation_factor = data[(l, m, n)][:,7] + 1j * data[(l, m, n)][:,8]
    omega /= 2  # Convert from units with 2M = 1 to units with M = 1.
    excitation_factor /= 16  # Convert from units with 2M = 1 to units with M = 1.
    interp_omega_dict[(l, m, n)] = CubicSpline(spin, omega)
    interp_excitation_factor_dict[(l, m, n)] = CubicSpline(spin, excitation_factor)

def kerr_qnm_spectrum(*, ell, m, n, chif, mf=None):
    omega = interp_omega_dict[(ell, m, n)](chif)
    if mf is not None:
        omega /= mf * T_MSUN_SECONDS
    return omega

def kerr_qnm_excitation_factor(*, ell, m, n, chif, mf=None):
    B = interp_excitation_factor_dict[(ell, m, n)](chif)
    if mf is not None:
        B /= (mf * T_MSUN_SECONDS)**4
    return B