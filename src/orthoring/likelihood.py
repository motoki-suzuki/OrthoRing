import copy
import scipy
import bilby
from bilby.core.prior import Interped
import lal
import lalsimulation
import numpy as np
from tqdm import tqdm
from scipy.linalg import toeplitz
from scipy.special import ive
from .kerrqnms import kerr_qnm_spectrum
T_MSUN_SECONDS = lal.G_SI * lal.MSUN_SI / lal.C_SI**3


def _log_iv(v, z):
    return np.abs(z.real) + np.log(ive(v, z))

def _logL_phase_marg(amp, l, D):
    if D == 2:
        return -amp**2 / 2 + _log_iv(0, amp * l) + np.log(np.pi)
    elif D == 4:
        return -amp**2 / 2 + _log_iv(1, amp * l) - np.log(amp * l) + np.log(np.pi)
    else:
        raise ValueError("D must be 2 or 4.")

def _logL_phase_amp_marg(l, D):
    if D == 2:
        return l**2 / 4 + _log_iv(0, l**2 / 4) + np.log(np.pi**(3 / 2) / np.sqrt(2))
    elif D == 4:
        return l**2 / 2 + np.log(ive(0, l**2 / 4) - ive(1, l**2 / 4)) + np.log(np.pi**(3 / 2) / (2 * np.sqrt(2)))
    else:
        raise ValueError("D must be 2 or 4.")

def _resample_amp(l, D, N, width=10, n_grid=10000):
    amp_array = np.linspace(max(1e-5, l - width), l + width, n_grid)
    likelihood_array = _logL_phase_marg(amp_array, l, D)
    likelihood_array -= np.max(likelihood_array)
    likelihood_array = np.exp(likelihood_array)
    return Interped(amp_array, likelihood_array).sample(N)

def _resample_theta_vectorized(amps, l, D, n_grid=10000):
    """
    Vectorized sampling of theta for multiple amplitudes.

    Parameters
    ----------
    amps : ndarray of shape (N,)
        Amplitudes for which to sample theta.
    l : float
        Norm of the mode vector.
    D : int
        Dimension (2 or 4).
    n_grid : int
        Number of grid points for theta.

    Returns
    -------
    theta_samples : ndarray of shape (N,)
        Sampled theta values.
    """
    N = len(amps)
    theta_array = np.linspace(1e-5, np.pi - 1e-5, n_grid)
    theta_array_2d = theta_array[None, :]
    amps_2d = amps[:, None]
    likelihood_array = -amps_2d**2 / 2 + amps_2d * l * np.cos(theta_array_2d) + (D - 2) * np.log(np.sin(theta_array_2d))
    likelihood_array -= np.max(likelihood_array, axis=1, keepdims=True)
    likelihood_array = np.exp(likelihood_array)

    cdf = np.cumsum(likelihood_array, axis=1)
    cdf /= cdf[:, -1][:, None]

    u = np.random.rand(N)
    theta_samples = np.empty(N)
    for i in range(N):
        theta_samples[i] = np.interp(u[i], cdf[i], theta_array)
    return theta_samples

def stack_templates(template_dict):
    return np.concatenate(
        [template_dict[d] for d in template_dict],
        axis=0
    )


class KerrFrequencyModel:
    def __init__(self,
        mode_array,
        mf_min: float = None,
        mf_max: float = None,
        chif_min: float = None,
        chif_max: float = None,
        dfs_min: np.ndarray = None,
        dfs_max: np.ndarray = None,
        dgs_min: np.ndarray = None,
        dgs_max: np.ndarray = None,
        beyond_kerr: bool = False,
    ):
        self.set_modes(mode_array)
        self.set_prior_range(
            mf_min, mf_max,
            chif_min, chif_max,
            dfs_min, dfs_max,
            dgs_min, dgs_max,
        )
        self.beyond_kerr = beyond_kerr

    def set_modes(self, mode_array):
        self.mode_array = mode_array
        self.n_modes = len(mode_array)
        self.mode_labels = [f"{ell}{m}{n}" for ell, m, n in mode_array]

    def set_prior_range(
        self,
        mf_min, mf_max,
        chif_min, chif_max,
        dfs_min, dfs_max,
        dgs_min, dgs_max,
    ):
        self.mf_min = mf_min
        self.mf_max = mf_max
        self.chif_min = chif_min
        self.chif_max = chif_max
        self.dfs_min = dfs_min
        self.dfs_max = dfs_max
        self.dgs_min = dgs_min
        self.dgs_max = dgs_max
    
    def build_parameters(self):
        params = {
            "mf": 0.,
            "chif": 0.,
        }
        if self.beyond_kerr:
            for label in self.mode_labels:
                params[f"df_{label}"] = 0.
                params[f"dg_{label}"] = 0.
        return params

    def get_frequencies(self, parameters):
        mf = parameters["mf"]
        chif = parameters["chif"]
        fs = []
        gs = []
        for (ell, m, n), label in zip(self.mode_array, self.mode_labels):
            omega = kerr_qnm_spectrum(ell=ell, m=m, n=n, mf=mf, chif=chif)
            f = np.real(omega) / (2*np.pi)
            g = - np.imag(omega)
            if self.beyond_kerr:
                df = parameters[f"df_{label}"]
                dg = parameters[f"dg_{label}"]
                f = f * np.exp(df)
                g = g * np.exp(dg)
            fs.append(f)
            gs.append(g)
        return np.array(fs), np.array(gs)

    def build_prior(
        self,
    ):
        priors = {
            "mf": bilby.core.prior.Uniform(
                self.mf_min,
                self.mf_max,
                "mf",
                latex_label="$M_f\\,[M_{\\odot}]$",
            ),
            "chif": bilby.core.prior.Uniform(
                self.chif_min,
                self.chif_max,
                "chif",
                latex_label="$\\chi_f$",
            ),
        }
        if self.beyond_kerr:
            for i, label in enumerate(self.mode_labels):
                if self.dfs_min[i] == self.dfs_max[i]:
                    priors[f"df_{label}"] = bilby.core.prior.DeltaFunction(
                        peak=self.dfs_min[i],
                        name=f"df_{label}",
                        latex_label=f"$df_{{{label}}}$",
                    )
                else:
                    priors[f"df_{label}"] = bilby.core.prior.Uniform(
                        self.dfs_min[i],
                        self.dfs_max[i],
                        f"df_{label}",
                        latex_label=f"$df_{{{label}}}$",
                    )
                if self.dgs_min[i] == self.dgs_max[i]:
                    priors[f"dg_{label}"] = bilby.core.prior.DeltaFunction(
                        peak=self.dgs_min[i],
                        name=f"dg_{label}",
                        latex_label=f"$d\\gamma_{{{label}}}$",
                    )
                else:
                    priors[f"dg_{label}"] = bilby.core.prior.Uniform(
                        self.dgs_min[i],
                        self.dgs_max[i],
                        f"dg_{label}",
                        latex_label=f"$d\\gamma_{{{label}}}$",
                    )
        return priors

    def check_constraints(self, fs, gs):
        return True
        

class AgnosticFrequencyModel:
    def __init__(self,
        n_modes,
        f_min: float = None,
        f_max: float = None,
        g_min: float = None,
        g_max: float = None,
        hierarchy="f",
    ):
        self.set_modes(n_modes)
        self.set_prior_range(
            f_min, f_max,
            g_min, g_max,
        )
        self.hierarchy = hierarchy
    
    def set_modes(self, n_modes):
        self.n_modes = n_modes
        self.mode_labels = [f"{i}" for i in range(n_modes)]

    def set_prior_range(
        self,
        f_min, f_max,
        g_min, g_max,
    ):
        self.f_min = f_min
        self.f_max = f_max
        self.g_min = g_min
        self.g_max = g_max

    def build_parameters(self):
        params = {}
        if self.hierarchy == "f":
            params["f_0"] = 0.
            for i in range(1, self.n_modes):
                params[f"df_{i}"] = 0.
            for i in range(self.n_modes):
                params[f"g_{i}"] = 0. 
        elif self.hierarchy == "g":
            params["g_0"] = 0.
            for i in range(1, self.n_modes):
                params[f"dg_{i}"] = 0.
            for i in range(self.n_modes):
                params[f"f_{i}"] = 0.
        else:
            raise ValueError("hierarchy must be 'f' or 'g'.")
        return params

    def get_frequencies(self, parameters):
        fs = np.zeros(self.n_modes)
        gs = np.zeros(self.n_modes)
        if self.hierarchy == "f":
            fs[0] = parameters["f_0"]
            for i in range(1, self.n_modes):
                fs[i] = fs[i-1] - parameters[f"df_{i}"]
            for i in range(self.n_modes):
                gs[i] = parameters[f"g_{i}"]
        elif self.hierarchy == "g":
            gs[0] = parameters["g_0"]
            for i in range(self.n_modes):
                fs[i] = parameters[f"f_{i}"]
            for i in range(1, self.n_modes):
                gs[i] = gs[i-1] + parameters[f"dg_{i}"]
        else:
            raise ValueError("hierarchy must be 'f' or 'g'.")
        return fs, gs

    def build_prior(self):
        priors = {}
        if self.hierarchy == "f":
            priors["f_0"] = bilby.core.prior.Uniform(
                self.f_min,
                self.f_max,
                "f_0",
                latex_label="$f_0$",
            )
            for i in range(self.n_modes):
                priors[f"g_{i}"] = bilby.core.prior.Uniform(
                    self.g_min,
                    self.g_max,
                    f"g_{i}",
                    latex_label=f"$\\gamma_{i}$",
                )
                if i > 0:
                    priors[f"df_{i}"] = bilby.core.prior.Uniform(
                        0,
                        self.f_max - self.f_min,
                        f"df_{i}",
                        latex_label=f"$\\delta f_{i}$",
                    )
        elif self.hierarchy == "g":
            priors["g_0"] = bilby.core.prior.Uniform(
                self.g_min,
                self.g_max,
                "g_0",
                latex_label="$\\gamma_0$",
            )
            for i in range(self.n_modes):
                priors[f"f_{i}"] = bilby.core.prior.Uniform(
                    self.f_min,
                    self.f_max,
                    f"f_{i}",
                    latex_label=f"$f_{i}$",
                )
                if i > 0:
                    priors[f"dg_{i}"] = bilby.core.prior.Uniform(
                        0,
                        self.g_max - self.g_min,
                        f"dg_{i}",
                        latex_label=f"$\\delta\\gamma_{i}$",
                    )
        else:
            raise ValueError("hierarchy must be 'f' or 'g'.")
        return priors

    def check_constraints(self, fs, gs):
        if np.any((fs < self.f_min) | (fs > self.f_max)):
            return False
        if np.any((gs < self.g_min) | (gs > self.g_max)):
            return False
        return True


class Likelihood(bilby.Likelihood):
    def __init__(
        self,
        *,
        data_dict: dict,
        time_dict: dict,
        ra: float,
        dec: float,
        psi: float, 
        D: int,  # Number of coefficients (or damped sinusoids) per mode (D = 2 or 4 in Phys. Rev. D 112, 124083 (2025))
        sampling_rate: float,
        geocent_time: float,
        analysis_start_time: dict,
        auto_correlation_dict: dict,
        frequency_model: str,
        parametrization: str = "combined_coeff",
        marginalized: bool = True,
    ):
        self.set_data(data_dict, time_dict)
        self.set_analysis_settings(sampling_rate, analysis_start_time, parametrization)
        self.set_geometry(ra, dec, psi, geocent_time)
        self.set_model(frequency_model, D, marginalized)
        self.compute_covariance(auto_correlation_dict)
        self.compute_cholesky()
        self.compute_whiten_data()
        self.compute_antenna_patterns()
        parameters = self.build_sampling_parameters()
        super().__init__(parameters)

    def set_data(self, data_dict, time_dict):
        self.data_dict = data_dict
        self.time_dict = time_dict
        self.ifos = sorted(self.data_dict.keys())
        self.n_time = len(next(iter(self.data_dict.values())))

    def set_analysis_settings(self, sampling_rate, analysis_start_time, parametrization):
        self.sampling_rate = sampling_rate
        self.analysis_start_time = analysis_start_time
        self.parametrization = parametrization
        self.duration = self.n_time / self.sampling_rate

    def set_geometry(self, ra, dec, psi, geocent_time):
        self.ra = ra
        self.dec = dec
        self.psi = psi
        self.geocent_time = geocent_time
        self.gps_geocent_time = lal.LIGOTimeGPS(geocent_time)
        self.gmst = lal.GreenwichMeanSiderealTime(self.gps_geocent_time)

    def set_model(self, frequency_model, D, marginalized):
        self.frequency_model = frequency_model
        self.n_modes = frequency_model.n_modes
        self.mode_labels = frequency_model.mode_labels
        self.D = D
        self.marginalized = marginalized

    def compute_covariance(self, auto_correlation_dict):
        self.auto_correlation_dict = auto_correlation_dict
        if self.auto_correlation_dict is not None:
            self.covariance_dict = {
                ifo: toeplitz(self.auto_correlation_dict[ifo][:self.n_time])
                for ifo in self.ifos
            }
            self.inverse_covariance_dict = {
                ifo: np.linalg.inv(self.covariance_dict[ifo])
                for ifo in self.covariance_dict
            }
        if self.auto_correlation_dict is None:
            self.covariance_dict = {
                ifo: np.eye(self.n_time)
                for ifo in self.ifos
            }
            self.inverse_covariance_dict = {
                ifo: np.eye(self.n_time)
                for ifo in self.ifos
            }

    def compute_cholesky(self):
        self.L_dict = {
            ifo: np.linalg.cholesky(self.covariance_dict[ifo])
            for ifo in self.ifos
        }

    def compute_whiten_data(self):
        self.whitened_data_dict = {}
        for ifo in self.ifos:
            self.whitened_data_dict[ifo] = scipy.linalg.solve_triangular(
                self.L_dict[ifo],
                self.data_dict[ifo],
                lower=True,
            )
    
    def compute_antenna_patterns(self):
        self.Fp_Fc_dict = {}
        for ifo in self.ifos:
            Fp, Fc = lal.ComputeDetAMResponse(
                lalsimulation.DetectorPrefixToLALDetector(ifo).response,
                self.ra,
                self.dec,
                self.psi,
                self.gmst
            )
            self.Fp_Fc_dict[ifo] = (Fp, Fc)

    def build_sampling_parameters(self):
        parameters = self.frequency_model.build_parameters()
        if not self.marginalized:
            for label in self.mode_labels:
                parameters[f"amp_{label}"] = 0.
                if self.D == 2:
                    parameters[f"theta0_{label}"] = 0.
                elif self.D == 4:
                    parameters[f"theta1_{label}"] = 0.
                    parameters[f"theta2_{label}"] = 0.
                    parameters[f"theta3_{label}"] = 0.
                else:
                    raise ValueError("D must be 2 or 4.")
        return parameters

    def build_prior(
        self,
        amp_min=None,
        amp_max=None,
    ):
        priors = self.frequency_model.build_prior()

        if self.marginalized:
            return priors      
        elif amp_min is None or amp_max is None:
            raise ValueError(
                "amp_min and amp_max must be provided when marginalized=False."
            )

        theta_ranges = {
            0: (0, 2 * np.pi),
            1: (0, np.pi),
            2: (0, np.pi),
            3: (0, 2 * np.pi),
        }

        for key in self.parameters.keys():
            if key in priors:
                continue
            if key.startswith("amp"):
                mode_label = key[4:]
                priors[key] = bilby.core.prior.Uniform(
                    amp_min,
                    amp_max,
                    key,
                    latex_label=f"$A_{{{mode_label}}}$",
                )
            elif key.startswith("theta"):
                theta_idx = int(key[5])
                mode_label  = key[7:]
                minimum, maximum = theta_ranges[theta_idx]
                priors[key] = bilby.core.prior.Uniform(
                    minimum,
                    maximum,
                    key,
                    latex_label=f"$\\theta_{{{theta_idx},{mode_label}}}$",
                    boundary="periodic",
                )
            else:
                raise ValueError(f"No prior rule defined for {key}")
        return priors

    def inner_product(self, a_dict, b_dict):
        s = 0
        for ifo in a_dict:
            s += np.dot(a_dict[ifo], b_dict[ifo])
        return s

    def noise_weighted_inner_product(self, a_dict, b_dict):
        s = 0
        for ifo in a_dict:
            s += np.dot(a_dict[ifo], np.dot(self.inverse_covariance_dict[ifo], b_dict[ifo]))
        return s

    def get_templates(self, fs, gs):
        templates = []
        if self.D not in (2, 4):
            raise ValueError("'D' must be 2 or 4.")

        for i in range(self.n_modes):
            f = fs[i]
            g = gs[i]
            quadrature = {}
            for ifo in self.ifos:
                t = self.time_dict[ifo] - self.analysis_start_time[ifo]
                decay = np.exp(-g * t)
                c = decay * np.cos(2 * np.pi * f * t)
                s = decay * np.sin(2 * np.pi * f * t)
                quadrature[ifo] = (c, s)

            if self.parametrization == "combined_coeff":
                # F+cos, F+sin, Fxcos, Fxsin
                templates.append({
                    ifo: self.Fp_Fc_dict[ifo][0] * quadrature[ifo][0]
                    for ifo in self.ifos
                })
                templates.append({
                    ifo: self.Fp_Fc_dict[ifo][0] * quadrature[ifo][1]
                    for ifo in self.ifos
                })
                if self.D == 4:
                    templates.append({
                        ifo: self.Fp_Fc_dict[ifo][1] * quadrature[ifo][0]
                        for ifo in self.ifos
                    })
                    templates.append({
                        ifo: self.Fp_Fc_dict[ifo][1] * quadrature[ifo][1]
                        for ifo in self.ifos
                    })
            elif self.parametrization == "explicit_coeff":
                # F+cos + Fxsin, F+sin - Fxcos, F+cos - Fxsin, - F+sin - Fxcos
                templates.append({
                    ifo: self.Fp_Fc_dict[ifo][0] * quadrature[ifo][0] +
                         self.Fp_Fc_dict[ifo][1] * quadrature[ifo][1]
                    for ifo in self.ifos
                })
                templates.append({
                    ifo: self.Fp_Fc_dict[ifo][0] * quadrature[ifo][1] -
                         self.Fp_Fc_dict[ifo][1] * quadrature[ifo][0]
                    for ifo in self.ifos
                })
                if self.D == 4:
                    templates.append({
                        ifo: self.Fp_Fc_dict[ifo][0] * quadrature[ifo][0] -
                             self.Fp_Fc_dict[ifo][1] * quadrature[ifo][1]
                        for ifo in self.ifos
                    })
                    templates.append({
                        ifo: - self.Fp_Fc_dict[ifo][0] * quadrature[ifo][1] -
                             self.Fp_Fc_dict[ifo][1] * quadrature[ifo][0]
                        for ifo in self.ifos
                    })
            else:
                raise ValueError(
                    f"Unknown parametrization '{self.parametrization}'."
                    "Available options are 'combined_coeff' or 'explicit_coeff'."
                )
        return templates

    def get_whitened_templates(self, fs, gs):
        templates = self.get_templates(fs, gs)
        wt_templates = copy.deepcopy(templates)
        for j in range(len(wt_templates)):
            for ifo in wt_templates[j]:
                wt_templates[j][ifo] = scipy.linalg.solve_triangular(
                    self.L_dict[ifo],
                    wt_templates[j][ifo],
                    lower=True,
                )
        return wt_templates

    def get_orthonormal_templates(self, fs, gs):
        templates = self.get_templates(fs, gs)
        wt_templates = copy.deepcopy(templates)
        for j in range(len(wt_templates)):
            for ifo in wt_templates[j]:
                wt_templates[j][ifo] = scipy.linalg.solve_triangular(
                    self.L_dict[ifo],
                    wt_templates[j][ifo],
                    lower=True,
                )
        for j in range(len(templates)):
            norm = np.sqrt(self.inner_product(wt_templates[j], wt_templates[j]))
            for ifo in templates[j]:
                templates[j][ifo] /= norm
                wt_templates[j][ifo] /= norm
            for k in range(j + 1, len(templates)):
                ip = self.inner_product(wt_templates[j], wt_templates[k])
                for ifo in templates[k]:
                    templates[k][ifo] -= ip * templates[j][ifo]
                for ifo in wt_templates[k]:
                    wt_templates[k][ifo] -= ip * wt_templates[j][ifo]
        return templates, wt_templates

    def log_likelihood_marginalized(self, fs, gs):
        _, wt_ortho_templates = self.get_orthonormal_templates(fs, gs)
        logL = 0
        idx = 0
        for _ in range(self.n_modes):
            l_sq = 0
            for _ in range(self.D):
                l_sq += self.inner_product(wt_ortho_templates[idx], self.whitened_data_dict)**2
                idx += 1
            l = np.sqrt(l_sq)
            logL += _logL_phase_amp_marg(l, self.D)
        return logL

    def resample_coefficients(self, fs, gs, N, amp_width=10, amp_n_grid=10000, theta_n_grid=10000):
        _, wt_ortho_templates = self.get_orthonormal_templates(fs, gs)
        l_vecs = np.zeros((self.n_modes, self.D))
        idx = 0
        for mode_idx in range(self.n_modes):
            for d in range(self.D):
                l_vecs[mode_idx, d] = self.inner_product(wt_ortho_templates[idx], self.whitened_data_dict)
                idx += 1
        l_norms = np.linalg.norm(l_vecs, axis=1)

        amplitude_samples = np.array([
            _resample_amp(l_norms[i], self.D, N, width=amp_width, n_grid=amp_n_grid)
            for i in range(self.n_modes)
        ]).T

        theta_samples = np.zeros_like(amplitude_samples)
        for i in range(self.n_modes):
            theta_samples[:, i] = _resample_theta_vectorized(
                amplitude_samples[:, i], l_norms[i], self.D, theta_n_grid
            )

        coefficient_samples = np.zeros((N, self.n_modes * self.D))
        for mode_idx in range(self.n_modes):
            l_vec = l_vecs[mode_idx]
            l = l_norms[mode_idx]

            basis_simple = np.identity(self.D)
            basis = [l_vec / l]
            for _ in range(self.D - 1):
                v = basis[-1]
                for k in range(self.D):
                    basis_simple[k] -= np.dot(v, basis_simple[k]) * v
                norms = np.linalg.norm(basis_simple, axis=1)
                idx_max = np.argmax(norms)
                basis.append(basis_simple[idx_max] / norms[idx_max])
            basis = np.array(basis)

            amps = amplitude_samples[:, mode_idx]
            thetas = theta_samples[:, mode_idx]
            v = np.zeros((self.D, N))

            if self.D == 2:
                v[0, :] = amps * np.cos(thetas)
                v[1, :] = amps * np.sin(thetas)
            elif self.D == 4:
                coss = np.random.uniform(-1, 1, size=N)
                sins = np.sqrt(1 - coss**2)
                phis = np.random.uniform(0, 2 * np.pi, size=N)
                v[0, :] = amps * np.cos(thetas)
                v[1, :] = amps * np.sin(thetas) * coss
                v[2, :] = amps * np.sin(thetas) * sins * np.cos(phis)
                v[3, :] = amps * np.sin(thetas) * sins * np.sin(phis)

            start_idx = mode_idx * self.D
            coefficient_samples[:, start_idx:start_idx + self.D] = (basis.T @ v).T

        return amplitude_samples, coefficient_samples

    def get_coefficients(self, parameters):
        coeffs = []
        for label in self.mode_labels:
            amp = parameters[f"amp_{label}"]
            if self.D == 2:
                theta0 = parameters[f"theta0_{label}"]
                vec = np.array([
                    np.cos(theta0),
                    np.sin(theta0),
                ])
            elif self.D == 4:
                theta1 = parameters[f"theta1_{label}"]
                theta2 = parameters[f"theta2_{label}"]
                theta3 = parameters[f"theta3_{label}"]
                vec = np.array([
                    np.cos(theta1),
                    np.sin(theta1) * np.cos(theta2),
                    np.sin(theta1) * np.sin(theta2) * np.cos(theta3),
                    np.sin(theta1) * np.sin(theta2) * np.sin(theta3),
                ])
            else:
                raise ValueError("D must be 2 or 4.")
            coeffs.extend(amp * vec)
        return np.array(coeffs)

    def log_likelihood(self, parameters):
        fs, gs = self.frequency_model.get_frequencies(parameters)

        if not self.frequency_model.check_constraints(fs, gs):
            return -np.inf

        if not self.marginalized:
            wt_templates = self.get_whitened_templates(fs, gs)
            L = np.array([self.inner_product(self.whitened_data_dict, wt_temp) for wt_temp in wt_templates])
            Q = np.array([
                [self.inner_product(wt_temp_i, wt_temp_j) for wt_temp_j in wt_templates]
                for wt_temp_i in wt_templates
            ])
            coeffs = self.get_coefficients(parameters)
            return np.dot(L, coeffs) - 0.5 * np.dot(coeffs, np.dot(Q, coeffs))

        else:
            return self.log_likelihood_marginalized(fs, gs)

    def run_sampler(
        self,
        *,
        amp_min=None,
        amp_max=None,
        resample_coefficients=False,
        amp_width=10,
        amp_n_grid=10000,
        theta_n_grid=10000,
        **kwargs,
    ):
        if not self.marginalized and resample_coefficients:
            raise ValueError(
                "resample_coefficients=True is only available "
                "when marginalized=True."
            )

        priors = self.build_prior(
            amp_min=amp_min,
            amp_max=amp_max,
        )

        result = bilby.run_sampler(
            likelihood=self,
            priors=priors,
            **kwargs,
        )

        fs_samples, gs_samples = self.get_frequency_samples(result)
        for i, label in enumerate(self.mode_labels):
            result.posterior[f"f_{label}"] = fs_samples[:, i]
            result.posterior[f"g_{label}"] = gs_samples[:, i]

        expected_resampled_columns = []
        for label in self.mode_labels:
            expected_resampled_columns.append(f"ortho_amp_{label}")
            for j in range(self.D):
                expected_resampled_columns.append(
                    f"ortho_coeff_{j}_{label}"
                )
        coefficients_already_resampled = all(
            column in result.posterior.columns
            for column in expected_resampled_columns
        )

        if (
            self.marginalized
            and resample_coefficients
            and not coefficients_already_resampled
        ):
            amplitude_samples = []
            coefficient_samples = []
            for fs, gs in tqdm(
                zip(fs_samples, gs_samples),
                total=len(fs_samples),
                desc="Sampling coefficients",
            ):
                amplitudes, coefficients = self.resample_coefficients(
                    fs, gs, 1,
                    amp_width=amp_width,
                    amp_n_grid=amp_n_grid,
                    theta_n_grid=theta_n_grid,
                )
                amplitude_samples.append(amplitudes[0])
                coefficient_samples.append(coefficients[0])
            amplitude_samples = np.array(amplitude_samples)
            coefficient_samples = np.array(coefficient_samples)
            for i, label in enumerate(self.mode_labels):
                result.posterior[f"ortho_amp_{label}"] = amplitude_samples[:,i]
                for j in range(self.D):
                    result.posterior[f"ortho_coeff_{j}_{label}"] = coefficient_samples[:,self.D * i + j]

        elif (
            self.marginalized
            and resample_coefficients
            and coefficients_already_resampled
        ):
            print(
                "Coefficient samples already exist in the result. "
                "Skipping coefficient resampling."
            )

        if not self.marginalized:
            coefficient_samples = np.array([
                self.get_coefficients(row.to_dict())
                for _, row in result.posterior.iterrows()
            ])
            for i, label in enumerate(self.mode_labels):
                for j in range(self.D):
                    result.posterior[f"coeff_{j}_{label}"] = coefficient_samples[:, self.D * i + j]

        result.save_to_file()
        return result

    def orthonormal_to_nonorthogonal_coefficients(
        self,
        result,
        save=True,
    ):
        ortho_coeff_samples = result.posterior[
            [f"ortho_coeff_{j}_{label}" for label in self.mode_labels for j in range(self.D)]
        ].to_numpy()
        f_samples = result.posterior[
            [f"f_{label}" for label in self.mode_labels]
        ].to_numpy()
        g_samples = result.posterior[
            [f"g_{label}" for label in self.mode_labels]
        ].to_numpy()
        c_samples = np.zeros_like(ortho_coeff_samples)

        for i in tqdm(
            range(len(ortho_coeff_samples)),
            desc="Converting orthonormal coefficients"
        ):
            U_matrix, _ = self.get_U_matrix(f_samples[i], g_samples[i])
            c_samples[i] = U_matrix @ ortho_coeff_samples[i]

        A_samples = np.zeros(
            (len(c_samples), self.n_modes)
        )
        for mode in range(self.n_modes):
            start = mode * self.D
            stop = start + self.D
            A_samples[:, mode] = np.sqrt(
                np.sum(
                    c_samples[:, start:stop]**2,
                    axis=1,
                )
            )

        for i, label in enumerate(self.mode_labels):
            result.posterior[f"amp_{label}"] = A_samples[:, i]
            for j in range(self.D):
                result.posterior[f"coeff_{j}_{label}"] = (
                    c_samples[:, self.D * i + j]
                )
        if save:
            result.save_to_file()
        return result

    def get_frequency_samples(self, result):
        fs_samples = []
        gs_samples = []
        for _, row in result.posterior.iterrows():
            params = {
                key: row[key]
                for key in self.frequency_model.build_parameters().keys()
            }
            fs, gs = self.frequency_model.get_frequencies(params)
            fs_samples.append(fs)
            gs_samples.append(gs)
        return np.array(fs_samples), np.array(gs_samples)

    def get_U_matrix(self, fs, gs):
        _, wt_ortho_templates = self.get_orthonormal_templates(fs, gs)
        wt_templates = self.get_whitened_templates(fs, gs)
        T_tilde = np.stack([stack_templates(t) for t in wt_templates])
        E_tilde = np.stack([stack_templates(e) for e in wt_ortho_templates])

        U_inv_matrix = E_tilde @ T_tilde.T
        Q, R = np.linalg.qr(U_inv_matrix)
        U_matrix = np.linalg.solve(R, Q.T)
        return U_matrix, U_inv_matrix

    def reconstruct_waveforms(
        self,
        result,
        mode="auto",
    ):
            f_samples = result.posterior[
                [f"f_{label}" for label in self.mode_labels]
            ].to_numpy()
            g_samples = result.posterior[
                [f"g_{label}" for label in self.mode_labels]
            ].to_numpy()

            if mode == "auto":
                mode = (
                    "orthonormal" if self.marginalized
                    else "nonorthogonal"
                )
            if mode == "orthonormal":
                coeff_samples = result.posterior[
                    [f"ortho_coeff_{j}_{label}" for label in self.mode_labels for j in range(self.D)]
                ].to_numpy()
            elif mode == "nonorthogonal":
                coeff_samples = result.posterior[
                    [f"coeff_{j}_{label}" for label in self.mode_labels for j in range(self.D)]
                ].to_numpy()
            else:
                raise ValueError

            waveforms = {ifo: [] for ifo in self.ifos}
            mode_waveforms = {label: {ifo: [] for ifo in self.ifos} for label in self.mode_labels}

            for fs, gs, coeffs in zip(f_samples, g_samples, coeff_samples):
                if mode == "orthonormal":
                    templates, _ = self.get_orthonormal_templates(fs, gs)
                elif mode == "nonorthogonal":
                    templates = self.get_templates(fs, gs)
                for ifo in self.ifos:
                    templates_ifo = np.array([t[ifo] for t in templates])
                    waveforms[ifo].append(coeffs @ templates_ifo)
                    for i, label in enumerate(self.mode_labels):
                        start = i * self.D
                        stop = start + self.D
                        mode_waveforms[label][ifo].append(coeffs[start:stop] @ templates_ifo[start:stop])
            waveforms = {ifo: np.asarray(w) for ifo, w in waveforms.items()}
            mode_waveforms = {
                label: {
                    ifo: np.asarray(w)
                    for ifo, w in mode_waveforms[label].items()
                }
                for label in self.mode_labels
            }
            return waveforms, mode_waveforms

    def compute_mismatches(self, waveforms, waveforms_true):
        mismatches = []
        for i in range(len(waveforms[self.ifos[0]])):
            h_rec = {ifo: waveforms[ifo][i] for ifo in self.ifos}
            ip_rec_true = self.noise_weighted_inner_product(h_rec, waveforms_true)
            ip_rec = self.noise_weighted_inner_product(h_rec, h_rec)
            ip_true = self.noise_weighted_inner_product(waveforms_true, waveforms_true)
            mismatch = 1 - ip_rec_true / np.sqrt(ip_rec * ip_true)
            mismatches.append(mismatch)
        return np.array(mismatches)
