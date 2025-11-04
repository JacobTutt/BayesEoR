import numpy as np
from astropy.constants import c
from astropy import units
from astropy.units import Quantity
import matplotlib.pyplot as plt
from pathlib import Path

from ..params import BayesEoRParser
from .model.instrument import load_inst_model
from .model.beam import Beam

def get_uv_model_params(
    config : Path | str | None = None,
    fov_ra_eor : Quantity | float | None = None,
    fov_dec_eor : Quantity | float | None = None,
    fov_ra_fg : Quantity | float | None = None,
    fov_dec_fg : Quantity | float | None = None,
    nf : int | None = None,
    freq_min : Quantity | float | None = None,
    df : Quantity | float | None = None,
    beam_type : str | None = None,
    fwhm_deg : Quantity | float | None = None,
    diam : Quantity | float | None = None,
    Nsigma : float | None = 5,
    uvs_path : Path | str | None = None,
    plot : bool = False
):
    """
    Calculate the required nu (nv) given sky and instrument model parameters.

    Parameters
    ----------
    config : pathlib.Path or str, optional
        Path to a yaml configuration file parseable by
        :class:`bayeseor.params.BayesEoRParser`. If `config` is None,
        `fov_ra_eor`, `nf`, `freq_min`, `df`, `beam_type`, and `uvs_path` are
        required. Defaults to None.
    fov_ra_eor : astropy.units.Quantity or float, optional
        Field of view of the Right Ascension axis of the EoR sky model in
        degrees. Required if `config` is None. Defaults to None.
    fov_dec_eor : astropy.units.Quantity or float, optional
        Field of view of the Declination axis of the EoR sky model in degrees.
        Defaults to `fov_ra_eor`.
    fov_ra_fg : astropy.units.Quantity or float, optional
        Field of view of the Right Ascension axis of the foreground sky model
        in degrees. Defaults to `fov_ra_eor`.
    fov_dec_fg : astropy.units.Quantity or float, optional
        Field of view of the Declination axis of the foreground sky model in
        degrees. Defaults to `fov_dec_eor` if `fov_ra_fg` is None or
        `fov_ra_fg`.
    nf : int, optional
        Number of frequency channels. Required if `config` is None. Defaults
        to None.
    freq_min : astropy.units.Quantity or float, optional
        Minimum frequency in hertz if not a Quantity. Required if `config` is
        None. Defaults to None.
    df : astropy.units.Quantity or float, optional
        Frequency channel width in hertz if not a Quantity. Required if
        `config` is None. Defaults to None.
    beam_type : str, optional
        One of 'uniform', 'gaussian', or 'airy'. Required if `config` is None.
        Defaults to None.
    fwhm_deg : astropy.units.Quantity or float, optional
        Full width at half maximum (FWHM) of the beam in degrees if not a
        Quantity. Used if `beam_type` is 'airy' or 'gaussian'. If `beam_type`
        is 'airy', the effective antenna diameter is calculated from the FWHM
        at a user-specified frequency. Defaults to None.
    diam : astropy.units.Quantity or float, optional
        Antenna (aperture) diameter in meters if not a Quantity. Used if
        `beam_type` is 'airy' or 'gaussian'. If `beam_type` is 'gaussian', the
        effective full width at half maximum is calculated from `diam` at a
        user-specified frequency. Defaults to None.
    Nsigma : float, optional
        Number of standard deviations to include in the aperture function if
        using a Gaussian beam. Defaults to 5.
    uvs_path : pathlib.Path or str, optional
        Path to a numpy-compatible file containing a 2D array of sampled
        (u, v) in units of meters with shape (Nbls, 2), where Nbls is the
        number of baselines and the u (v) axis is indexed by the 0th (1st)
        column index, i.e. `u = uvs[:, 0]` and `v = uvs[:, 1]`. Required if
        `config` is None. Optionally, if `config` and `uvs_path` are not None,
        the (u, v) loaded from `uvs_path` will supersede any (u, v) from an
        instrument model specified via `inst_model` in the configuration yaml
        file. Defaults to None.
    plot : bool, optional
        Plot the model uv planes for the EoR and foreground models with the
        model grid and aperture function as a circle with the diameter
        calculated based on the beam model in wavelengths. The aperture
        function is only plotted for the baseline along the u (v) axis with
        the largest |u| (|v|). Defaults to False.

    Returns
    -------
    model_dict : dict
        Dictionary containing the required model uv-plane parameters `nu_eor`,
        `nv_eor`, `nu_fg`, `nv_fg`, and `nside`.
    """
    if config is not None:
        if not isinstance(config, Path):
            config = Path(config)
        if not config.exists():
            raise FileNotFoundError(f"{config} does not exist")
        parser = BayesEoRParser()
        args = parser.parse_args(["--config", config.as_posix()])
        fov_ra_eor = args.fov_ra_eor * units.deg
        nf = args.nf
        freq_min = args.freq_min * units.Hz
        df = args.df * units.Hz
        beam_type = args.beam_type
        if args.inst_model is not None and uvs_path is None:
            uvs_m = Quantity(
                load_inst_model(args.inst_model)[0][:, :2], unit="m"
            )
        else:
            raise ValueError(
                "No valid uv sampling found. The configuration yaml config "
                "does not contain a path to an instrument model."
            )
    if uvs_path is not None:
        if not isinstance(uvs_path, Path):
            uvs_path = Path(uvs_path)
        if not uvs_path.exists():
            raise FileNotFoundError(f"{uvs_path} does not exist")
        uvs_m = Quantity(np.load(uvs), unit="m")
        if not len(uvs_m.shape) == 2:
            raise ValueError(f"uvs_m should be 2D but has shape {uvs_m.shape}")
        if not uvs_m.shape[1] == 2:
            raise ValueError(
                f"uvs_m should have 2 columns but has {uvs_m.shape[1]}"
            )
    else:
        raise ValueError(
            "No valid uv sampling found. "
            "config and uvs_path cannot both be None"
        )

    required_args = np.array([
        "fov_ra_eor",
        "nf",
        "freq_min",
        "df",
        "beam_type"
    ])
    missing_args = [
        fov_ra_eor is None,
        nf is None,
        freq_min is None,
        df is None,
        beam_type is None
    ]
    assert np.all(np.logical_not(missing_args)), (
        f"The following parameters are required: {', '.join(required_args)}\n"
        f"The following parameters are missing: "
        f"{', '.join(required_args[missing_args])}\n"
        )
    beam_type = beam_type.lower()
    if beam_type in ["gaussian", "airy"]:
        if diam is None and fwhm_deg is None:
            raise ValueError(
                f"One of fwhm_deg or diam must not be None if beam_type is "
                f"{beam_type}"
            )
        if beam_type == "gaussian":
            if Nsigma is None:
                raise ValueError(
                    "Nsigma cannot be none if beam_type is gaussian"
                )
            if Nsigma <= 0:
                raise ValueError("Nsigma must be greater than 0")
    
    # FoV params
    if not isinstance(fov_ra_eor, Quantity):
        fov_ra_eor = Quantity(fov_ra_eor, unit="deg")
    if fov_dec_eor is None:
        fov_dec_eor = fov_ra_eor
    elif not isinstance(fov_dec_eor, Quantity):
        fov_dec_eor = Quantity(fov_dec_eor, unit="deg")
    if fov_ra_fg is None:
        fov_ra_fg = fov_ra_eor
    elif not isinstance(fov_ra_fg, Quantity):
        fov_ra_fg = Quantity(fov_ra_fg, unit="deg")
    if fov_dec_fg is None:
        fov_dec_fg = fov_ra_fg
    elif not isinstance(fov_dec_fg, Quantity):
        fov_dec_fg = Quantity(fov_dec_fg, unit="deg")

    # Frequency params
    if not isinstance(freq_min, Quantity):
        freq_min = Quantity(freq_min, unit="Hz")
    if not isinstance(df, Quantity):
        df = Quantity(df, unit="Hz")
    freqs = freq_min + np.arange(nf)*df
    # The instrument appears largest in the uv plane in wavelengths
    # at the highest frequency (smallest wavelength) in the data
    wavelength = c.to('m/s') / freqs[-1].to('1/s')

    # uv sampling
    uvs = uvs_m / wavelength
    uv_mags = np.sqrt(np.sum(uvs**2, axis=1))
    u_max_inst_model = np.abs(uvs[:, 0]).max() / units.rad
    v_max_inst_model = np.abs(uvs[:, 1]).max() / units.rad

    # Beam params
    beam = Beam(beam_type=beam_type, fwhm_deg=fwhm_deg, diam=diam)
    if beam.beam_type == "airy":
        if beam.diam is not None:
            diam_eff = beam.diam
        else:
            # The effective diameter is largest at the lowest frequency
            diam_eff = beam.fwhm_to_diam(beam.fwhm_deg, freqs[0])
        aperture_width = diam_eff / wavelength / units.rad
    elif beam.beam_type == "gaussian":
        if beam.diam is not None:
            # The effective standard deviation is largest at the
            # lowest frequency
            stddev_eff = beam.diam_to_stddev(beam.diam, freqs[0])
        else:
            stddev_eff = beam.fwhm_to_stddev(beam.fwhm_deg)
        stddev_uv = 1 / (2 * np.pi * stddev_eff)
        aperture_width = stddev_uv * Nsigma

    if plot:
        def plot_model_uv_grid(ax, Nu, delta_u, Nv, delta_v):
            u_centers_labels = (np.arange(Nu) - Nu//2)
            u_centers = u_centers_labels * delta_u.value
            u_edges = (np.arange(Nu + 1) - Nu//2) * delta_u.value
            u_edges -= delta_u.value/2
            ax.set_xticks(u_edges, minor=True)
            ax.set_xlim([u_edges[0], u_edges[-1]])
            v_centers_labels = (np.arange(Nv) - Nv//2)
            v_centers = v_centers_labels * delta_v.value
            v_edges = (np.arange(Nv + 1) - Nv//2) * delta_v.value
            v_edges -= delta_v.value/2
            ax.set_yticks(v_edges, minor=True)
            ax.set_ylim([v_edges[0], v_edges[-1]])
            ax.grid(which='minor')
