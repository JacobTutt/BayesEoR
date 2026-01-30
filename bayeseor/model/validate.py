import numpy as np
from astropy.constants import c
from astropy import units
from astropy.time import Time
from astropy.units import Quantity
import astropy_healpix as ahp
import matplotlib.pyplot as plt
from pathlib import Path
from rich import print as rprint
from rich.panel import Panel

from .beam import Beam
from .instrument import load_inst_model
from .healpix import Healpix
from ..params import BayesEoRParser
from ..setup import get_vis_data
from ..utils import mpiprint


def validate_model(
    config : Path | str | None = None,
    neta : int | None = None,
    nu : int | None = None,
    nv : int | None = None,
    nu_fg : int | None = None,
    nv_fg : int | None = None,
    fit_for_monopole : bool = False,
    nside : int | None = None,
    fov_ra_eor : Quantity | float | None = None,
    fov_dec_eor : Quantity | float | None = None,
    fov_ra_fg : Quantity | float | None = None,
    fov_dec_fg : Quantity | float | None = None,
    simple_za_filter : bool = True,
    single_fov : bool = False,
    beam_type : str | None = None,
    fwhm_deg : Quantity | float | None = None,
    diam : Quantity | float | None = None,
    Nsigma : float | None = 5,
    data_path : Path | str | None = None,
    ant_str : str = "cross",
    bl_cutoff : Quantity | float | None = None,
    freq_idx_min : int | None = None,
    freq_min : Quantity | float | None = None,
    freq_center : Quantity | float | None = None,
    nf : int | None = None,
    df : Quantity | float | None = None,
    nq : int = 0,
    jd_idx_min : int | None = None,
    jd_min : Time | float | None = None,
    jd_center : Time | float | None = None,
    nt : int | None = None,
    dt : Quantity | float | None = None,
    form_pI : bool = True,
    pI_norm : float = 1.0,
    pol : str = "xx",
    redundant_avg : bool = False,
    uniform_redundancy : bool = False,
    inst_model : Path | str | None = None,
    telescope_latlonalt : list[float] | None = None,
    plot : bool = True,
    verbose : bool = True,
    rank : int = 0
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
    verbose : bool, optional
        Verbose output. Defaults to True.
    """
    if config is not None:
        if not isinstance(config, Path):
            config = Path(config)
        if not config.exists():
            raise FileNotFoundError(f"{config} does not exist")
        parser = BayesEoRParser()
        args = parser.parse_args(["--config", config.as_posix()])
        neta = args.neta
        nu = args.nu
        nv = args.nv
        nu_fg = args.nu_fg
        nv_fg = args.nv_fg
        fit_for_monopole = args.fit_for_monopole
        nside = args.nside
        fov_ra_eor = args.fov_ra_eor
        fov_dec_eor = args.fov_dec_eor
        fov_ra_fg = args.fov_ra_fg
        fov_dec_fg = args.fov_dec_fg
        simple_za_filter = args.simple_za_filter
        single_fov = args.single_fov
        beam_type = args.beam_type
        fwhm_deg = args.fwhm_deg
        diam = args.antenna_diameter
        data_path = args.data_path
        ant_str = args.ant_str
        bl_cutoff = args.bl_cutoff
        freq_idx_min = args.freq_idx_min
        freq_min = args.freq_min
        freq_center = args.freq_center
        nf = args.nf
        df = args.df
        nq = args.nq
        jd_idx_min = args.jd_idx_min
        jd_min = args.jd_min
        jd_center = args.jd_center
        nt = args.nt
        dt = args.dt
        form_pI = args.form_pI
        pI_norm = args.pI_norm
        pol = args.pol
        redundant_avg = args.redundant_avg
        uniform_redundancy = args.uniform_redundancy
        inst_model = args.inst_model
        telescope_latlonalt = args.telescope_latlonalt
    else:
        required_kwargs = [
            neta, nu, nv, nu_fg, nv_fg, fit_for_monopole, nside,
            fov_ra_eor, fov_dec_eor, fov_ra_fg, fov_dec_fg,
            simple_za_filter, single_fov, beam_type, fwhm_deg, diam,
            Nsigma, data_path, ant_str, bl_cutoff, freq_idx_min,
            freq_min, freq_center, nf, df, nq, jd_idx_min,
            jd_min, jd_center, nt, dt, form_pI, pI_norm, pol,
            redundant_avg, uniform_redundancy, inst_model,
            telescope_latlonalt
        ]
        missing_kwargs = [kwarg is None for kwarg in required_kwargs]
        required_kwargs = [
            "neta", "nu", "nv", "nu_fg", "nv_fg", "fit_for_monopole", "nside",
            "fov_ra_eor", "fov_dec_eor", "fov_ra_fg", "fov_dec_fg",
            "simple_za_filter", "single_fov", "beam_type", "fwhm_deg", "diam",
            "Nsigma", "data_path", "ant_str", "bl_cutoff", "freq_idx_min",
            "freq_min", "freq_center", "nf", "df", "nq", "jd_idx_min",
            "jd_min", "jd_center", "nt", "dt", "form_pI", "pI_norm", "pol",
            "redundant_avg", "uniform_redundancy", "inst_model",
            "telescope_latlonalt"
        ]
        if np.any(missing_kwargs):
            raise ValueError(
                f"The following parameters are required: "
                f"{', '.join(required_kwargs)}\n"
                f"The following parameters are missing: "
                f"{', '.join(required_kwargs[missing_kwargs])}\n"
            )
    if telescope_latlonalt is None:
        raise ValueError("telescope_latlonalt is required and cannot be None")

    # print_rank will only trigger print if verbose is True and rank == 0
    print_rank = 1 - (verbose and rank == 0)
    if verbose and rank == 0:
        rprint("\n", Panel("Configuration"))
        print(f"{neta                = }")
        print(f"{nu                  = }")
        print(f"{nv                  = }")
        print(f"{nu_fg               = }")
        print(f"{nv_fg               = }")
        print(f"{fit_for_monopole    = }")
        print(f"{nside               = }")
        print(f"{fov_ra_eor          = }")
        print(f"{fov_dec_eor         = }")
        print(f"{fov_ra_fg           = }")
        print(f"{fov_dec_fg          = }")
        print(f"{simple_za_filter    = }")
        print(f"{single_fov          = }")
        print(f"{beam_type           = }")
        print(f"{fwhm_deg            = }")
        print(f"{diam                = }")
        print(f"{Nsigma              = }")
        print(f"{data_path           = }")
        print(f"{ant_str             = }")
        print(f"{bl_cutoff           = }")
        print(f"{freq_idx_min        = }")
        print(f"{freq_min            = }")
        print(f"{freq_center         = }")
        print(f"{nf                  = }")
        print(f"{df                  = }")
        print(f"{nq                  = }")
        print(f"{jd_idx_min          = }")
        print(f"{jd_min              = }")
        print(f"{jd_center           = }")
        print(f"{nt                  = }")
        print(f"{dt                  = }")
        print(f"{form_pI             = }")
        print(f"{pI_norm             = }")
        print(f"{pol                 = }")
        print(f"{redundant_avg       = }")
        print(f"{uniform_redundancy  = }")
        print(f"{inst_model          = }")
        print(f"{telescope_latlonalt = }")

    data_path = Path(data_path)
    if data_path.suffix == ".npy":
        missing_freq_kwargs = (
            nf is None
            or df is None
            or (freq_min is None and freq_center is None)
        )
        if missing_freq_kwargs:
            raise ValueError(
                "nf, df, and one of (freq_min, freq_center) are all "
                "required kwargs when loading a preprocessed data vector "
                "(data_path has a .npy suffix)"
            )
        missing_time_kwargs = (
            nt is None
            or dt is None
            or (jd_min is None and jd_center is None)
        )
        if missing_time_kwargs:
            raise ValueError(
                "nt, dt, and one of (jd_min, jd_center) are all "
                "required kwargs when loading a preprocessed data vector "
                "(data_path has a .npy suffix)"
            )
        if inst_model is None:
            raise ValueError(
                "inst_model is required when loading a preprocessed data "
                "vector (data_path has a .npy suffix)"
            )

    if data_path.suffix == ".npy":
        # Frequency params
        df = Quantity(df, unit="Hz")
        if freq_min is not None:
            freq_min = Quantity(freq_min, unit="Hz")
            freqs = freq_min + np.arange(nf)*df
        else:
            freq_center = Quantity(freq_center, unit="Hz")
            freqs = freq_center + np.arange(-(nf//2), nf//2 + nf%2)*df
        # Time params
        dt = Quantity(dt, unit="s")
        if jd_min is not None:
            jd_min = Time(jd_min, format="jd")
            jds = jd_min + np.arange(nt)*dt.to("d")
        else:
            jd_center = Time(jd_center, format="jd")
            jds = jd_center + np.arange(-(nt//2), nt//2 + nt%2)*dt.to("d")
        # uv sampling
        if not isinstance(inst_model, Path):
            inst_model = Path(inst_model)
        if not inst_model.exists():
            raise FileNotFoundError(f"{inst_model} does not exist")
        uvs_m = Quantity(load_inst_model(inst_model)[0][0, :, :2], unit="m")
    else:
        vis_dict = get_vis_data(
            data_path=data_path,
            ant_str=ant_str,
            bl_cutoff=bl_cutoff,
            freq_idx_min=freq_idx_min,
            freq_min=freq_min,
            freq_center=freq_center,
            nf=nf,
            df=df,
            jd_idx_min=jd_idx_min,
            jd_min=jd_min,
            jd_center=jd_center,
            nt=nt,
            dt=dt,
            form_pI=form_pI,
            pI_norm=pI_norm,
            pol=pol,
            redundant_avg=redundant_avg,
            uniform_redundancy=uniform_redundancy,
            inst_model=inst_model,
            sigma=0,
            verbose=verbose,
            rank=rank
        )
        freqs = Quantity(vis_dict["freqs"], unit="Hz")
        nf = freqs.size
        jds = Time(vis_dict["jds"], format="jd")
        nt = jds.size
        jd_center = jds[nt//2]
        dt = (jds[1] - jds[0]).to("s")
        uvs_m = Quantity(vis_dict["uvws"][0, :, :2], unit="m")

    if neta is None:
        neta = nf
    if nv is None:
        nv = nu
    if nu_fg is None:
        nu_fg = nu
    if nv_fg is None:
        nv_fg = nu_fg

    # Sky model params
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

    hpx = Healpix(
        fov_ra_eor=fov_ra_eor,
        fov_dec_eor=fov_dec_eor,
        fov_ra_fg=fov_ra_fg,
        fov_dec_fg=fov_dec_fg,
        simple_za_filter=simple_za_filter,
        single_fov=single_fov,
        nside=nside,
        telescope_latlonalt=telescope_latlonalt,
        jd_center=jd_center,
        nt=nt,
        dt=dt
    )

    # Beam model params
    beam_type = beam_type.lower()
    if beam_type in ["gaussian", "airy"]:
        if diam is None and fwhm_deg is None:
            raise ValueError(
                f"fwhm_deg or diam is required for a {beam_type} beam"
            )
        if beam_type == "gaussian":
            if Nsigma is None:
                raise ValueError("Nsigma is required if beam_type is gaussian")
            if Nsigma <= 0:
                raise ValueError("Nsigma must be greater than 0")
        if fwhm_deg is not None:
            fwhm_deg = Quantity(fwhm_deg, unit="deg")
        if diam is not None:
            diam = Quantity(diam, unit="m")

    # --- Derived Params ---
    if verbose and rank == 0:
        mpiprint("\n", Panel("Required Model uv-Plane Parameters"))
    nu_req, nv_req, nu_fg_req, nv_fg_req, nside_req = get_uv_model_params(
        fov_ra_eor=fov_ra_eor,
        fov_dec_eor=fov_dec_eor,
        fov_ra_fg=fov_ra_fg,
        fov_dec_fg=fov_dec_fg,
        freqs=freqs,
        beam_type=beam_type,
        fwhm_deg=fwhm_deg,
        diam=diam,
        Nsigma=Nsigma,
        uvs_m=uvs_m,
        plot=plot,
        verbose=False
    )
    if verbose and rank == 0:
        col_str = f"Parameter     Required"
        dash_str = f"---------     --------"
        if np.any([x is not None for x in [nu, nv, nu_fg, nv_fg]]):
            col_str += "     Input"
            dash_str += "     -----"
        mpiprint(f"{col_str}\n{dash_str}")
        mpiprint(f"nu            {nu_req:<13.0f}", end="")
        if nu is not None:
            if nu < nu_req:
                style = "red bold"
            else:
                style = "green"
            mpiprint(f"{nu}", style=style)
        mpiprint(f"nv            {nv_req:<13.0f}", end="")
        if nv is not None:
            if nv < nv_req:
                style = "red bold"
            else:
                style = "green"
            mpiprint(f"{nv}", style=style)
        mpiprint(f"nu_fg         {nu_fg_req:<13.0f}", end="")
        if nu_fg is not None:
            if nu_fg < nu_fg_req:
                style = "red bold"
            else:
                style = "green"
            mpiprint(f"{nu_fg}", style=style)
        mpiprint(f"nv_fg         {nv_fg_req:<13.0f}", end="")
        if nv_fg is not None:
            if nv_fg < nv_fg_req:
                style = "red bold"
            else:
                style = "green"
            mpiprint(f"{nv_fg}", style=style)
    if nu is not None:
        if nu < nu_req:
            mpiprint(
                f"\n[bold red]WARNING:[/bold red] the provided value of nu "
                f"({nu}) is smaller than the required nv ({nu_req}) for the "
                f"instrument and beam model.  Set nu >= {nu_req} to fully "
                f"encompass the specified baselines in the EoR model "
                f"uv-plane.",
                rank=print_rank
            )
        elif nu > nu_req:
            mpiprint(
                f"\n[bold]SUGGESTION:[/bold] the provided value of nu ({nu}) "
                f"is larger than the required nu ({nu_req}) for the given "
                f"instrument and beam model. Consider decreasing nu to "
                f"{nu_req} to reduce runtimes.",
                rank=print_rank
            )
    if nv is not None:
        if nv < nv_req:
            mpiprint(
                f"\n[bold red]WARNING:[/bold red] the provided value of nv "
                f"({nv}) is smaller than the required nv ({nv_req}) for the "
                f"given instrument and beam model.  Set nv >= {nv_req} to "
                f"fully encompass the specified baselines in the EoR model "
                f"uv-plane.",
                rank=print_rank
            )
        elif nv > nv_req:
            mpiprint(
                f"\n[bold]SUGGESTION:[/bold] the provided value of nv ({nv}) "
                f"is larger than the required nv ({nv_req}) for the given "
                f"instrument and beam model. Consider decreasing nv to "
                f"{nv_req} to reduce runtimes.",
                rank=print_rank
            )
    if nu_fg is not None:
        if nu_fg < nu_fg_req:
            mpiprint(
                f"\n[bold red]WARNING:[/bold red] the provided value of nu_fg "
                f"({nu_fg}) is smaller than the required nu_fg ({nu_fg_req}) "
                f"for the given instrument and beam model.  Set nu_fg >= "
                f"{nu_fg_req} to fully encompass the specified baselines in "
                f"the foreground model uv-plane.",
                rank=print_rank
            )
        elif nu_fg > nu_fg_req:
            mpiprint(
                f"\n[bold]SUGGESTION:[/bold] the provided value of nu_fg "
                f"({nu_fg}) is larger than the required nu_fg ({nu_fg_req}) "
                f"for the given instrument and beam model. Consider "
                f"decreasing nu_fg to {nu_fg_req} to reduce runtimes.",
                rank=print_rank
            )
    if nv_fg is not None:
        if nv_fg < nv_fg_req:
            mpiprint(
                f"\n[bold red]WARNING:[/bold red] the provided value of nv_fg "
                f"({nv_fg}) is smaller than the required nv_fg ({nv_fg_req}) "
                f"for the given instrument and beam model.  Set nv_fg >= "
                f"{nv_fg_req} to fully encompass the specified baselines in "
                f"the foreground model uv-plane.",
                rank=print_rank
            )
        elif nv_fg > nv_fg_req:
            mpiprint(
                f"\n[bold]SUGGESTION:[/bold] the provided value of nv_fg "
                f"({nv_fg}) is larger than the required nv_fg ({nv_fg_req}) "
                f"for the given instrument and beam model. Consider "
                f"decreasing nv_fg to {nv_fg_req} to reduce runtimes.",
                rank=print_rank
            )
    
    if verbose and rank == 0:
        mpiprint("\n", Panel("Required Sky Model Parameters"))
        col_str = f"Parameter     Required"
        dash_str = f"---------     --------"
        if nside is not None:
            col_str += "     Input"
            dash_str += "     -----"
        mpiprint(f"{col_str}\n{dash_str}")
        mpiprint(f"nside         {nside_req:<13.0f}", end="")
        if nside is not None:
            if nside < nside_req:
                style = "red bold"
            else:
                style = "green"
            mpiprint(f"{nside}", style=style)
    if nside is not None:
        if nside < nside_req:
            mpiprint(
                f"\n[bold red]WARNING:[/bold red] the provided value of nside "
                f"({nside}) is smaller than the required nside ({nside_req}). "
                f" Set nside >= {nside_req} to Nyquist sample the HEALPix "
                f"sky model.",
                rank=print_rank
            )
        if nside > nside_req:
            mpiprint(
                f"\n[bold]SUGGESTION:[/bold] the provided value of nside "
                f"({nside}) is larger than the required nside ({nside_req}).  "
                f"Consider decreasing nside to {nside_req} to reduce "
                f"runtimes.",
                rank=print_rank
            )
    # Check if any pointing centers lie outside the extent of the
    # EoR or foreground sky model
    ra_start = hpx.pointing_centers[0][0]
    ra_end = hpx.pointing_centers[-1][0]
    delta_ra = ra_end - ra_start
    if delta_ra < 0:
        delta_ra = ra_end + 360 - ra_start
    bad_ptctrs_eor = np.zeros(hpx.nt, dtype=bool)
    bad_ptctrs_fg = np.zeros(hpx.nt, dtype=bool)
    for i_t in range(hpx.nt):
        ra, dec = hpx.pointing_centers[i_t]
        hpx_ind = ahp.lonlat_to_healpix(
            ra*units.deg, dec*units.deg, hpx.nside
        )
        bad_ptctrs_eor[i_t] = hpx_ind not in hpx.pix_eor
        bad_ptctrs_fg[i_t] = hpx_ind not in hpx.pix_fg
    if verbose and rank == 0:
        mpiprint(f"fov_ra_eor    ", end="")
        mpiprint(f">{delta_ra:<12.2f}", end="")
        if np.any(bad_ptctrs_eor):
            style = "bold red"
        else:
            style = "green"
        if single_fov:
            mpiprint(f"{fov_ra_eor.value:.2f}", style=style)
        else:
            mpiprint(
                f"{hpx.fov_ra_eor_eff.value:.2f} (fov_ra_eor_eff)",
                style=style
            )
        mpiprint(f"fov_ra_fg     ", end="")
        mpiprint(f">{delta_ra:<12.2f}", end="")
        if np.any(bad_ptctrs_fg):
            style = "bold red"
        else:
            style = "green"
        if single_fov:
            mpiprint(f"{fov_ra_fg.value:.2f}", style=style)
        else:
            mpiprint(
                f"{hpx.fov_ra_fg_eff.value:.2f} (fov_ra_fg_eff)", style=style
            )
    if np.any(bad_ptctrs_eor):
        bad_inds = list(np.where(bad_ptctrs_eor)[0])
        mpiprint(
            f"\n[bold red]WARNING:[/bold red] the following time indices "
            f"produce pointing centers in the EoR sky model in (RA, Dec) "
            f"that lie outside the included sky model pixels: {bad_inds}."
            f"  Consider increasing fov_ra_eor, setting single_fov = True "
            f"or selecting a dataset with a smaller integration time (dt) "
            f"or fewer integrations (nt) to reduce the range of observed "
            f"RA.",
            rank=print_rank
        )
    if np.any(bad_ptctrs_fg):
        bad_inds = list(np.where(bad_ptctrs_fg)[0])
        mpiprint(
            f"\n[bold red]WARNING:[/bold red] the following time indices "
            f"produce pointing centers in the foreground sky model in (RA,"
            f" Dec) that lie outside the included sky model pixels: "
            f"{bad_inds}.  Consider increasing fov_ra_fg, setting "
            f"single_fov = True or selecting a dataset with a smaller "
            f"integration time (dt) or fewer integrations (nt) to reduce "
            f"the range of observed RA.",
            rank=print_rank
        )
    
    
    # Linear system conditioning
    nmodel = (
        # Number of EoR model parameters
        (neta - 1)*(nu*nv - 1)
        # Number of foreground model parameters
        + (nq + 1)*(nu_fg*nv_fg - 1) + fit_for_monopole*(neta + nq)
    )
    nbls = uvs_m.shape[0]
    ndata = nf * nt * nbls
    if verbose and rank == 0:
        mpiprint("\n", Panel("Linear System Conditioning"))
        mpiprint(f"nmodel         = {nmodel}")
        mpiprint(f"ndata          = {ndata}")
        if ndata / nmodel < 2:
            style = "bold red"
        else:
            style = "green"
        mpiprint(f"ndata / nmodel = ", end="")
        mpiprint(f"{ndata / nmodel}", style=style)
    if ndata / nmodel < 2:
        mpiprint(
            f"[bold]WARNING:[/bold] ndata / nmodel < 2 .  For some datasets, "
            f"this has empirically resulted in an under-constrained system.  "
            f"While correlations in the data prevent this from being a "
            f"straightforward counting exercise, consider increasing the "
            f"number of times in the dataset to be analyzed to afford a "
            f"higher number of data points without increasing the runtime of "
            f"the power spectrum analysis."
        )

def get_uv_model_params(
    *,
    fov_ra_eor : Quantity | float,
    fov_dec_eor : Quantity | float | None = None,
    fov_ra_fg : Quantity | float | None = None,
    fov_dec_fg : Quantity | float | None = None,
    freqs : Quantity | np.ndarray,
    beam_type : str,
    fwhm_deg : Quantity | float = None,
    diam : Quantity | float = None,
    Nsigma : float = 5,
    uvs_m : Quantity | np.ndarray,
    plot : bool = False,
    verbose : bool = True
):
    """
    Calculate the required nu (nv) given sky and instrument model parameters.

    Parameters
    ----------
    fov_ra_eor : astropy.units.Quantity or float, optional
        Field of view of the Right Ascension axis of the EoR sky model in
        degrees.
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
    freqs : astropy.units.Quantity or np.ndarray
        Frequencies in Hz if not a Quantity.
    beam_type : str
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
    uvs_m : astropy.units.Quantity or np.ndarray
        2D array of sampled (u, v), in units of meters if not a Quantity,
        with shape (Nbls, 2), where Nbls is the number of baselines and the
        u (v) axis is indexed by the 0th (1st) column index, i.e.
        `u = uvs[:, 0]` and `v = uvs[:, 1]`.
    plot : bool, optional
        Plot the model uv planes for the EoR and foreground models with the
        model grid and aperture function as a circle with the diameter
        calculated based on the beam model in wavelengths. The aperture
        function is only plotted for the baseline along the u (v) axis with
        the largest |u| (|v|). Defaults to False.
    verbose : bool, optional
        Verbose output. Defaults to True.

    Returns
    -------
    nu: int
        Number of pixels on a side for the u-axis in the EoR model uv-plane.
    nv: int
        Number of pixels on a side for the v-axis in the EoR model uv-plane.
    nu_fg: int
        Number of pixels on a side for the u-axis in the foreground model
        uv-plane.
    nv_fg: int
        Number of pixels on a side for the v-axis in the foreground model
        uv-plane.
    nside: int
        HEALPix nside which satisfies Nyquist sampling.
    """
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
    
    # --- BayesEoR model parameters ---
    # Model uv plane
    delta_u_eor = 1 / fov_ra_eor.to('rad')
    delta_v_eor = 1 / fov_dec_eor.to('rad')
    delta_u_fg = 1 / fov_ra_fg.to('rad')
    delta_v_fg = 1 / fov_dec_fg.to('rad')

    """
    The FoV along the RA and Dec axes set the separation between adjacent u
    and v modes in the model uv plane, respectively.  We need to choose the
    number of model uv plane grid points such that we Nyquist sample the image
    domain which equates to having two image domain pixels per minimum fringe
    wavelength.  The logic is identical for v.

    The minimum fringe wavelength is `1 / u_max` where `u_max` is the maximum
    u sampled by the instrument along the u axis.  We add a buffer of half the
    width of the aperture function to this u_max, i.e.

    u'_max = u_max + 0.5 * aperture_width

    We thus need to choose u for the model uv plane which produces a fringe
    wavelength which is smaller than the minimum fringe wavelength sampled by
    the instrument, i.e. we need to solve

    1 / u <= 1 / u'_max    ==>    u'_max <= u

    Given that we have a rectilinear grid for the model uv plane and the
    spacing between adjacent u is given by

    delta_u = 1 / FoV_RA    (for delta_v we replace FoV_RA with FoV_Dec)

    we must choose N_u such that

    u'_max <= N_u * delta_u    ==>    N_u >= u'_max / delta_u
    """
    nu = int(
        np.ceil((u_max_inst_model + 0.5*aperture_width) / delta_u_eor)
    )
    nv = int(
        np.ceil((v_max_inst_model + 0.5*aperture_width) / delta_v_eor)
    )
    nu_fg = int(np.ceil((u_max_inst_model + 0.5*aperture_width) / delta_u_fg))
    nv_fg = int(np.ceil((v_max_inst_model + 0.5*aperture_width) / delta_v_fg))

    # The calculation above determines the number of model uv plane pixels
    # required for u > 0 (v > 0).  But, the model uv plane in BayesEoR is
    # specified for all u (v) (positive and negative).  The model uv plane
    # also requires an odd number of pixels along the u (v) axis for modelling
    # the (u, v)=(0, 0) monopole.
    nu = nu * 2 + 1
    nv = nv * 2 + 1
    nu_fg = nu_fg * 2 + 1
    nv_fg = nv_fg * 2 + 1

    # Sky model

    """
    The sky model and model uv plane must satisfy Nyquist sampling to avoid
    any spurious errors in the analysis.  In this case, Nyquist sampling
    requires at least two image domain pixels per minimum fringe wavelength.
    The minimum fringe wavelength is the inverse of the maximum sampled |u|,
    i.e.

    min_fringe_wavelength = 1 / |u| = 1 / sqrt(u^2 + v^2)

    Given this wavelength, we then need to choose the nside of the sky model
    such that the pixel width, calculated as the square root of the pixel
    area, is less than or equal to `min_fringe_wavelength / 2` or

    2 * pixel_width(nside) <= min_fringe_wavelength
    """
    u_max_uv_model = 1 / units.rad * np.max((
        delta_u_eor.to('1/rad').value*(nu//2 - 1),
        delta_u_fg.to('1/rad').value*(nu_fg//2 - 1)
    ))
    v_max_uv_model = 1 / units.rad * np.max((
        delta_v_eor.to('1/rad').value*(nv//2 - 1),
        delta_v_fg.to('1/rad').value*(nv_fg//2 - 1)
    ))
    uv_max_uv_model = np.sqrt(u_max_uv_model**2 + v_max_uv_model**2)
    min_fringe_wavelength = 1 / uv_max_uv_model

    nside = 16  # initial guess
    while (
        2*ahp.nside_to_pixel_resolution(nside).to("rad")
        > min_fringe_wavelength
    ):
        nside *= 2
    
    if verbose:
        rprint("\n", Panel("Configuration"))
        print(f"{fov_ra_eor        = :f}")
        print(f"{fov_dec_eor       = :f}")
        print(f"{fov_ra_fg         = :f}")
        print(f"{fov_dec_fg        = :f}")
        print(f"{freqs             = }")
        print(f"{beam_type         = }")
        if beam_type == "airy":
            print(f"{diam              = :f}")
        elif beam_type == "gaussian":
            print(f"fwhm_deg          = {fwhm_deg.to('deg'):f}")
            print(f"Nsigma            = {Nsigma:f}")

        rprint(Panel("BayesEoR Model Parameters"))
        print(f"{nu = }")
        print(f"{nv = }")
        print(f"{nu_fg  = }")
        print(f"{nv_fg  = }")
        print(f"{nside  = }", end="\n\n")
    
    if plot:
        fig, axs = plt.subplots(
            1, 2, figsize=(21, 10), sharey=False, gridspec_kw={'wspace': 0.1}
        )

        axs[0].set_title('EoR Model')
        axs[1].set_title('FG Model')

        for ax in axs:
            # Plot the uv sampling of the instrument
            ax.scatter(
                uvs[:, 0],
                uvs[:, 1],
                color='k',
                marker='o',
                label='UV Sampling'
            )

        # Plot aperture width as a circle around the max u (v)
        u_max_ind = np.where(uvs[:, 0] == u_max_inst_model.value)
        v_max_ind = np.where(uvs[:, 1] == v_max_inst_model.value)
        for ax in axs:
            circle_u = plt.Circle(
                *uvs[u_max_ind].value, aperture_width.value/2, ec='k',
                fc='none'
            )
            label = 'Aperture function width'
            if beam_type == "gaussian":
                label += fr' ($N_\sigma$ = {Nsigma})'
            circle_v = plt.Circle(
                *uvs[v_max_ind].value, aperture_width.value/2, ec='k',
                fc='none', label=label
            )
            ax.add_patch(circle_u)
            ax.add_patch(circle_v)

        plot_model_uv_grid(axs[0], nu, delta_u_eor, nv, delta_v_eor)
        plot_model_uv_grid(axs[1], nu_fg, delta_u_fg, nv_fg, delta_v_fg)

        xmin = np.min((axs[0].get_xlim()[0], axs[1].get_xlim()[0]))
        xmax = np.max((axs[0].get_xlim()[1], axs[1].get_xlim()[1]))
        ymin = np.min((axs[0].get_ylim()[0], axs[1].get_ylim()[0]))
        ymax = np.max((axs[0].get_ylim()[1], axs[1].get_ylim()[1]))
        for ax in axs:
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set_aspect('equal')
            ax.legend(loc='upper right')
            ax.set_xlabel(r'$u$ [$\lambda$]')
        axs[0].set_ylabel(r'$v$ [$\lambda$]')

        plt.show()
    
    return nu, nv, nu_fg, nv_fg, nside