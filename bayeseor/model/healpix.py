"""
Interface for the HEALPix image domain model in BayesEoR.
"""

import numpy as np
from astropy_healpix import HEALPix
from astropy_healpix import healpy as hp
from astropy.coordinates import\
    EarthLocation, AltAz, ICRS, SkyCoord
from astropy.time import Time
import astropy.units as units
from astropy.units import Quantity
from astropy.constants import c
from functools import reduce
from pyuvdata import UVBeam
from pyuvdata import utils as uvutils
from scipy.special import j1
import warnings

c_ms = c.to("m/s").value

HERA_LAT_LON_ALT = (
    -30.72152777777791,  # deg
    21.428305555555557,  # deg
    1073.0000000093132  # meters
)

class Healpix(HEALPix):
    """
    HEALPix-related functionality for the image domain model.

    Parameters
    ----------
    # TODO: make sure fov_ra_eor used as Quantity!
    fov_ra_eor : astropy.units.Quantity or float
        Field of view, in degrees if not a Quantity, of the RA axis of the
        EoR sky model.
    # TODO: make sure fov_dec_eor used as Quantity!
    fov_dec_eor : astropy.units.Quantity or float, optional
        Field of view, in degrees if not a Quantity, of the Dec axis of the
        EoR sky model. Defaults to `fov_ra_eor`.
    # TODO: make sure fov_ra_fg used as Quantity!
    fov_ra_fg : astropy.units.Quantity or float, optional
        Field of view, in degrees if not a Quantity, of the RA axis of the
        foreground sky model. Must be greater than or equal to `fov_ra_eor`.
        Defaults to `fov_ra_eor`.
    # TODO: make sure fov_dec_fg used as Quantity!
    fov_dec_fg : astropy.units.Quantity or float, optional
        Field of view, in degrees if not a Quantity, of the Dec axis of the
        foreground sky model. Must be greater than or equal to `fov_dec_eor`.
        Defaults to `fov_ra_fg`.
    simple_za_filter : bool, optional
        Filter pixels in the FoV by zenith angle only. Otherwise, filter
        pixels in a rectangular region. See the docstring for
        :func:`.get_pixel_filter` for more details. We strongly suggest
        leaving `simple_za_filter` as True as the rectangular pixel selection
        is not always reliable (please see BayesEoR issue #11 for more
        details). Defaults to True.
    # TODO: add single_fov to bayeseor.params and bayeseor.matrices.build
    single_fov : bool, optional
        Use a single field of view at the central time step to form the sky
        model pixel mask(s). Otherwise, calculate the pixel masks for each
        time and form the total pixel masks as the union of the pixel indices
        at each time. See the docstring for :func:`.get_pixel_filter` for more
        details. This setting can be enabled to reproduce the results in
        Burba+23a (2023MNRAS.520.4443B). Defaults to False.
    nside : int
        Nside resolution of the HEALPix map. Defaults to 256.
    telescope_latlonalt : tuple or list of float, optional
        Tuple containing the latitude, longitude, and altitude of the
        telescope in degrees, degrees, and meters, respectively. Defaults
        to the location of the HERA telescope, i.e. (-30.72152777777791, 
        21.428305555555557, 1073.0000000093132).
    # TODO: make sure jd_center used as Time!
    jd_center : astropy.time.Time or float
        Central time as a Julian date if not a Time.
    nt : int, optional
        Number of times. Defaults to 1.
    # TODO: make sure dt used as Quantity!
    dt : astropy.units.Quantity or float, optional
        Integration time in seconds if not a Quantity. Required if `nt` > 1.
        Defaults to None.
    beam_type : str, optional
        Can be either a path to a pyuvdata-compatible beam file or one of
        'uniform', 'gaussian', 'airy', 'gausscosine', or 'taperairy'. Defaults
        to 'uniform'.
    peak_amp : float, optional
        Peak amplitude of the beam. Defaults to 1.0.
    # TODO: make sure fwhm_deg used as Quantity!
    fwhm_deg : float, optional
        Full width at half maximum (FWHM) of the beam in degrees if not a
        Quantity. Used if `beam_type` is 'airy', 'gaussian', or 'gausscosine'.
        If `beam_type` is 'airy', the effective antenna diameter is calculated
        from the FWHM at a user-specified frequency.
    # TODO: make sure diam used as Quantity!
    diam : astropy.units.Quantity or float, optional
        Antenna (aperture) diameter in meters if not a Quantity. Used if
        `beam_type` is 'airy', 'gaussian', or 'gausscosine'. If `beam_type` is
        'gaussian' or 'gausscosine', the effective full width at half maximum
        is calculated from `diam` at a user-specified frequency.
    # TODO: make sure cosfreq used as Quantity!
    cosfreq : asropy.units.Quantity or float, optional
        Cosine frequency, in inverse degrees if not a Quantity, which
        multiplies a Gaussian beam to produce sidelobe-like structure. Used if
        `beam_type` is 'gausscosine'.
    # TODO: make sure tanh_freq used as Quantity!
    tanh_freq : asropy.units.Quantity or float, optional
        Exponential frequency (rate parameter) in inverse degrees if not a
        Quantity. Used if `beam_type` is 'tanhairy' to apply an exponential
        taper to an Airy pattern to suppress sidelobes.
    tanh_sl_red : float, optional
        Airy sidelobe amplitude reduction as a fractional percent. For
        example, passing 0.99 reduces the sidelobes by 0.01, i.e. two orders
        of magnitude. Used if `beam_type` is 'tanhairy'.
    pol : str, optional
        Polarization string. Can be 'xx', 'yy', or 'pI'. Only used if
        `beam_type` is a path to a pyuvdata-compatible beam file. Defaults to
        'xx'.
    freq_interp_kind : str, optional
        Frequency interpolation kind. Please see `scipy.interpolate.interp1d`
        for valid options and more details. Defaults to 'cubic'.

    """
    def __init__(
        self,
        *,
        fov_ra_eor : Quantity | float,
        fov_dec_eor : Quantity | float | None = None,
        fov_ra_fg : Quantity | float | None = None,
        fov_dec_fg : Quantity | float | None = None,
        simple_za_filter : bool = True,
        single_fov : bool = False,
        nside : int = 256,
        telescope_latlonalt : tuple[float] | list[float] = HERA_LAT_LON_ALT,
        jd_center : Time | float | None = None,
        nt : int = 1,
        dt : Quantity | float | None = None,
        beam_type : str | None = None,
        peak_amp : float = 1.0,
        fwhm_deg : Quantity | float | None = None,
        diam : Quantity | float | None = None,
        cosfreq : Quantity | float | None = None,
        tanh_freq : Quantity | float | None = None,
        tanh_sl_red : float | None = None,
        pol : str = "xx",
        freq_interp_kind : str = "cubic"
    ):
        if not simple_za_filter:
            warnings.warn(
                "It is advised to set `simple_za_filter` to True to avoid "
                "issues with the rectangular pixel selection.  Please see "
                "BayesEoR issue #11 for more details.  `simple_za_filter` "
                "should only be set to False if reproducing results from "
                "Burba+23a (2023MNRAS.520.4443B)."
            )

        # Use HEALPix as parent class to get useful astropy_healpix functions
        super().__init__(nside, frame=ICRS())

        if not isinstance(fov_ra_eor, Quantity):
            fov_ra_eor = Quantity(fov_ra_eor, unit="deg")
        self.fov_ra_eor = fov_ra_eor.to("deg")
        if fov_dec_eor is None:
            fov_dec_eor = fov_ra_eor
        elif not isinstance(fov_dec_eor, Quantity):
            fov_dec_eor = Quantity(fov_dec_eor, unit="deg")
        self.fov_dec_eor = fov_dec_eor.to("deg")

        if fov_ra_fg is None:
            fov_ra_fg = fov_ra_eor
        else:
            if not isinstance(fov_ra_fg, Quantity):
                fov_ra_fg = Quantity(fov_ra_fg, unit="deg")
            if fov_ra_fg.to("deg") < fov_ra_eor.to("deg"):
                raise ValueError(
                    "fov_ra_fg must be greater than or equal to fov_ra_eor"
                )
        self.fov_ra_fg = fov_ra_fg.to("deg")
        if fov_dec_fg is None:
            fov_dec_fg = fov_ra_fg
        else:
            if not isinstance(fov_dec_fg, Quantity):
                fov_dec_fg = Quantity(fov_dec_fg, unit="deg")
            if fov_dec_fg.to("deg") < fov_dec_eor.to("deg"):
                raise ValueError(
                    "fov_dec_fg must be greater than or equal to fov_dec_eor"
                )
        self.fov_dec_fg = fov_dec_fg.to("deg")

        self.fovs_match = np.logical_and(
            self.fov_ra_eor == self.fov_ra_fg,
            self.fov_dec_eor == self.fov_dec_fg
        )
        
        self.tele_lat = telescope_latlonalt[0] * units.deg
        self.tele_lon = telescope_latlonalt[1] * units.deg
        self.tele_alt = telescope_latlonalt[2] * units.m
        # Set telescope location
        telescope_xyz = uvutils.XYZ_from_LatLonAlt(
            self.tele_lat.to("rad").value,
            self.tele_lon.to("rad").value,
            self.tele_alt.to("m").value
        )
        self.tele_loc = EarthLocation.from_geocentric(*telescope_xyz, unit="m")

        # Set time axis params for calculating (l(t),  m(t), n(t))
        self.nt = nt
        if jd_center is not None and not isinstance(jd_center, Quantity):
            jd_center = Time(jd_center, format="jd")
        self.jd_center = jd_center
        if dt is not None and not isinstance(dt, Quantity):
            dt = Quantity(dt, unit="s")
        self.dt = dt
        if self.nt % 2:
            self.time_inds = np.arange(-(self.nt//2), self.nt//2 + 1)
        else:
            self.time_inds = np.arange(-(self.nt//2), self.nt//2)
        # Calculate JD per integration from `jd_center`
        if self.dt is not None:
            self.jds = self.jd_center + self.time_inds*self.dt
        else:
            self.jds = np.array([self.jd_center])

        # Calculate pointing center per integration
        self.pointing_centers = []
        for jd in self.jds:
            zen = AltAz(
                alt=90*units.deg,
                az=0*units.deg,
                obstime=jd,
                location=self.tele_loc
            )
            zen_radec = zen.transform_to(ICRS())
            self.pointing_centers.append((zen_radec.ra.deg, zen_radec.dec.deg))
        self.field_center = self.pointing_centers[self.nt//2]

        # Beam params
        if beam_type is not None:
            if not "." in str(beam_type):
                beam_type = beam_type.lower()
                allowed_types = [
                    "uniform", "gaussian", "airy", "gausscosine", "taperairy",
                    "tanhairy"
                ]
                if not beam_type in allowed_types:
                    raise ValueError(
                        f"Only {', '.join(allowed_types)} beams are supported."
                    )
                self.beam_type = beam_type
                self.uvb = None
            else:
                # assume beam_type is a path to a UVBeam compatible file
                uvb = UVBeam()
                uvb.read_beamfits(beam_type)
                if uvb.beam_type == "efield" and pol in ["xx", "yy"]:
                    uvb.efield_to_power()
                elif uvb.beam_type == "efield" and pol == "pI":
                    uvb.efield_to_pstokes()
                uvb.select(polarizations=[uvutils.polstr2num(pol)])
                uvb.freq_interp_kind = freq_interp_kind
                if uvb.pixel_coordinate_system == "healpix":
                    uvb.interpolation_function = "healpix_simple"
                else:
                    uvb.interpolation_function = "az_za_simple"
                self.beam_type = "uvbeam"
                self.uvb = uvb
        else:
            self.beam_type = "uniform"
            self.uvb = None
        self.peak_amp = peak_amp

        if beam_type in ["gaussian", "airy", "taperairy"]:
            required_params = [diam, fwhm_deg]
            if not self._check_required_params(required_params, all_req=False):
                raise ValueError(
                    f"If using a {beam_type} beam, must pass either "
                    "'fwhm_deg' or 'diam'."
                )
        elif beam_type == "gausscosine":
            required_params = [fwhm_deg, cosfreq]
            if not self._check_required_params(required_params):
                raise ValueError(
                    "If using a gausscosine beam, must pass 'fwhm_deg' and "
                    "'cosfreq'."
                )
        elif beam_type == "tanhairy":
            required_params = [diam, tanh_freq, tanh_sl_red]
            if not self._check_required_params(required_params):
                raise ValueError(
                    "If using a tanhairy beam, must pass 'diam', 'tanh_freq', "
                    "and 'tanh_sl_red'."
                )
        if fwhm_deg is not None and not isinstance(fwhm_deg, Quantity):
            fwhm_deg = Quantity(fwhm_deg, unit="deg")
        if diam is not None and not isinstance(diam, Quantity):
            diam = Quantity(diam, unit="m")
        if cosfreq is not None and not isinstance(cosfreq, Quantity):
            cosfreq = Quantity(cosfreq, unit="1/deg")
        if tanh_freq is not None and not isinstance(tanh_freq, Quantity):
            tanh_freq = Quantity(tanh_freq, unit="1/deg")
        self.fwhm_deg = fwhm_deg
        self.diam = diam
        self.cosfreq = cosfreq
        self.tanh_freq = tanh_freq
        self.tanh_sl_red = tanh_sl_red

        # Pixel filters
        self.simple_za_filter = simple_za_filter
        self.single_fov = single_fov
        pix_eor, ra_eor, dec_eor = self.get_pixel_filter(
            fov_ra=self.fov_ra_eor,
            fov_dec=self.fov_dec_eor,
            return_radec=True,
            simple_za_filter=self.simple_za_filter,
            single_fov=self.single_fov
        )
        self.pix_eor = pix_eor
        self.ra_eor = ra_eor
        self.dec_eor = dec_eor
        self.npix_fov_eor = self.pix_eor.size
        if not self.single_fov:
            # Effective FoV along the RA axis due to combining multiple
            # FoVs across all times
            skycoord = SkyCoord(
                self.ra_eor*units.deg, self.dec_eor*units.deg, frame="icrs"
            )
            altaz = skycoord.transform_to(
                AltAz(obstime=self.jd_center, location=self.tele_loc)
            )
            self.fov_ra_eor_eff = 2 * (90 - altaz.alt.deg).max() * units.deg

        if self.fovs_match:
            self.pix_fg = self.pix_eor.copy()
            self.ra_fg = self.ra_eor.copy()
            self.dec_fg = self.dec_eor.copy()
            self.npix_fov_fg = self.pix_fg.size
            if not self.single_fov:
                self.fov_ra_fg_eff = self.fov_ra_eor_eff
        else:
            pix_fg, ra_fg, dec_fg = self.get_pixel_filter(
                fov_ra=self.fov_ra_fg,
                fov_dec=self.fov_dec_fg,
                return_radec=True,
                simple_za_filter=self.simple_za_filter,
                single_fov=self.single_fov
            )
            self.pix_fg = pix_fg
            self.npix_fov_fg = self.pix_fg.size
            self.ra_fg = ra_fg
            self.dec_fg = dec_fg
            if not self.single_fov:
                # Effective FoV along the RA axis due to combining multiple
                # FoVs across all times
                skycoord = SkyCoord(
                    self.ra_fg*units.deg, self.dec_fg*units.deg, frame="icrs"
                )
                altaz = skycoord.transform_to(
                    AltAz(obstime=self.jd_center, location=self.tele_loc)
                )
                self.fov_ra_fg_eff = 2 * (90 - altaz.alt.deg).max() * units.deg
        self.pix = self.pix_fg
        self.ra = self.ra_fg
        self.dec = self.dec_fg
        self.npix_fov = self.npix_fov_fg
        self.fov_ra = self.fov_ra_fg
        self.fov_dec = self.fov_dec_fg
        if not self.single_fov:
            self.fov_ra_eff = self.fov_ra_fg_eff
        # If the FoV values of the two models are different, so to are their
        # HEALPix pixel index arrays.  This mask allows you to take a set of
        # pixel values for the EoR model and propagate them into the FG model.
        self.eor_to_fg_pix = np.in1d(self.pix_fg, self.pix_eor)

    def get_pixel_filter(
        self,
        *,
        fov_ra : Quantity | float,
        fov_dec : Quantity | float | None = None,
        return_radec : bool = False,
        inverse : bool = False,
        simple_za_filter : bool = True,
        single_fov : bool = False
    ):
        """
        Return HEALPix pixel indices lying inside an observed region.

        Parameters
        ----------
        fov_ra : astropy.units.Quantity or float
            Field of view in degrees if not a Quantity. If `fov_dec` is None,
            `fov_ra` represents the diameter of a circular region or the arc
            length of each side of a rectangular region centered on each
            pointing center if `single_fov` is False or only the pointing
            center at the central time, `self.jd_center`, if `single_fov` is
            True.
        fov_dec : astropy.units.Quantity or float, optional
            Field of view of the DEC axis in degrees if not a Quantity.
            `fov_dec` is only used if `simple_za_filter` is False and sets
            the arc length along the DEC axis of the rectangular pixel
            selection. Defaults to `fov_ra`.
        return_radec : bool, optional
            Return the (RA, DEC) coordinates associated with each pixel center.
            Defaults to False.
        inverse : bool, optional
            Return the pixels within (outside) the observed region if True
            (False). Defaults to True.
        simple_za_filter : bool, optional
            Return the pixels inside a circular region with diameter `fov_ra`
            (True, default). Otherwise, return the pixels inside a rectangular
            region with equal arc length on all sides. Defaults to True.
        single_fov : bool, optional
            Use only the central time to form the pixel mask. Otherwise,
            calculate the pixel mask as the union of masks across all times.
            `single_fov` should be set to True to reproduce results from
            Burba+23a (2023MNRAS.520.4443B). Defaults to False.

        Returns
        -------
        pix : numpy.ndarray
            HEALPix pixel indices.
        ra : numpy.ndarray
            RA values for each pixel center. Only returned if `return_radec`
            is True.
        dec : numpy.ndarray
            DEC values for each pixel center. Only returned if `return_radec`
            is True.
        
        Notes
        -----
        * The rectangular pixel selection (`simple_za_filter` is False) has
          been left for posterity. It has been found to be flawed (see issue
          #11 in the BayesEoR repo for more details). We advise setting
          `simple_za_filter` to True (default) to avoid any potential issues
          with the rectangular pixel selection.

        """
        if isinstance(fov_ra, Quantity):
            fov_ra = fov_ra.to("deg").value
        if fov_dec is None:
            fov_dec = fov_ra
        elif isinstance(fov_dec, Quantity):
            fov_dec = fov_dec.to("deg").value
        ras, decs = hp.pix2ang(self.nside, np.arange(self.npix), lonlat=True)

        pointing_centers = self.pointing_centers
        jds = self.jds
        if single_fov:
            pointing_centers = [pointing_centers[self.nt//2]]
            jds = [jds[self.nt//2]]
        pix_all = []
        for jd, ra_dec in zip(jds, pointing_centers):
            if simple_za_filter:
                _, _, _, _, za = self.calc_lmn_from_radec(
                    jd.jd, ras, decs, return_azza=True
                )
                if inverse:
                    pix = np.where(za > np.deg2rad(fov_ra/2))[0]
                else:
                    pix = np.where(za <= np.deg2rad(fov_ra/2))[0]
            else:
                # This rectangular pixel selection functionality has been left
                # in place for posterity only. It can be used to reproduce
                # results from Burba+23a (2023MNRAS.520.4443B), otherwise it
                # should not be used. Please see BayesEoR issue #11 for more
                # details.
                thetas = (90 - decs) * np.pi/180
                if ra_dec[0] - fov_ra/2 < 0:
                    ras[ras > 180] -= 360  # RA in (-180, 180]
                ras_inds = np.logical_and(
                    (ras - ra_dec[0])*np.sin(thetas) >= -fov_ra/2,
                    (ras - ra_dec[0])*np.sin(thetas) <= fov_ra/2,
                )
                decs_inds = np.logical_and(
                    decs >= ra_dec[1] - fov_dec/2,
                    decs <= ra_dec[1] + fov_dec/2
                )
                if inverse:
                    pix = np.where(np.logical_not(ras_inds * decs_inds))[0]
                else:
                    pix = np.where(ras_inds * decs_inds)[0]
                ras[ras < 0] += 360  # RA in [0, 360)
            pix_all.append(pix)

        pix_unique = reduce(np.union1d, pix_all)
        if not return_radec:
            return pix_unique
        else:
            return pix_unique, ras[pix_unique], decs[pix_unique]
    
    def get_extent_ra_dec(self, fov_ra, fov_dec, fov_fac=1.0):
        """
        Get the sampled extent of the sky in RA and DEC.

        Parameters
        ----------
        fov_ra : float
            Field of view in degrees of the RA axis.
        fov_dec : float
            Field of view in degrees of the DEC axis.
        fov_fac : float
            Scaling factor for the sampled extent.

        Returns
        -------
        range_ra : tuple
            `fov_fac` scaled (min, max) sampled RA values.
        range_dec : tuple
            `fov_fac` scaled (min, max) sampled DEC values.

        """
        range_ra = [
            self.field_center[0] - fov_fac*fov_ra/2,
            self.field_center[0] + fov_fac*fov_ra/2
        ]
        range_dec = [
            self.field_center[1] - fov_fac*fov_dec/2,
            self.field_center[1] + fov_fac*fov_dec/2
        ]
        return range_ra, range_dec

    def calc_lmn_from_radec(
            self, time, ra, dec, return_azza=False, radec_offset=None):
        """
        Return arrays of (l, m, n) coordinates in radians for all (RA, DEC).

        Parameters
        ----------
        time : float
            Julian date used in ICRS to AltAz coordinate frame conversion.
        ra : numpy.ndarray
            RA values in degrees.
        dec : numpy.ndarray
            DEC values in degrees.
        return_azza : bool
            If True, return both (l, m, n) and (az, za) coordinate arrays.
            Otherwise return only (l, m, n). Defaults to 'False'.
        radec_offset : tuple of floats
            Will likely be deprecated.

        Returns
        -------
        l : numpy.ndarray of floats
            Array containing the EW direction cosine of each HEALPix pixel.
        m : numpy.ndarray of floats
            Array containing the NS direction cosine of each HEALPix pixel.
        n : numpy.ndarray of floats
            Array containing the radial direction cosine of each HEALPix pixel.

        Notes
        -----
        * Adapted from `pyradiosky.skymodel.update_positions` 
          (https://github.com/RadioAstronomySoftwareGroup/pyradiosky).

        """
        if not isinstance(time, Time):
            time = Time(time, format="jd")

        skycoord = SkyCoord(ra*units.deg, dec*units.deg, frame="icrs")
        altaz = skycoord.transform_to(
            AltAz(obstime=time, location=self.tele_loc)
        )
        az = altaz.az.rad
        za = np.pi/2 - altaz.alt.rad

        # Convert from (az, za) to (l, m, n)
        ls = np.sin(za) * np.sin(az)
        ms = np.sin(za) * np.cos(az)
        ns = np.cos(za)

        if return_azza:
            return ls, ms, ns, az, za
        else:
            return ls, ms, ns

    def get_beam_vals(self, az, za, freq=None):
        """
        Get an array of beam values from (az, za) coordinates.
        
        If `self.beam_type` is 'gaussian', this function assumes that the
        beam width is symmetric along the l and m axes.

        Parameters
        ----------
        az : numpy.ndarray of floats
            Azimuthal angle of each pixel in radians.
        za : numpy.ndarray of floats
            Zenith angle of each pixel in radians.
        freq : float, optional
            Frequency in Hz.

        Returns
        -------
        beam_vals : numpy.ndarray
            Array containing beam amplitude values at each (az, za).

        """
        if self.beam_type == "uniform":
            beam_vals = np.ones(self.npix_fov)

        elif self.beam_type in ["gaussian", "gausscosine"]:
            if self.fwhm_deg is not None:
                stddev_rad = np.deg2rad(
                    self._fwhm_to_stddev(self.fwhm_deg)
                )
            else:
                stddev_rad = self._diam_to_stddev(self.diam, freq)
            if self.beam_type == "gaussian":
                beam_vals = self._gaussian_za(za, stddev_rad, self.peak_amp)
            else:
                beam_vals = self._gausscosine(
                    za, stddev_rad, self.peak_amp, self.cosfreq
                )

        elif self.beam_type == "airy":
            if self.diam is not None:
                beam_vals = self._airy_disk(za, self.diam, freq)
            else:
                diam_eff = self._fwhm_to_diam(self.fwhm_deg, freq)
                beam_vals = self._airy_disk(za, diam_eff, freq)
        
        elif self.beam_type == "taperairy":
            stddev_rad = np.deg2rad(self._fwhm_to_stddev(self.fwhm_deg))
            beam_vals = (
                self._airy_disk(za, self.diam, freq)
                * self._gaussian_za(za, stddev_rad, self.peak_amp)
            )
        
        elif self.beam_type == "tanhairy":
            beam_vals = (
                self._airy_disk(za, self.diam, freq)
                * self._tanh_taper(za, self.tanh_freq, self.tanh_sl_red)
            )
        
        elif self.beam_type == "uvbeam":
            beam_vals, _ = self.uvb.interp(
                az_array=az, za_array=za, freq_array=np.array([freq]),
                reuse_spline=True
            )
            beam_vals = beam_vals[0, 0, 0, 0].real

        return beam_vals

    def _gaussian_za(self, za, sigma, amp):
        """
        Calculate azimuthally symmetric Gaussian beam amplitudes.

        Parameters
        ----------
        za : numpy.ndarray
            Zenith angle of each pixel in radians.
        sigma : float
            Standard deviation in radians.
        amp : float
            Peak amplitude at zenith.

        Returns
        -------
        beam_vals : numpy.ndarray
            Array of Gaussian beam amplitudes for each zenith angle in `za`.

        """
        beam_vals = amp * np.exp(-za ** 2 / (2 * sigma ** 2))
        return beam_vals
    
    def _gausscosine(self, za, sigma, amp, cosfreq):
        """
        Calculate azimuthally symmetric Gaussian * cosine^2 beam amplitudes.

        Parameters
        ----------
        za : numpy.ndarray
            Zenith angle of each pixel in radians.
        sigma : float
            Standard deviation in radians.
        amp : float
            Peak amplitude at zenith.
        cosfreq : float
            Cosine squared frequency in inverse radians.

        Returns
        -------
        beam_vals : numpy.ndarray
            Array of Gaussian beam amplitudes for each zenith angle in `za`.

        """
        beam_vals = amp * np.exp(-za ** 2 / (2 * sigma ** 2))
        beam_vals *= np.cos(2 * np.pi * za * cosfreq/2)**2
        return beam_vals

    def _fwhm_to_stddev(self, fwhm):
        """
        Calculate standard deviation from full width at half maximum.

        Parameters
        ----------
        fwhm : float
            Full width half maximum in degrees.

        """
        return fwhm / 2.355

    def _airy_disk(self, za, diam, freq):
        """
        Calculate Airy disk amplitudes.

        Parameters
        ----------
        za : numpy.ndarray of floats
            Zenith angle of each pixel in radians.
        diam : float
            Antenna (aperture) diameter in meters.
        freq : float
            Frequency in Hz.

        Returns
        -------
        beam_vals : numpy.ndarray
            Array of Airy disk amplitudes for each zenith angle in `za`.

        """
        xvals = (
                diam / 2. * np.sin(za)
                * 2. * np.pi * freq / c_ms
        )
        beam_vals = np.zeros_like(xvals)
        nz = xvals != 0.
        ze = xvals == 0.
        beam_vals[nz] = 2. * j1(xvals[nz]) / xvals[nz]
        beam_vals[ze] = 1.
        return beam_vals ** 2
    
    def _tanh_taper(self, za, tanh_freq, tanh_sl_red):
        """
        Calculate a tanh tapering function.

        Parameters
        ----------
        za : numpy.ndarray of floats
            Zenith angle of each pixel in radians.
        tanh_freq : float, optional
            Exponential frequency (rate parameter) in inverse radians.
        tanh_sl_red : float, optional
            Airy sidelobe amplitude reduction as a fractional percent. For
            example, passing 0.99 reduces the sidelobes by 0.01, i.e. two
            orders of magnitude.
        
        Returns
        -------
        taper_vals : numpy.ndarray
            Array of tanh taper amplitudes for each zenith angle in `za`.

        """
        taper_vals = 1 - tanh_sl_red * np.tanh(tanh_freq * za)
        return taper_vals

    def _fwhm_to_diam(self, fwhm, freq):
        """
        Calculates the effective diameter of an Airy disk from a FWHM.

        Parameters
        ----------
        fwhm : float
            Full width at half maximum of a Gaussian beam in degrees.
        freq : float
            Frequency in Hz.

        Returns
        -------
        diam : float
            Antenna (aperture) diameter in meters with an Airy disk beam
            pattern whose main lobe is described by a Gaussian beam with a
            FWHM of `fwhm`.

        Notes
        -----
        * Modified from `pyuvsim.analyticbeam.diameter_to_sigma`
          (https://github.com/RadioAstronomySoftwareGroup/pyuvsim).


        """
        scalar = 2.2150894
        wavelength = c_ms / freq
        fwhm = np.deg2rad(fwhm)
        diam = scalar * wavelength / (np.pi * np.sin(fwhm / np.sqrt(2)))
        return diam

    def _diam_to_stddev(self, diam, freq):
        """
        Calculate an effective standard deviation of an Airy disk.

        Parameters
        ----------
        diam : float
            Antenna (aperture) diameter in meters.
        freq : float
            Frequency in Hz.

        Returns
        -------
        sigma : float
            Standard deviation of a Gaussian envelope which describes the main
            lobe of an Airy disk with aperture `diam`.

        Notes
        -----
        * Copied from `pyuvsim.analyticbeam.diameter_to_sigma`
          (https://github.com/RadioAstronomySoftwareGroup/pyuvsim).

        """
        scalar = 2.2150894
        wavelength = c_ms / freq
        sigma = np.arcsin(scalar * wavelength / (np.pi * diam))
        sigma *= np.sqrt(2) / 2.355
        return sigma

    def _check_required_params(self, required_params, all_req=True):
        """
        Check if params in required_params are not None.

        Parameters
        ----------
        required_params : iterable
            Iterable of param values.
        all_req : bool
            If True, require all params are not None. Otherwise, require
            that only one param is not None. Defaults to True.

        """
        if not all_req:
            return np.any([p is not None for p in required_params])
        else:
            return np.all([p is not None for p in required_params])
