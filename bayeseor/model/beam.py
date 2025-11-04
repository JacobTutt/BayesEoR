"""
Beam model class and functions.
"""

import numpy as np
from astropy.units import Quantity
from astropy.constants import c
from pyuvdata import UVBeam
from pyuvdata import utils as uvutils
from scipy.special import j1

class Beam():
    """
    Class and functions for the beam model.

    Parameters
    ----------
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
                stddev = self.fwhm_to_stddev(self.fwhm_deg)
            else:
                stddev = self.diam_to_stddev(self.diam, freq)
            if self.beam_type == "gaussian":
                beam_vals = self.gaussian_za(za, stddev, self.peak_amp)
            else:
                beam_vals = self.gausscosine(
                    za, stddev, self.peak_amp, self.cosfreq
                )

        elif self.beam_type == "airy":
            if self.diam is not None:
                beam_vals = self.airy_disk(za, self.diam, freq)
            else:
                diam_eff = self.fwhm_to_diam(self.fwhm_deg, freq)
                beam_vals = self.airy_disk(za, diam_eff, freq)
        
        elif self.beam_type == "taperairy":
            stddev = self.fwhm_to_stddev(self.fwhm_deg)
            beam_vals = (
                self.airy_disk(za, self.diam, freq)
                * self.gaussian_za(za, stddev, self.peak_amp)
            )
        
        elif self.beam_type == "tanhairy":
            beam_vals = (
                self.airy_disk(za, self.diam, freq)
                * self.tanh_taper(za, self.tanh_freq, self.tanh_sl_red)
            )
        
        elif self.beam_type == "uvbeam":
            if isinstance(az, Quantity):
                az = az.to("rad").value
            if isinstance(za, Quantity):
                za = za.to("rad").value
            if isinstance(freq, Quantity):
                freq = freq.to("Hz").value
            beam_vals, _ = self.uvb.interp(
                az_array=az, za_array=za, freq_array=np.array([freq]),
                reuse_spline=True
            )
            beam_vals = beam_vals[0, 0, 0, 0].real

        return beam_vals

    def gaussian_za(self, za, sigma, amp):
        """
        Calculate azimuthally symmetric Gaussian beam amplitudes.

        Parameters
        ----------
        za : astropy.units.Quantity or numpy.ndarray
            Zenith angle of each pixel in radians if not a Quantity.
        sigma : astropy.units.Quantity or float
            Standard deviation in radians if not a Quantity.
        amp : float
            Peak amplitude at zenith.

        Returns
        -------
        beam_vals : numpy.ndarray
            Array of Gaussian beam amplitudes for each zenith angle in `za`.

        """
        if isinstance(za, Quantity):
            za = za.to("rad").value
        if isinstance(sigma, Quantity):
            sigma = sigma.to("rad").value
        beam_vals = amp * np.exp(-za**2 / (2 * sigma**2))
        return beam_vals
    
    def gausscosine(self, za, sigma, amp, cosfreq):
        """
        Calculate azimuthally symmetric Gaussian * cosine^2 beam amplitudes.

        Parameters
        ----------
        za : astropy.units.Quantity or numpy.ndarray
            Zenith angle of each pixel in radians if not a Quantity.
        sigma : astropy.units.Quantity or float
            Standard deviation in radians if not a Quantity.
        amp : float
            Peak amplitude at zenith.
        cosfreq : astropy.units.Quantity or float
            Cosine squared frequency in inverse radians if not a Quantity.

        Returns
        -------
        beam_vals : numpy.ndarray
            Array of Gaussian beam amplitudes for each zenith angle in `za`.

        """
        if isinstance(za, Quantity):
            za = za.to("rad").value
        if isinstance(sigma, Quantity):
            sigma = sigma.to("rad").value
        if isinstance(cosfreq, Quantity):
            cosfreq = cosfreq.to("1/rad").value
        beam_vals = self.gaussian_za(za, sigma, amp)
        beam_vals *= np.cos(2 * np.pi * za * cosfreq/2)**2
        return beam_vals

    def fwhm_to_stddev(self, fwhm):
        """
        Calculate standard deviation from full width at half maximum.

        Parameters
        ----------
        fwhm : astropy.units.Quantity or float
            Full width half maximum.

        """
        return fwhm / 2.355

    def airy_disk(self, za, diam, freq):
        """
        Calculate Airy disk amplitudes.

        Parameters
        ----------
        za : astropy.units.Quantity or numpy.ndarray
            Zenith angle of each pixel in radians if not a Quantity.
        diam : astropy.units.Quantity or float
            Antenna (aperture) diameter in meters if not a Quantity.
        freq : astropy.units.Quantity or float
            Frequency in Hz if not a Quantity.

        Returns
        -------
        beam_vals : numpy.ndarray
            Array of Airy disk amplitudes for each zenith angle in `za`.

        """
        if isinstance(za, Quantity):
            za = za.to("rad").value
        if isinstance(diam, Quantity):
            diam = diam.to("m").value
        if isinstance(freq, Quantity):
            freq = freq.to("Hz").value
        xvals = (
                diam / 2. * np.sin(za)
                * 2. * np.pi * freq / c.to("m/s").value
        )
        beam_vals = np.zeros_like(xvals)
        nz = xvals != 0.
        ze = xvals == 0.
        beam_vals[nz] = 2. * j1(xvals[nz]) / xvals[nz]
        beam_vals[ze] = 1.
        return beam_vals ** 2
    
    def tanh_taper(self, za, tanh_freq, tanh_sl_red):
        """
        Calculate a tanh tapering function.

        Parameters
        ----------
        za : astropy.units.Quantity or numpy.ndarray
            Zenith angle of each pixel in radians if not a Quantity.
        tanh_freq : astropy.units.Quantity or float
            Exponential frequency (rate parameter) in inverse radians if not a
            Quantity.
        tanh_sl_red : float
            Airy sidelobe amplitude reduction as a fractional percent. For
            example, passing 0.99 reduces the sidelobes by 0.01, i.e. two
            orders of magnitude.
        
        Returns
        -------
        taper_vals : numpy.ndarray
            Array of tanh taper amplitudes for each zenith angle in `za`.

        """
        if isinstance(za, Quantity):
            za = za.to("rad").value
        if isinstance(tanh_freq, Quantity):
            tanh_freq = tanh_freq.to("1/rad").value
        taper_vals = 1 - tanh_sl_red * np.tanh(tanh_freq * za)
        return taper_vals

    def fwhm_to_diam(self, fwhm, freq):
        """
        Calculates the effective diameter of an Airy disk from a FWHM.

        Parameters
        ----------
        fwhm : astropy.units.Quantity or float
            Full width at half maximum of a Gaussian beam in degrees if not a
            Quantity.
        freq : astropy.units.Quantity or float
            Frequency in Hz if not a Quantity.

        Returns
        -------
        diam : astropy.units.Quantity
            Antenna (aperture) diameter in meters with an Airy disk beam
            pattern whose main lobe is described by a Gaussian beam with a
            FWHM of `fwhm`.

        Notes
        -----
        * Modified from `pyuvsim.analyticbeam.diameter_to_sigma`
          (https://github.com/RadioAstronomySoftwareGroup/pyuvsim).


        """
        scalar = 2.2150894
        wavelength = c.to("m/s") / freq.to("1/s")
        diam = (
            scalar * wavelength
            / (np.pi * np.sin(fwhm.to("rad").value / np.sqrt(2)))
        )
        return diam

    def diam_to_stddev(self, diam, freq):
        """
        Calculate an effective standard deviation of an Airy disk.

        Parameters
        ----------
        diam : astropy.units.Quantity or float
            Antenna (aperture) diameter in meters.
        freq : astropy.units.Quantity or float
            Frequency in Hz.

        Returns
        -------
        sigma : astropy.units.Quantity
            Standard deviation in radians of a Gaussian envelope which
            describes the main lobe of an Airy disk with aperture `diam`.

        Notes
        -----
        * Copied from `pyuvsim.analyticbeam.diameter_to_sigma`
          (https://github.com/RadioAstronomySoftwareGroup/pyuvsim).

        """
        if not isinstance(diam, Quantity):
            diam = Quantity(diam, unit="m")
        if not isinstance(freq, Quantity):
            freq = Quantity(freq, unit="Hz")
        scalar = 2.2150894
        wavelength = c.to("m/s") / freq.to("1/s")
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