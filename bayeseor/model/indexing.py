import numpy as np
from astropy_healpix import healpy as hp
from scipy import sparse

from bayeseor.model.healpix import Healpix

class ArrayIndexing():
    """
    Class for convenient vector slicing in various data spaces.

    Parameters
    ----------
    nu : int
        Number of pixels on a side for the u-axis in the EoR model uv-plane.
    nv : int, optional
        Number of pixels on a side for the v-axis in the EoR model uv-plane.
        Defaults to `nu`.
    nu_fg : int, optional
        Number of pixels on a side for the u-axis in the foreground model
        uv-plane. Defaults to `nu`.
    nv_fg : int, optional
        Number of pixels on a side for the v-axis in the foreground model
        uv-plane. Defaults to `nv` if `nu_fg` is None or `nu_fg`.
    fit_for_monopole : bool, optional
        Fit for the (u, v) = (0, 0) pixel in the model uv-plane.
    neta : int, optional
        Number of Line of Sight (LoS, frequency axis) Fourier modes. Defaults
        to `nf`.
    nq : int, optional
        Number of large spectral scale model basis vectors. Defaults to 0.
    nf : int
        Number of frequency channels.
    nside : int
        HEALPix nside parameter.
    fov_ra_eor : float
        Field of view of the Right Ascension axis of the EoR sky model in
        degrees.
    fov_dec_eor : float, optional
        Field of view of the Declination axis of the EoR sky model in degrees.
        Defaults to `fov_ra_eor`.
    fov_ra_fg : float, optional
        Field of view of the Right Ascension axis of the foreground sky model
        in degrees. Defaults to `fov_ra_eor`.
    fov_dec_fg : float, optional
        Field of view of the Declination axis of the foreground sky model in
        degrees. Defaults to `fov_dec_eor` if `fov_ra_fg` is None or
        `fov_ra_fg`.
    simple_za_filter : bool, optional
        Filter pixels in the sky model by zenith angle only. Defaults to True.
    single_fov : bool, optional
        If True, use a single field of view at the central time step to form
        the sky model pixel mask(s). Otherwise, calculate the pixel masks for
        each time and form the total pixel masks as the union of the pixel
        indices at each time. See :class:`bayeseor.model.healpix.Healpix` for
        more details. Defaults to False.
    nt : int
        Number of times.
    jd_center : float
        Central time as a Julian date.
    telescope_latlonalt : sequence of float
        Telescope location tuple as (latitude in degrees, longitude in degrees,
        altitude in meters).
    nbls : int
        Number of baselines.

    """
    def __init__(
        self,
        *,
        nu : int,
        nv : int | None = None,
        nu_fg : int | None = None,
        nv_fg : int | None = None,
        fit_for_monopole : bool = False,
        neta : int | None = None,
        nq : int = 0,
        nf : int,
        nside : int,
        fov_ra_eor : float,
        fov_dec_eor : float | None = None,
        fov_ra_fg : float | None = None,
        fov_dec_fg : float | None = None,
        simple_za_filter : bool = True,
        single_fov : bool = False,
        nt : int,
        jd_center : float,
        telescope_latlonalt : list[float] | tuple[float],
        nbls : int
    ):
        if nv is None:
            nv = nu
        if nu_fg is None:
            nu_fg = nu
            nv_fg = nv
        elif nv_fg is None:
            nv_fg = nu_fg
        if neta is None:
            neta = nf
        if fov_dec_eor is None:
            fov_dec_eor = fov_ra_eor
        if fov_ra_fg is None:
            fov_ra_fg = fov_ra_eor
            fov_dec_fg = fov_dec_eor
        elif fov_dec_fg is None:
            fov_dec_fg = fov_ra_fg

        hpx = Healpix(
            fov_ra_eor=fov_ra_eor,
            fov_dec_eor=fov_dec_eor,
            fov_ra_fg=fov_ra_fg,
            fov_dec_fg=fov_dec_fg,
            simple_za_filter=simple_za_filter,
            single_fov=single_fov,
            nside=nside,
            telescope_latlonalt=telescope_latlonalt,
            jd_center=jd_center
        )

        self.nu = nu
        self.nv = nv
        self.nu_fg = nu_fg
        self.nv_fg = nv_fg
        self.fit_for_monopole = fit_for_monopole
        self.neta = neta
        self.nq = nq
        self.nf = nf
        self.nside = nside
        self.fov_ra_eor = fov_ra_eor
        self.fov_dec_eor = fov_dec_eor
        self.fov_ra_fg = fov_ra_fg
        self.fov_dec_fg = fov_dec_fg
        self.simple_za_filter = simple_za_filter
        self.single_fov = single_fov
        self.hpx = hpx
        self.npix_eor = hpx.npix_fov_eor
        self.npix_fg = hpx.npix_fov_fg
        self.nt = nt
        self.jd_center = jd_center
        self.telescope_latlonalt = telescope_latlonalt
        self.nbls = nbls

        # Measurement space
        # Number of visibilities in the model visibility vector
        self.nvis = nbls * nf * nt

        # Image space
        # Number of EoR pixel amplitudes in the model image vector
        self.nimg_eor = self.npix_eor * nf
        # Number of foreground pixel amplitudes in the model image vector
        self.nimg_fg = self.npix_fg * nf
        # Number of joint EoR + foreground pixel amplitudes in the model image
        # vector. This is always self.nimg_fg as the EoR and foreground image
        # models share the pixels used in the EoR image model and the
        # foreground model field of view must always be greater than or equal
        # to the EoR model field of view.
        self.nimg = self.nimg_fg

        # uvf space
        # Number of (u, v, f) voxels in the EoR model uvf vector. The monopole,
        # (u, v) = (0, 0), is lumped in to the foreground model, so we only
        # have nu*nv - 1 (u, v) per frequency in the EoR model.
        self.nuv_eor = (nu*nv - 1)
        self.nuvf_eor = self.nuv_eor * nf
        # Number of (u, v, f) voxels in the foreground model uvf vector
        self.nuv_fg = (nu_fg*nv_fg - (not fit_for_monopole))
        self.nuvf_fg = self.nuv_fg * nf
        # Number of (u, v, f) voxels in the joint
        # EoR + foreground model uvf vector
        self.nuvf = self.nuvf_eor + self.nuvf_fg

        # uveta space
        # Number of (u, v, eta) voxels in the EoR model uveta vector
        self.nuveta_eor = (nu*nv - 1) * (neta - 1)
        # Number of (u, v, eta) voxels in the foreground model uveta vector
        self.nuveta_fg = (
            (nu_fg*nv_fg - 1) * (nq + 1) + fit_for_monopole * (neta + nq)
        )
        # Number of (u, v, eta) in the joint
        # EoR + foregrounds model uveta vector
        self.nuveta = self.nuveta_eor + self.nuveta_fg

    def get_vis_waterfall(self, i_bl : int, vis_vec : np.ndarray):
        """
        Get a visibility waterfall for a single baseline.

        Parameters
        ----------
        i_bl : int
            Baseline index. Must be less than `self.nbls`.
        vis_vec : numpy.ndarray
            Visibility vector with shape `(self.nvis,)`.

        Returns
        -------
        vis : numpy.ndarray
            Visibility waterfall with shape `(self.nt, self.nf)`.
        """
        if i_bl >= self.nbls:
            raise ValueError(
                f"i_bl ({i_bl}) must be less than self.nbls ({self.nbls})"
            )
        vis = vis_vec[i_bl :: self.nf*self.nt].reshape(self.nt, self.nf)
        return vis
    
    # TODO: add a function which takes the model visibility vector and
    # returns a dictionary(?) with keys of antenna pair tuples, if available,
    # or baseline indices and values of visibility waterfalls for each baseline

    def get_img_arr(self, img_vec : np.ndarray, masked : bool = False):
        """
        Get a joint, masked EoR + foreground model image array.

        Parameters
        ----------
        img_vec : numpy.ndarray
            Model image-space vector with shape `(self.nimg,)`.
        masked : bool
            Return a masked array using the full pixel count for `self.nside`.

        Returns
        -------
        img_arr : numpy.ndarray or numpy.ma.masked_array
            If `masked` is False, model image-space pixel amplitudes with shape
            `(nf, self.npix_fg)`. Otherwise, masked model image-space pixel
            amplitudes with shape `(nf, 12*self.nside**2)`. Masked pixels are
            filled with `np.nan`. Returned if `masked` is True.
        """
        if not masked:
            img_arr = img_vec.reshape(self.nf, self.npix_fg)
        else:
            pix_vals = np.zeros(
                (self.nf, hp.nside2npix(self.nside)), dtype=img_vec.dtype
            )
            pix_vals[:, self.hpx.pix] = img_vec.reshape(self.nf, self.npix_fg)
            mask = np.ones_like(pix_vals, dtype=bool)
            mask[:, self.hpx.pix] = False
            img_arr = np.ma.masked_array(data=pix_vals, mask=mask, fill_value=np.nan)
        return img_arr

    def get_uvf_arrays(self, uvf_vec: np.ndarray):
        """
        Get separate (u, v, freq) arrays for the EoR and foreground models.

        Parameters
        ----------
        uvf_vec : np.ndarray
            Combined EoR and foreground model (u, v, freq) vector with shape
            `(self.nuvf,)`.

        Returns
        -------
        uvf_eor : np.ma.masked_array
            EoR model (u, v, freq) coefficients as a masked array with shape
            `(self.nf, self.nv, self.nu)`.
        uvf_fg : np.ma.masked_array
            Foreground model (u, v, freq) coefficients as a masked array with
            shape `(self.nf, self.nv_fg, self.nu_fg)`. If
            `self.fit_for_monopole` is False, the central (u, v) = (0, 0) mode
            is masked.

        Notes
        -----
        * The combined EoR and foreground model (u, v, freq) vector can be
          easily separated as the EoR model occupies the first `self.nuvf_eor`
          entries and the foreground model occupies the last `self.nuvf_fg`
          entries.
        * Both EoR and foreground model (u, v, freq) vector subsets are
          chan-ordered vectors.  For the EoR model vector subset, the first
          `self.nuv_eor` entries correspond to the flattened
          model uv-plane at the lowest frequency, the next `self.nuv_eor`
          entries correspond to the second lowest freuqency, etc. and are
          ordered in order of increasing frequency.  The flattening
          of the model uv-plane follows "C ordering", i.e. flatten along u
          (rows) then v (columns) as (-u_max, -v_max), (-u_max + du, -v_max),
          ..., (-u_max, -v_max + dv), (-u_max + du, -v_max + dv), ...,
          (u_max - du, -v_max + dv), ..., (u_max - du, v_max), (u_max, v_max).
          The same is true for the foreground model vector subset except there
          are `self.nuv_fg` entries for each frequency channel.
        """
        if not uvf_vec.shape == (self.nuvf,):
            raise ValueError(
                f"uvf_vec must have shape {(self.nuvf,)} but has shape "
                f"{uvf_vec.shape}"
            )
        
        uvf_eor_vec = uvf_vec[:self.nuvf_eor].reshape(self.nf, -1)
        # Insert NaN into the EoR model coefficients at the location of
        # (u, v) = (0, 0) along the flattened (u, v) axis
        uvf_eor_vec = np.insert(uvf_eor_vec, (self.nu*self.nv)//2, np.nan, axis=1)
        uvf_eor = uvf_eor_vec.reshape(self.nf, self.nv, self.nu)
        # Create mask for EoR model uv-plane with (u, v) = (0, 0) masked
        uv_eor_mask = np.zeros((self.nv, self.nu), dtype=bool)
        uv_eor_mask[self.nv//2, self.nu//2] = True
        uvf_eor_mask = np.repeat(uv_eor_mask[None, :, :], self.nf, axis=0)
        uvf_eor = np.ma.masked_array(
            uvf_eor, mask=uvf_eor_mask, fill_value=np.nan
        )

        uvf_fg_vec = uvf_vec[self.nuvf_eor:].reshape(self.nf, -1)
        if not self.fit_for_monopole:
            # Insert NaN into the foreground model coefficients at the location
            # of (u, v) = (0, 0) along the flattened (u, v) axis
            uvf_fg_vec = np.insert(
                uvf_fg_vec, (self.nu_fg*self.nv_fg)//2, np.nan, axis=1
            )
        uvf_fg = uvf_fg_vec.reshape(self.nf, self.nv_fg, self.nu_fg)
        uvf_fg_mask = np.zeros_like(uvf_fg, dtype=bool)
        if not self.fit_for_monopole:
            uvf_fg_mask[:, self.nv_fg//2, self.nu_fg//2] = True
        uvf_fg = np.ma.masked_array(
            uvf_fg, mask=uvf_fg_mask, fill_value=np.nan
        )

        return uvf_eor, uvf_fg

    def get_uveta_arrays(self, uveta_vec: np.ndarray):
        """
        Get separate (u, v, eta) arrays for the EoR and foreground models.

        Parameters
        ----------
        uveta_vec : numpy.ndarray
            Combined EoR and foreground model (u, v, eta) vector with shape
            `(self.nuveta,)`.

        Returns
        -------
        uveta_eor : numpy.ma.masked_array
            EoR model (u, v, eta) coefficients as a masked array with shape
            `(self.neta, self.nv, self.nu)`.
        uveta0_fg : numpy.ma.masked_array
            Foreground model (u, v) coefficients as a masked array with shape
            `(self.nv_fg, self.nu_fg)`. If `self.fit_for_monopole` is False,
            the central (u, v) = (0, 0) mode is masked.
        u0v0eta_fg : numpy.ndarray
            Foreground model (u, v) = (0, 0) coefficients for all eta with
            shape `(self.neta,)`.

        Notes
        -----
        * The combined EoR and foreground model (u, v, eta) vector can be easily
          separated as the EoR model occupies the first `self.nuveta_eor`
          entries while the foreground model occupies the last `self.nuveta_fg`
          entries.
        * The EoR model (u, v, eta) vector subset is a vis ordered
          vector where the first `self.neta - 1` entries correspond to the eta
          spectrum of the most negative (u, v) = (-u_max, -v_max) in the EoR
          model uv-plane, the second `self.neta - 1` entries correspond to the
          eta spectrum of (u, v) = (-u_max + du, -v_max), etc.  The flattening
          of the model uv-plane follows "C ordering", i.e. flatten along u (rows)
          then v (columns).
        * The foreground model (u, v, eta) vector contains only the eta=0
          uv-plane and the (u, v) = (0, 0) coefficients.  The eta=0 uv-plane
          coefficients come first in the vector and include all (u, v) except
          (u, v) = (0, 0), which is only included if `fit_for_monopole` is
          True.  These eta=0 uv-plane coefficients are ordered identically to
          the EoR model uv-plane coefficients, i.e. they follow "C ordering".
          The corresponding (u, v) locations in the eta=0 uv-plane are thus
          (-u_max, -v_max), (-u_max + du, -v_max), ..., (-u_max, -v_max + dv),
          (-u_max + du, -v_max + dv), ..., (u_max - du, -v_max + dv), ...,
          (u_max - du, v_max), (u_max, v_max).  The remaining (u, v) = (0, 0)
          coefficients are only included in the model if `fit_for_monopole`
          is True.  These are ordered according to -eta_max, -eta_max + deta,
          ...,-deta, 0, deta, ..., eta_max - deta, eta_max.
        """
        if not uveta_vec.shape == (self.nuveta,):
            raise ValueError(
                f"uveta_vec must have shape {(self.nuveta,)} but has shape "
                f"{uveta_vec.shape}"
            )
        uveta_vec_eor = uveta_vec[:self.nuveta_eor]
        uveta_vec_fg = uveta_vec[self.nuveta_eor:]

        # Create a masked array for the EoR model (u, v, eta) coefficients
        # where eta=0 and (u, v) = (0, 0) are masked values which belong
        # to the foreground model.
        uveta_eor = uveta_vec_eor.reshape(self.neta-1, -1, order="F")
        uveta_eor_mask = np.zeros_like(uveta_eor, dtype=bool)
        # Insert NaN into the coefficients and True into the mask at the
        # location of eta=0 along the eta axis
        uveta_eor = np.insert(uveta_eor, self.neta//2, np.nan, axis=0)
        uveta_eor_mask = np.insert(uveta_eor_mask, self.neta//2, True, axis=0)
        # Insert Nan into the coefficients and True into the mask at the
        # location of (u, v) = (0, 0) along the flattened (u, v) axis
        uveta_eor = np.insert(uveta_eor, (self.nu*self.nv)//2, np.nan, axis=1)
        uveta_eor_mask = np.insert(
            uveta_eor_mask, (self.nu*self.nv)//2, True, axis=1
        )
        # Reshape the coefficient and mask arrays to (neta, nv, nu)
        uveta_eor = uveta_eor.reshape(self.neta, self.nv, self.nu, order="C")
        uveta_eor_mask = uveta_eor_mask.reshape(
            self.neta, self.nv, self.nu, order="C"
        )
        # Create a masked array from coefficient and mask arrays
        uveta_eor = np.ma.masked_array(data=uveta_eor, mask=uveta_eor_mask)
        
        # Create a sparse array for the foreground model (u, v, eta)
        # coefficients for eta=0 (excluding (u, v) = (0, 0)).  There are 1 + nq
        # Large Spectral Scale Model (LSSM) coefficients per (u, v) in the eta=0
        # slice.  We always include the constant term (i.e. eta=0) in the LSSM,
        # even when nq=0.
        uveta0_fg = np.zeros((1 + self.nq, self.nv_fg, self.nu_fg), dtype=complex)
        nuv_fg_eta0 = self.nu_fg * self.nv_fg - 1
        # Mask the (u, v) = (0, 0) mode for easy slicing
        uvmp_mask = np.ones((self.nv_fg, self.nu_fg), dtype=bool)
        uvmp_mask[self.nv_fg//2, self.nu_fg//2] = False
        # The first nuv_fg_eta0 entries in uveta_vec_fg are the eta=0
        # coefficients for (u, v) != (0, 0) where the uv plane is flattened
        # in "C ordering", i.e. along u (rows) then v (columns) starting with
        # the most negative (u, v) = (-u_max, -v_max).
        uveta0_fg[0, uvmp_mask] = uveta_vec_fg[:nuv_fg_eta0]
        uveta0_fg_mask = np.logical_not(uvmp_mask)  # FIXME: wrong shape, need 0th nq axis!!!
        if self.nq > 0:
            # The LSSM coefficients for a given (u, v) are adjacent in
            # uveta_vec_fg and are located after the (u, v) != (0, 0)
            # coefficients totalling nq*(nu_fg*nv_fg - 1) entries.
            lssm_coeffs = uveta_vec_fg[nuv_fg_eta0 : nuv_fg_eta0*(1 + self.nq)]
            lssm_coeffs = lssm_coeffs.reshape(nuv_fg_eta0, self.nq).T
            uveta0_fg[1:, uvmp_mask] = lssm_coeffs
        if self.fit_for_monopole:
            # The (u, v) = (0, 0) mode coefficients are located after the LSSM
            # coefficients in uveta_vec_fg totalling neta entries.
            u0v0_start_ind = nuv_fg_eta0*(1 + self.nq)
            # The first neta entries correspond to the (u, v) = (0, 0)
            # coefficients for all eta, with eta=0 in the middle of this slice.
            u0v0_fg = uveta_vec_fg[u0v0_start_ind : u0v0_start_ind + self.neta]
            # The last nq entries correspond to the LSSM of (u, v) = (0, 0)
            u0v0_lssm_coeffs = uveta_vec_fg[-self.nq:].reshape(self.nq, 1)
            # Fill the (eta, u, v) = (0, 0, 0) mode
            uveta0_fg[0, uveta0_fg_mask] = u0v0_fg[self.neta//2]
            # Add the LSSM coefficients for (eta, u, v) = (0, 0, 0)
            uveta0_fg[1:, uveta0_fg_mask] = u0v0_lssm_coeffs
            # (u, v) = (0, 0) included in model, no need to mask it
            uveta0_fg_mask[self.nv_fg//2, self.nu_fg//2] = False
        else:
            u0v0_fg = None
        # Mimic the shape of uveta0_fg with (nq, nu_fg, nv_fg)
        uveta0_fg_mask = np.repeat(
            uveta0_fg_mask[None, ...], 1 + self.nq, axis=0
        )
        uveta0_fg = np.ma.masked_array(
            data=uveta0_fg, mask=uveta0_fg_mask, fill_value=np.nan
        )

        return uveta_eor, uveta0_fg, u0v0_fg