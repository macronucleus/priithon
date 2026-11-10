"""
Reusable boundary-effect (ripple) reduction for FFT-based deconvolution.

Based on

    M. Bertero & P. Boccacci,
    "A simple method for the reduction of boundary effects in the
     Richardson-Lucy approach to image deconvolution",
    Astronomy & Astrophysics 437, 369-374 (2005).
    doi:10.1051/0004-6361:20052717

`prepare_boundary()` builds, from an image + PSF, everything needed to run *any*
FFT-based deconvolution on an extended object domain without boundary ripples:

    padded_image (g)  mask (M)  weight (W = K^T (x) M)  valid
    otf (K)           otf_conj (K^T)                    grid geometry

The essential ingredient is the weight `W`; a plain zero-pad is not enough (see
README note below).  Plug these into your own iteration, then `setup.crop()` the
result back to the original image domain.

    Ordinary RL update:   f <- f * K^T(g / K*f)
    Boundary-safe update: f <- f * [K^T( M*g / K*f )] / W        (only on `valid`)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# FFT backend: prefer Priithon (Priithon.fftfuncs, "F"), so the transforms,
# threading and padded sizes match whatever the rest of the pipeline uses
# (pyfftw when available, else numpy/scipy).  Fall back to scipy when Priithon
# is not importable, so the module still runs standalone.
#
# Conventions relied on (verified to match between the two backends):
#   * rfft/irfft are normalised, so irfftn(rfftn(x)) == x;
#   * irfft needs the real-space `shape` to recover an odd last axis (method 1).
# Every array here is already built at the extended grid size, so the forward
# transform never needs an `s=` padding argument.
try:
    try:
        from . import fftfuncs as _F        # when imported as Priithon.boundary
    except (ImportError, ValueError):
        from Priithon import fftfuncs as _F  # fallback (e.g. run as a script)

    def _rfftn(a):
        return _F.rfft(a)

    def _irfftn(a, shape):
        return _F.irfft(a, shape=shape)

    def _next_fast_len(n, real):
        return int(_F.next_fast_len(int(n), real=real))
except Exception:  # pragma: no cover - environment without Priithon
    from scipy.fft import rfftn as _sp_rfftn, irfftn as _sp_irfftn
    from scipy.fft import next_fast_len as _sp_nfl

    def _rfftn(a):
        return _sp_rfftn(a)

    def _irfftn(a, shape):
        return _sp_irfftn(a, s=shape)

    def _next_fast_len(n, real):
        return int(_sp_nfl(int(n), real=real))


def _psf_to_otf(psf: np.ndarray, shape: tuple, work_dtype) -> np.ndarray:
    """Embed a (sum-normalised) PSF into an array of `shape` with its centre at
    the origin, and return its rfftn (the optical transfer function)."""
    otf_input = np.zeros(shape, dtype=work_dtype)
    corner = tuple(slice(0, s) for s in psf.shape)
    otf_input[corner] = psf.astype(work_dtype)
    for axis, s in enumerate(psf.shape):
        otf_input = np.roll(otf_input, -(s // 2), axis=axis)
    return _rfftn(otf_input)


@dataclass
class BoundarySetup:
    """Everything needed to run an FFT-based deconvolution on the extended
    (boundary-safe) object domain.

    Attributes
    ----------
    padded_image : ndarray
        The data `g` embedded in the extended grid (zeros outside the image
        domain S).
    mask : ndarray
        `M`: 1.0 inside the image domain S, 0.0 outside.
    weight : ndarray
        `W = K^T (x) M`, the normalisation that replaces the constant "1" of
        ordinary deconvolution near the boundary.
    valid : ndarray of bool
        `weight > weight_floor * weight.max()`.  True where the object is
        constrained by data; update only these pixels (freeze the rest).
    otf : ndarray (complex)
        `rfftn` of the centred, sum-normalised PSF on the extended grid — the
        forward operator K.  Use with `convolve()`.
    otf_conj : ndarray (complex)
        `conj(otf)` — the adjoint operator K^T (correlation).  Use with
        `correlate()`.
    ext_shape : tuple
        Shape of the extended object grid (FFT grid).
    image_slices : tuple of slice
        Location of the image domain S inside the extended grid; use `crop()`
        (or index with these) to recover the original-size result.
    pads : tuple
        Padding applied per axis, per side.
    weight_floor : float
        The relative floor used to compute `valid`.
    """

    padded_image: np.ndarray
    mask: np.ndarray
    weight: np.ndarray
    valid: np.ndarray
    otf: np.ndarray
    otf_conj: np.ndarray
    ext_shape: tuple
    image_slices: tuple
    pads: tuple
    weight_floor: float

    # -- convenience operators on the extended grid --------------------------
    def convolve(self, f: np.ndarray) -> np.ndarray:
        """Forward blur  K (x) f  (real-space result on the extended grid)."""
        return _irfftn(self.otf * _rfftn(f), self.ext_shape)

    def correlate(self, f: np.ndarray) -> np.ndarray:
        """Adjoint / correlation  K^T (x) f  on the extended grid."""
        return _irfftn(self.otf_conj * _rfftn(f), self.ext_shape)

    def crop(self, f: np.ndarray) -> np.ndarray:
        """Crop an extended-grid array back to the original image domain S."""
        return f[self.image_slices]

    def full(self, value=0.0, dtype=None) -> np.ndarray:
        """A fresh extended-grid array filled with `value` (e.g. an initial
        object estimate)."""
        return np.full(self.ext_shape, value,
                       dtype=dtype or self.padded_image.dtype)


def prepare_boundary(
    image: np.ndarray,
    psf: np.ndarray,
    pad="auto",
    work_dtype=np.float32,
    weight_floor: float = 1e-3,
) -> BoundarySetup:
    """Build the boundary-safe grid, mask, weight and transforms for an image.

    Parameters
    ----------
    image : ndarray
        The blurred / detected image (any ndim: 2D, 3D, ...).
    psf : ndarray
        PSF, same ndim as `image`, centred at its geometric centre
        (shape // 2).  Normalised to sum = 1 internally.
    pad : "auto" | int | sequence of int
        Extension per side, per axis, of the reconstruction domain beyond the
        image domain.  "auto" = psf.shape // 2 (just enough to hold the PSF
        spill from an edge object).  Increase if ripples remain.
    work_dtype : numpy dtype
        Working precision for the FFTs (float32 halves memory).
    weight_floor : float
        Relative floor for `valid`: pixels with weight below
        `weight_floor * weight.max()` are considered unconstrained by the data.

    Returns
    -------
    BoundarySetup
        See the dataclass docstring.  Plug its fields into any FFT-based
        deconvolution; crop the final object with `setup.crop(...)`.
    """
    image = np.asarray(image)
    img = image.astype(work_dtype)
    ndim = img.ndim

    psf = np.asarray(psf, dtype=np.float64)
    if psf.ndim != ndim:
        raise ValueError(f"psf.ndim ({psf.ndim}) must match image.ndim ({ndim})")
    psf = (psf / psf.sum()).astype(work_dtype)

    # pad amount per axis
    if pad == "auto" or pad is None:
        pads = tuple(s // 2 for s in psf.shape)
    elif np.isscalar(pad):
        pads = tuple(int(pad) for _ in range(ndim))
    else:
        pads = tuple(int(p) for p in pad)
        if len(pads) != ndim:
            raise ValueError("pad sequence length must equal image.ndim")

    img_shape = img.shape
    # Only the last axis undergoes the half-spectrum real transform, so ask for a
    # real- (even-)friendly length there and a plain fast length on the others.
    # (Parity of the last axis is otherwise lost on rfft->irfft; see module note.)
    ext_shape = tuple(
        _next_fast_len(img_shape[i] + 2 * pads[i], real=(i == ndim - 1))
        for i in range(ndim)
    )
    image_slices = tuple(
        slice(pads[i], pads[i] + img_shape[i]) for i in range(ndim)
    )

    # embed data and mask
    g = np.zeros(ext_shape, dtype=work_dtype)
    g[image_slices] = img
    mask = np.zeros(ext_shape, dtype=work_dtype)
    mask[image_slices] = 1.0

    # transforms and weight
    otf = _psf_to_otf(psf, ext_shape, work_dtype)
    otf_conj = np.conj(otf)
    weight = _irfftn(otf_conj * _rfftn(mask), ext_shape)
    weight = np.clip(weight, 0.0, None)
    valid = weight > weight_floor * weight.max()

    return BoundarySetup(
        padded_image=g,
        mask=mask,
        weight=weight,
        valid=valid,
        otf=otf,
        otf_conj=otf_conj,
        ext_shape=ext_shape,
        image_slices=image_slices,
        pads=pads,
        weight_floor=float(weight_floor),
    )


# ---------------------------------------------------------------------------
# Edge taper (MATLAB `edgetaper`-style, PSF-aware boundary softening)
# ---------------------------------------------------------------------------

def _cosine_edge_weight(n, w, dtype):
    """1-D weight of length n: 0 at both ends, 1 in the interior, with a
    raised-cosine ramp of `w` pixels at each edge."""
    w = int(max(1, min(w, n // 2)))
    a = np.ones(n, dtype=dtype)
    ramp = 0.5 * (1.0 - np.cos(np.pi * (np.arange(w) + 0.5) / w))  # 0 -> 1
    a[:w] = ramp
    a[n - w:] = ramp[::-1]
    return a


def _fit_psf(psf, shape):
    """Centre-crop `psf` so it is not larger than `shape` along any axis (a PSF
    wider than the image -- common in Z with few planes -- otherwise cannot be
    embedded)."""
    psf = np.asarray(psf)
    slc = []
    for k in range(psf.ndim):
        if psf.shape[k] > shape[k]:
            start = (psf.shape[k] - shape[k]) // 2
            slc.append(slice(start, start + shape[k]))
        else:
            slc.append(slice(None))
    return psf[tuple(slc)]


def _auto_width(psf, k, n):
    """A modest default taper width for axis k: about the PSF half-width along
    that axis, but capped to n//8 so the taper stays a thin border and never
    eats into the interior (crucial in Z, where few planes + a broad axial PSF
    would otherwise blur the whole stack)."""
    if psf is not None and k < psf.ndim:
        w = psf.shape[k] // 2
    else:
        w = n // 16
    return int(max(1, min(w, n // 8)))


def edge_taper(arr, psf=None, otf=None, axes=None, width=None,
               work_dtype=np.float32, eps=1e-8):
    """MATLAB-``edgetaper``-style boundary softening for FFT-based deconvolution.

    Blends `arr` toward its PSF-blurred (circular-convolution) self within a
    THIN band at the edges, so periodic-FFT deconvolution (Wiener, RL, ...) does
    not ring at the borders:

        out = alpha * arr + (1 - alpha) * blur(arr, psf)

    `alpha` is exactly 1 across the interior (so the interior is returned
    unchanged -- NOT blurred) and ramps to 0 with a raised cosine over `width`
    pixels at each edge of the chosen axes.  The PSF is used only to build the
    blurred *target*, not to set the width.

    Unlike a PSF-free apodisation (which only forces opposite borders to match),
    the border blends toward the actual PSF blur.

    Width
    -----
    `width` controls the taper thickness (pixels per axis).  The default is a
    thin border -- about the PSF half-width, capped to axis//8 -- so with a
    broad microscopy PSF the image is NOT blurred everywhere (that was the
    failure of tying the width to the full PSF support).  Pass `width` to tune.

    Z (axis 0)
    ----------
    Include 0 in `axes` to soften the Z boundary planes too.  With few Z planes
    the default Z width is only 1-2 planes (capped), giving a mild roll-off
    instead of a hard zero-padding step.  It softens the array's OWN edges, so
    apply it to the data's Z boundary (taper before/instead of hard zero
    padding); tapering *after* padding zeros only touches the outer zeros.
    With very few Z planes, tapering can only do so much -- a smooth ramp pad or
    the extended-domain (`prepare_boundary`) approach may serve Z better.

    Parameters
    ----------
    arr : ndarray (2D/3D/...)
    psf : ndarray, optional
        PSF (same ndim as arr, centred at its geometric centre, any size; if it
        is larger than arr along an axis it is centre-cropped).  Required unless
        `otf` is given.  Also used for the default `width` unless `width` is set.
    otf : ndarray, optional
        Precomputed rfftn of the centred, sum-normalised PSF on arr's grid
        (e.g. ``BoundarySetup.otf``); reuses a transform instead of recomputing.
    axes : sequence of int, optional
        Axes to taper.  Default: all axes.  ``axes=(-1,-2)`` = XY only,
        ``axes=(0,)`` = Z only.
    width : int or sequence of int, optional
        Taper width in pixels per axis.  Scalar applies to every tapered axis.
        Default: see *Width* above.
    """
    src = arr
    arr = np.asarray(arr, work_dtype)
    ndim = arr.ndim
    if axes is None:
        axes = tuple(range(ndim))
    else:
        axes = tuple(a % ndim for a in axes)

    # blurred (circular-convolution) target
    if otf is None:
        if psf is None:
            raise ValueError("edge_taper needs either psf or otf")
        psf_fit = _fit_psf(np.asarray(psf, np.float64), arr.shape)
        s = psf_fit.sum()
        if s == 0:
            raise ValueError("psf sums to 0")
        psf_fit = (psf_fit / s).astype(work_dtype)   # normalise: blur must preserve the mean
        otf = _psf_to_otf(psf_fit, arr.shape, work_dtype)
    blurred = _irfftn(otf * _rfftn(arr), arr.shape)

    # separable weight: 1 in the interior, thin raised-cosine ramp at the edges
    alpha = np.ones(arr.shape, dtype=work_dtype)
    for k in axes:
        n = arr.shape[k]
        if width is not None:
            w = width[k] if hasattr(width, "__len__") else int(width)
        else:
            w = _auto_width(psf, k, n)

        a1d = _cosine_edge_weight(n, w, work_dtype)
        shape1d = [1] * ndim
        shape1d[k] = n
        alpha = alpha * a1d.reshape(shape1d)

    out = alpha * arr + (1.0 - alpha) * blurred
    out_dtype = getattr(src, "dtype", work_dtype)
    return out.astype(out_dtype, copy=False)


def rl_deconvolve_no_boundary(
    image: np.ndarray,
    psf: np.ndarray,
    num_iter: int = 50,
    pad="auto",
    init: np.ndarray | None = None,
    clip: bool = True,
    work_dtype=np.float32,
    weight_floor: float = 1e-3,
    eps: float = 1e-9,
) -> np.ndarray:
    """Richardson-Lucy deconvolution with reduced boundary effects, built on
    top of `prepare_boundary()` (kept as a ready-to-use reference).

    See `prepare_boundary` for the parameters; `num_iter`, `init`, `clip` and
    `eps` are RL-specific.  Requires NON-NEGATIVE data.
    """
    s = prepare_boundary(image, psf, pad=pad, work_dtype=work_dtype,
                         weight_floor=weight_floor)
    g, mask, weight, valid = s.padded_image, s.mask, s.weight, s.valid

    if init is None:
        f = s.full(float(s.crop(g).mean()))
    else:
        init = np.asarray(init, dtype=work_dtype)
        if init.shape != image.shape:
            raise ValueError("init must have the same shape as image")
        f = s.full(float(init.mean()))
        f[s.image_slices] = init

    for _ in range(int(num_iter)):
        blur = s.convolve(f)
        np.clip(blur, eps, None, out=blur)
        ratio = (mask * g) / blur                    # 0 outside S
        corr = s.correlate(ratio)
        f_new = f.copy()
        f_new[valid] = f[valid] * corr[valid] / weight[valid]
        if clip:
            np.clip(f_new, 0.0, None, out=f_new)
        f = f_new

    return s.crop(f).astype(image.dtype, copy=False)


if __name__ == "__main__":
    # Demo: the standalone setup driving a hand-written RL loop, showing the
    # boundary machinery is fully decoupled from the algorithm.
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(1)
    true = np.zeros((16, 96, 96), np.float32)
    true[8, 0:5, 44:52] = 500.0                      # object touching top edge

    zz, yy, xx = np.mgrid[-7:8, -7:8, -7:8]
    sigma = (1.2, 2.5, 2.5)
    psf = np.exp(-(zz**2 / (2*sigma[0]**2)
                   + yy**2 / (2*sigma[1]**2)
                   + xx**2 / (2*sigma[2]**2))).astype(np.float32)
    psf /= psf.sum()
    blurred = rng.poisson(gaussian_filter(true, sigma) + 3.0).astype(np.float32)

    # --- generic usage: prepare once, then run any FFT-based method ---
    s = prepare_boundary(blurred, psf)
    print("ext grid:", s.ext_shape, " valid pixels:", int(s.valid.sum()),
          "/", s.valid.size)
    print("weight  min/max: %.4f / %.4f" % (s.weight.min(), s.weight.max()))

    f = s.full(float(blurred.mean()))
    for _ in range(80):                              # a bare RL loop, by hand
        blur = np.clip(s.convolve(f), 1e-9, None)
        corr = s.correlate((s.mask * s.padded_image) / blur)
        f[s.valid] = np.clip(f[s.valid] * corr[s.valid] / s.weight[s.valid], 0, None)
    rec = s.crop(f)
    print("edge-object peak recovered: %.1f (true 500), nan: %s"
          % (rec[8, 0:5, 44:52].max(), bool(np.isnan(rec).any())))
