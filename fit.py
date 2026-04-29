import numpy as np


def simple_local_maxima(x, y, vicinity=3, include_edges=False):
    """
    Minimal local-max finder.

    include_edges=False avoids forcing bad edge pixels into the continuum.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    idx = []

    for i in range(vicinity, len(y) - vicinity):
        window = y[i - vicinity:i + vicinity + 1]
        if y[i] == np.max(window):
            idx.append(i)

    idx = np.array(idx, dtype=int)

    if include_edges:
        idx = np.unique(np.concatenate([[0], idx, [len(y) - 1]]))

    return x[idx], y[idx], idx

def rolling_pin_anchors(wave, flux, R):
    """
    Rolling-pin algorithm.

    Parameters
    ----------
    wave : array
        Wavelengths of local maxima.
    flux : array
        Flux values of local maxima.
    R : float or array
        Rolling-pin radius. If array, must have same length as wave.

    Returns
    -------
    keep_wave, keep_flux, keep_indices
        Anchor points selected by the rolling pin.
    """

    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux, dtype=float)

    order = np.argsort(wave)
    wave = wave[order]
    flux = flux[order]

    if np.isscalar(R):
        radius = np.full(len(wave), float(R))
    else:
        radius = np.asarray(R, dtype=float)[order]

    waves = wave - wave[:, None]
    distance = np.sign(waves) * np.sqrt(
        waves**2 + (flux - flux[:, None])**2
    )
    distance[distance < 0] = 0

    numero = np.arange(len(wave), dtype=int)

    keep = [0]
    j = 0

    while len(wave) - j > 3:
        par_R = float(radius[j])

        mask = (distance[j, :] > 0) & (distance[j, :] < 2.0 * par_R)

        # If no point is reachable, enlarge radius until one is.
        while np.sum(mask) == 0:
            par_R *= 1.5
            mask = (distance[j, :] > 0) & (distance[j, :] < 2.0 * par_R)

        p1 = np.array([wave[j], flux[j]])
        p2 = np.column_stack([wave[mask], flux[mask]])

        delta = p2 - p1
        c = np.sqrt(delta[:, 0]**2 + delta[:, 1]**2)

        h = np.sqrt(par_R**2 - 0.25 * c**2)

        cx = p1[0] + 0.5 * delta[:, 0] - h / c * delta[:, 1]
        cy = p1[1] + 0.5 * delta[:, 1] + h / c * delta[:, 0]

        cond1 = (cy - p1[1]) >= 0

        theta = (
            cond1 * (-np.arccos((cx - p1[0]) / par_R) + np.pi)
            + (1 - cond1) * (-np.arcsin((cy - p1[1]) / par_R) + np.pi)
        )

        j2 = np.argmin(theta)
        j = numero[mask][j2]

        keep.append(j)

    keep = np.array(keep, dtype=int)

    return wave[keep], flux[keep], keep

def rolling_pin_continuum(grid, spectrum, R, vicinity=3, y_scale=None):
    """
    Minimal rolling-pin continuum model.
    """

    grid = np.asarray(grid, dtype=float)
    spectrum = np.asarray(spectrum, dtype=float)

    order = np.argsort(grid)
    grid_sorted = grid[order]
    spectrum_sorted = spectrum[order]

    if y_scale is None:
        y_scale = np.ptp(spectrum_sorted) / np.ptp(grid_sorted)

    scaled_flux = spectrum_sorted / y_scale

    max_wave, max_flux, _ = simple_local_maxima(
        grid_sorted,
        scaled_flux,
        vicinity=vicinity,
        include_edges=False
    )

    anchor_wave, anchor_flux_scaled, _ = rolling_pin_anchors(
        max_wave,
        max_flux,
        R=R
    )

    continuum_scaled = np.interp(
        grid_sorted,
        anchor_wave,
        anchor_flux_scaled
    )

    # Prevent artificial edge ramps.
    continuum_scaled[grid_sorted < anchor_wave[0]] = anchor_flux_scaled[0]
    continuum_scaled[grid_sorted > anchor_wave[-1]] = anchor_flux_scaled[-1]

    continuum_sorted = continuum_scaled * y_scale
    anchor_flux = anchor_flux_scaled * y_scale

    continuum = np.empty_like(continuum_sorted)
    continuum[order] = continuum_sorted

    return continuum, anchor_wave, anchor_flux