# -*- coding: utf-8 -*-

import os
import re
import io
import contextlib
import numpy as np
import matplotlib.pyplot as plt
from .utils import wavepal_analyze

DEFAULT_PERCENTILES = (95., 99., 99.9)

DEFAULT_TIME_STRING = [0., 100., 200., 300., 400., 500., 600., 700., 800., 900., 1000.]
DEFAULT_PERIOD_STRING = [5., 10., 12., 15., 20., 25., 30., 35., 40., 45., 50.,
                         55., 60., 65., 70., 75., 80., 85., 90., 95., 100.]

# Colors for the three period categories that can be marked on a scalogram.
# Red is left free on purpose: wavepal itself uses it (via its own
# colormap/annotations) to flag the maximum signal in the scalogram, so we
# avoid clashing with that.
DEFAULT_ACTIVITY_COLOR = "peru"  # rotation-related periods (Prot, Prot/2, ...)
DEFAULT_PLANET_COLOR = "darkmagenta"      # known/candidate planet periods
DEFAULT_OTHER_COLOR = "slategray"    # unidentified/extra periods (soft yellow)


def _find_rv_axis(fig):
    """
    Finds the axis holding the RV time series (the top panel, labeled
    "RV (m/s)" by wavepal).

    Arguments:
    fig : matplotlib.figure.Figure
        The scalogram figure.
    """
    for ax in fig.axes:
        ylabel = (ax.get_ylabel() or "").lower()
        if "rv" in ylabel and "m/s" in ylabel:
            return ax
    return None


def _default_label(kind, index, total):
    """
    Builds the default (auto-generated) legend label prefix for a period
    entry that was passed without an explicit label, based on which
    category it belongs to. Labels are rendered in LaTeX math mode so they
    show up nicely typeset in the legend.

    Arguments:
    kind : str or None
        "activity" -> rotation-related harmonics: "$P_{\\mathrm{rot}}$",
        "$P_{\\mathrm{rot}}/2$", "$P_{\\mathrm{rot}}/3$", ... (in the order
        the periods were given).
        "planet" -> candidate/known planets: "$P_1$", "$P_2$", "$P_3$", ...
        "other" -> unidentified periods: "$P_X$" if there's only one, else
        "$P_{X1}$", "$P_{X2}$", ...
        None -> no prefix is used (caller falls back to the bare period
        value, as before).
    index : int
        Position of this period within its category's list (0-based).
    total : int
        Total number of periods in this category's list.

    Returns a LaTeX string prefix, or None if `kind` is None.
    """
    if kind == "activity":
        if index == 0:
            return r"$P_{\mathrm{rot}}$"
        return rf"$P_{{\mathrm{{rot}}}}/{index + 1}$"
    if kind == "planet":
        return rf"$P_{{{index + 1}}}$"
    if kind == "other":
        return r"$P_X$" if total == 1 else rf"$P_{{X{index + 1}}}$"
    return None


def _normalize_periods(entries, kind=None):
    """
    Normalizes `entries` into a list of (label, period) pairs.

    Arguments:
    entries : None, float, list of floats, list of (label, period) pairs, or dict
        Accepts a single period, a plain list of periods (auto-labeled
        according to `kind`, in LaTeX), a list of (label, period) pairs, or
        a {label: period} dict. Explicit labels (pairs/dict) are always
        kept exactly as given (not converted to LaTeX), regardless of
        `kind` -- wrap them in "$...$" yourself if you want LaTeX
        rendering for a custom label too.
    kind : str or None
        Category used to auto-generate LaTeX labels for plain (unlabeled)
        period entries -- "activity", "planet" or "other" (see
        `_default_label`). If None, unlabeled entries fall back to their
        bare period value (e.g. "27.50 d"), as before.

    Returns a list of (label, period) tuples.
    """
    if entries is None:
        return []
    if isinstance(entries, dict):
        return [(str(k), float(v)) for k, v in entries.items()]

    if np.isscalar(entries):
        period = float(entries)
        prefix = _default_label(kind, 0, 1)
        label = f"{prefix} = {period:.2f} d" if prefix else f"{period:.2f} d"
        return [(label, period)]

    entries = list(entries)
    total = len(entries)
    normalized = []
    for i, item in enumerate(entries):
        if isinstance(item, (tuple, list)) and len(item) == 2 and isinstance(item[0], str):
            normalized.append((item[0], float(item[1])))
        else:
            period = float(item)
            prefix = _default_label(kind, i, total)
            label = f"{prefix} = {period:.2f} d" if prefix else f"{period:.2f} d"
            normalized.append((label, period))
    return normalized


def _find_global_scalogram_axis(fig, primary_ax, orientation):
    """
    Finds the "global wavelet spectrum" panel: the narrow axis to the right
    of the scalogram/colorbar that shares the exact same period range as
    `primary_ax` (used by wavepal to plot the confidence-level curves next
    to the scalogram).

    Arguments:
    fig : matplotlib.figure.Figure
        The scalogram figure.
    primary_ax : matplotlib.axes.Axes
        The main scalogram panel (found via `_find_primary_period_axis`).
    orientation : str
        "y" if the period axis is vertical, "x" if horizontal.

    Returns the matching Axes, or None if not found.
    """
    primary_lim = primary_ax.get_ylim() if orientation == "y" else primary_ax.get_xlim()
    candidates = []
    for ax in fig.axes:
        if ax is primary_ax:
            continue
        lim = ax.get_ylim() if orientation == "y" else ax.get_xlim()
        if np.allclose(lim, primary_lim, rtol=1e-6):
            candidates.append(ax)
    if not candidates:
        return None
    return max(candidates, key=lambda a: a.get_position().x0)


def _find_colorbar_axis(fig, primary_ax, global_ax, rv_ax):
    """
    Finds the colorbar axis: the narrow axis sitting between the main
    scalogram panel (`primary_ax`) and the global wavelet spectrum panel
    (`global_ax`), i.e. the axis whose left edge sits at or past
    `primary_ax`'s right edge and whose right edge sits at or before
    `global_ax`'s left edge.

    Arguments:
    fig : matplotlib.figure.Figure
        The scalogram figure.
    primary_ax : matplotlib.axes.Axes
        The main scalogram panel.
    global_ax : matplotlib.axes.Axes or None
        The global wavelet spectrum panel (may be None).
    rv_ax : matplotlib.axes.Axes or None
        The RV panel (excluded from consideration).

    Returns the colorbar Axes, or None if not found.
    """
    primary_bbox = primary_ax.get_position()
    global_x0 = global_ax.get_position().x0 if global_ax is not None else np.inf

    best = None
    for ax in fig.axes:
        if ax in (primary_ax, global_ax, rv_ax):
            continue
        bbox = ax.get_position()
        if bbox.x0 < primary_bbox.x1 - 1e-6:
            continue
        if bbox.x0 >= global_x0 - 1e-6:
            continue
        if best is None or bbox.x0 < best.get_position().x0:
            best = ax
    return best


def _build_combined_legend(fig, percentile, cl_colors, groups,
                           linestyle="--", linewidth=2.5, gap_frac=0.07):
    from matplotlib.lines import Line2D

    primary_ax, orientation = _find_primary_period_axis(fig)
    if primary_ax is None:
        return

    old_legend = primary_ax.get_legend()
    if old_legend is not None:
        old_legend.remove()

    cl_handles = [Line2D([0], [0], color=color, linewidth=linewidth)
                  for color in cl_colors[:len(percentile)]]
    cl_labels = [f"{pct:g}% CL" for pct in percentile]

    period_handles, period_labels = [], []
    for color, entries in groups:
        for label, _period in entries:
            period_handles.append(Line2D([0], [0], color=color, linestyle=linestyle, linewidth=linewidth))
            period_labels.append(label)

    handles = cl_handles + period_handles
    labels = cl_labels + period_labels
    if not handles:
        return

    rv_ax = _find_rv_axis(fig)
    global_ax = _find_global_scalogram_axis(fig, primary_ax, orientation)
    colorbar_ax = _find_colorbar_axis(fig, primary_ax, global_ax, rv_ax)

    legend_fontsize = 24

    legend_style = dict(
        ncol=2, 
        fontsize=legend_fontsize,
        framealpha=0.95,
        facecolor="white",
        edgecolor="0.35",
        handlelength=2.3,
        handletextpad=0.9,
        labelspacing=0.6,
        columnspacing=2.0,
        borderpad=1.1,
        fancybox=True,
        shadow=True,
    )

    def _place_legend(x0, x1, y0, y1):
        width, height = x1 - x0, y1 - y0

        legend = fig.legend(
            handles, labels, loc="center",
            bbox_to_anchor=(x0, y0, width, height),
            bbox_transform=fig.transFigure,
            mode="expand", borderaxespad=0,
            **legend_style,
        )
        fig.canvas.draw()
        measured_height = legend.get_window_extent(
            fig.canvas.get_renderer()
        ).transformed(fig.transFigure.inverted()).height

        if measured_height > 0:
            scale = float(np.clip(height / measured_height, 0.5, 4.0))
        else:
            scale = 1.0

        if abs(scale - 1.0) > 0.03:
            legend.remove()
            scaled_style = dict(legend_style)
            scaled_style["fontsize"] = legend_fontsize * scale
            scaled_style["labelspacing"] = legend_style["labelspacing"] * scale
            scaled_style["borderpad"] = legend_style["borderpad"] * scale
            scaled_style["handletextpad"] = legend_style["handletextpad"] * scale
            legend = fig.legend(
                handles, labels, loc="center",
                bbox_to_anchor=(x0, y0, width, height),
                bbox_transform=fig.transFigure,
                mode="expand", borderaxespad=0,
                **scaled_style,
            )

    if rv_ax is not None and global_ax is not None and colorbar_ax is not None:
        rv_bbox = rv_ax.get_position()
        cbar_bbox = colorbar_ax.get_position()
        global_bbox = global_ax.get_position()

        _place_legend(cbar_bbox.x0, global_bbox.x1, rv_bbox.y0, rv_bbox.y1)

    elif rv_ax is not None:
        rv_bbox = rv_ax.get_position()
        period_bbox = primary_ax.get_position()

        existing_gap = rv_bbox.y0 - period_bbox.y1
        if existing_gap < gap_frac:
            new_rv_y0 = rv_bbox.y1 - (rv_bbox.height - (gap_frac - max(existing_gap, 0.)))
            rv_ax.set_position([rv_bbox.x0, new_rv_y0, rv_bbox.width, rv_bbox.y1 - new_rv_y0])
            gap_bottom, gap_top = period_bbox.y1, new_rv_y0
        else:
            gap_bottom, gap_top = period_bbox.y1, rv_bbox.y0

        _place_legend(rv_bbox.x0, rv_bbox.x1, gap_bottom, gap_top)

    else:
        legend = primary_ax.legend(handles, labels, loc="upper right", **legend_style)

    def _place_legend(x0, x1, y0, y1):
        """
        Draws the legend stretched to fill the [x0, x1] x [y0, y1] figure-
        fraction rectangle exactly: full width (mode="expand") and, since
        legends don't natively stretch vertically, a measured/rescaled
        font size so the content's height also fills [y0, y1].
        """
        width, height = x1 - x0, y1 - y0

        legend = fig.legend(
            handles, labels, loc="center",
            bbox_to_anchor=(x0, y0, width, height),
            bbox_transform=fig.transFigure,
            mode="expand", borderaxespad=0,
            **legend_style,
        )
        fig.canvas.draw()
        measured_height = legend.get_window_extent(
            fig.canvas.get_renderer()
        ).transformed(fig.transFigure.inverted()).height

        if measured_height > 0:
            scale = float(np.clip(height / measured_height, 0.5, 4.0))
        else:
            scale = 1.0

        if abs(scale - 1.0) > 0.03:
            legend.remove()
            scaled_style = dict(legend_style)
            scaled_style["fontsize"] = legend_fontsize * scale
            scaled_style["labelspacing"] = legend_style["labelspacing"] * scale
            scaled_style["borderpad"] = legend_style["borderpad"] * scale
            scaled_style["handletextpad"] = legend_style["handletextpad"] * scale
            legend = fig.legend(
                handles, labels, loc="center",
                bbox_to_anchor=(x0, y0, width, height),
                bbox_transform=fig.transFigure,
                mode="expand", borderaxespad=0,
                **scaled_style,
            )

    if rv_ax is not None and global_ax is not None and colorbar_ax is not None:
        rv_bbox = rv_ax.get_position()
        cbar_bbox = colorbar_ax.get_position()
        global_bbox = global_ax.get_position()

        _place_legend(cbar_bbox.x0, global_bbox.x1, rv_bbox.y0, rv_bbox.y1)

    elif rv_ax is not None:
        rv_bbox = rv_ax.get_position()
        period_bbox = primary_ax.get_position()

        existing_gap = rv_bbox.y0 - period_bbox.y1
        if existing_gap < gap_frac:
            new_rv_y0 = rv_bbox.y1 - (rv_bbox.height - (gap_frac - max(existing_gap, 0.)))
            rv_ax.set_position([rv_bbox.x0, new_rv_y0, rv_bbox.width, rv_bbox.y1 - new_rv_y0])
            gap_bottom, gap_top = period_bbox.y1, new_rv_y0
        else:
            gap_bottom, gap_top = period_bbox.y1, rv_bbox.y0

        _place_legend(rv_bbox.x0, rv_bbox.x1, gap_bottom, gap_top)

    else:
        legend = primary_ax.legend(handles, labels, loc="upper right", **legend_style)


def _find_primary_period_axis(fig):
    """
    Finds the axis whose label identifies it as the period axis (the main
    scalogram panel, labeled "Period (days)" by wavepal), and on which
    orientation ("y" or "x") that axis lives.

    Arguments:
    fig : matplotlib.figure.Figure
        The scalogram figure.
    """
    for ax in fig.axes:
        if "period" in (ax.get_ylabel() or "").lower():
            return ax, "y"
        if "period" in (ax.get_xlabel() or "").lower():
            return ax, "x"
    return None, None


def _period_transform_from_ticks(ax, orientation):
    """
    Builds a period(days) -> axis-data-coordinate function from the period
    ticks/labels wavepal already placed on `ax`. wavepal's period axis is
    NOT a normal matplotlib log-scale axis: it's a linear axis in an
    internal coordinate, with custom tick labels (e.g. "50.0", "45.0", ...)
    placed by hand at the right positions. Reading those existing
    (position, real period value) pairs back and interpolating in
    log10(period) is how we figure out where a period we want to mark
    (that may not already be a tick) really belongs.

    Arguments:
    ax : matplotlib.axes.Axes
        The axis carrying the period ticks (found via
        `_find_primary_period_axis`).
    orientation : str
        "y" if the period axis is vertical, "x" if horizontal.

    Returns a function period(days) -> axis-data-coordinate, or None if
    fewer than 2 numeric period ticks were found on `ax`.
    """
    ticks = ax.get_yticks() if orientation == "y" else ax.get_xticks()
    tick_labels = ax.get_yticklabels() if orientation == "y" else ax.get_xticklabels()

    positions, values = [], []
    for tick, lbl in zip(ticks, tick_labels):
        text = lbl.get_text().strip()
        try:
            val = float(text)
        except ValueError:
            continue
        if val > 0:
            positions.append(tick)
            values.append(val)

    if len(values) < 2:
        return None

    log_values = np.log10(values)
    order = np.argsort(log_values)
    log_values_sorted = np.array(log_values)[order]
    positions_sorted = np.array(positions)[order]

    def transform(period):
        return float(np.interp(np.log10(period), log_values_sorted, positions_sorted))

    return transform


def _drop_periods_close_to(periods, reference_periods, rel_tol=0.05):
    """
    Removes entries of `periods` that fall within a relative tolerance of
    any value in `reference_periods`. Used to keep wavepal's own white
    dashed period-tick lines from being drawn at (or very near) an
    activity/planet/other period, where they would visually compete with
    the colored reference line at that same spot.

    Arguments:
    periods : list of floats
        Candidate periods (days) to filter.
    reference_periods : list of floats
        Periods (days) to avoid being close to.
    rel_tol : float
        Relative tolerance (e.g. 0.05 = 5%) below which two periods are
        considered to coincide, given the log-scale period axis.

    Returns the filtered list of periods.
    """
    if not reference_periods:
        return list(periods)
    return [p for p in periods
            if not any(abs(p - q) <= rel_tol * q for q in reference_periods)]


def _mark_periods(fig, periods, color, permin, permax, linestyle="--",
                  linewidth=2.5, alpha=0.9, zorder=1.5):
    """
    Draws reference lines at `periods` (days), in `color`, on the main
    scalogram panel (found via `_find_primary_period_axis`) and on any
    other panel sharing the exact same period coordinate range (e.g. the
    global wavelet spectrum panel, which is a separate Axes but plotted
    over the same period range).

    The real drawing position is computed with
    `_period_transform_from_ticks`, since wavepal's period axis is not a
    plain log-scale matplotlib axis (see that function's docstring).

    Arguments:
    fig : matplotlib.figure.Figure
        The scalogram figure.
    periods : float or list of floats
        Period(s) to mark, in days.
    color : str
        Matplotlib color for the reference lines.
    permin : float
        Minimum period of the analysis (periods below this are skipped).
    permax : float
        Maximum period of the analysis (periods above this are skipped).
    linestyle : str
        Line style for the reference lines.
    linewidth : float
        Line width for the reference lines.
    alpha : float
        Transparency of the reference lines.
    zorder : float
        Drawing order. Kept below the default Line2D z-order (2) that
        wavepal itself uses for its white/black dashed period-tick lines
        and confidence-level curves, so those stay visible on top of these
        reference lines, while still sitting above the scalogram image
        (Collections/Images default to z-order 0-1).
    """
    if periods is None:
        return
    periods = [periods] if np.isscalar(periods) else list(periods)
    periods = [p for p in periods if permin <= p <= permax]
    if not periods:
        return

    primary_ax, orientation = _find_primary_period_axis(fig)
    if primary_ax is None:
        return  # no panel labeled "Period (...)" found, nothing to anchor to

    transform = _period_transform_from_ticks(primary_ax, orientation)
    if transform is None:
        return  # not enough numeric period ticks to build the conversion

    primary_lim = primary_ax.get_ylim() if orientation == "y" else primary_ax.get_xlim()

    for ax in fig.axes:
        lim = ax.get_ylim() if orientation == "y" else ax.get_xlim()
        if not np.allclose(lim, primary_lim, rtol=1e-6):
            continue  # this panel doesn't share the period coordinate system

        for p in periods:
            pos = transform(p)
            if orientation == "y":
                ax.axhline(pos, color=color, linestyle=linestyle,
                           linewidth=linewidth, alpha=alpha, zorder=zorder)
            else:
                ax.axvline(pos, color=color, linestyle=linestyle,
                           linewidth=linewidth, alpha=alpha, zorder=zorder)


def analyze_and_plot(wave, w0=7.0, permin=1.0, permax=200.0, deltaj=0.01,
                     percentile=DEFAULT_PERCENTILES, time_string=None,
                     period_string=None, dashed_periods=None,
                     activity_periods=None, planet_periods=None, other_periods=None,
                     activity_color=DEFAULT_ACTIVITY_COLOR,
                     planet_color=DEFAULT_PLANET_COLOR,
                     other_color=DEFAULT_OTHER_COLOR,
                     mark_linewidth=2.5, mark_linestyle="--", mark_alpha=0.9,
                     figsize=(24, 12), **plot_kwargs):
    """
    Runs wave.timefreq_analysis with a fixed w0 and returns the scalogram figure.

    Arguments:
    wave : wavepal.Wavepal
        The Wavepal object to analyze.
    w0 : float
        The wavelet parameter for the CWT.
    permin : float
        Minimum period for the analysis.
    permax : float
        Maximum period for the analysis.
    deltaj : float
        Scalogram resolution.
    percentile : tuple of floats
        Percentiles to compute for the scalogram.
    time_string : list of floats, optional
        Custom time ticks for the scalogram.
    period_string : list of floats, optional
        Custom period ticks for the scalogram.
    dashed_periods : list of floats, optional
        Periods to be highlighted with dashed lines on the scalogram (all
        drawn with wavepal's own single fixed style).
    activity_periods : float, list of floats, list of (label, period) pairs, or dict, optional
        Rotation-related periods (Prot, Prot/2, ...), marked in `activity_color`
        (default: orange) and listed in the combined legend. Drawn as colored
        reference lines on top of the figure, since wavepal itself has no
        per-line color option; any `dashed_periods`/`period_string` entry
        that nearly coincides with one of these is dropped, so wavepal's own
        white dashed line never competes with the colored one at that period.
        Plain periods (no explicit label) are auto-labeled, in LaTeX, in the
        order given as "$P_{\\mathrm{rot}}$", "$P_{\\mathrm{rot}}/2$",
        "$P_{\\mathrm{rot}}/3$", ...; pass a list of (label, period) pairs or
        a {label: period} dict instead if you want custom labels (e.g.
        {"$P_{\\mathrm{rot}}$": 12.3, "$P_{\\mathrm{rot}}/2$": 6.15}).
    planet_periods : float, list of floats, list of (label, period) pairs, or dict, optional
        Known/candidate planet periods, marked in `planet_color`
        (default: navy). Same treatment as `activity_periods`; plain
        periods are auto-labeled, in LaTeX, "$P_1$", "$P_2$", "$P_3$", ...
        in the order given.
    other_periods : float, list of floats, list of (label, period) pairs, or dict, optional
        Periods of unknown/unclear origin, marked in `other_color`
        (default: a soft/pale yellow). Same treatment as `activity_periods`;
        plain periods are auto-labeled, in LaTeX, "$P_X$" (single period) or
        "$P_{X1}$", "$P_{X2}$", ... (several periods).
    activity_color : str
        Matplotlib color used for `activity_periods`.
    planet_color : str
        Matplotlib color used for `planet_periods`.
    other_color : str
        Matplotlib color used for `other_periods`.
    mark_linewidth : float
        Line width shared by the activity/planet/other reference lines.
    mark_linestyle : str
        Line style shared by the activity/planet/other reference lines.
    mark_alpha : float
        Transparency shared by the activity/planet/other reference lines.
    figsize : tuple of floats
        Size of the figure to be created.
    plot_kwargs : dict
        Additional keyword arguments to pass to the plotting function.
    """
    time_string = time_string or DEFAULT_TIME_STRING
    period_string = period_string or DEFAULT_PERIOD_STRING

    activity_entries = _normalize_periods(activity_periods, kind="activity")
    planet_entries = _normalize_periods(planet_periods, kind="planet")
    other_entries = _normalize_periods(other_periods, kind="other")

    marked_periods = [p for _, p in activity_entries + planet_entries + other_entries]
    dashed_periods = list(dashed_periods) if dashed_periods is not None else list(period_string)
    dashed_periods = _drop_periods_close_to(dashed_periods, marked_periods)

    wave.timefreq_analysis(
        theta=wave.t, w0=float(w0), permin=permin, permax=permax, deltaj=deltaj,
        percentile=np.asarray(percentile), computes_amplitude=True, smoothing_coeff=0.0,
    )
    wave.plot_scalogram_custom(
        color_cl_anal=['indigo', 'black', 'orchid'],
        fontsize_ticks=20, fontsize_axes=20,
        time_string=time_string, period_string=period_string,
        dashed_periods=dashed_periods, linewidth_cl=4, decimals=2,
        linewidth_gscal=2.0, figsize=figsize, **plot_kwargs,
    )
    # plot_scalogram_custom's return value is not reliably the matplotlib
    # Figure itself (wavepal's own examples grab it via plt.gcf() after
    # calling plot_scalogram/plot_scalogram_custom) -- use that instead, so
    # the reference lines below actually land on the real figure.
    fig = plt.gcf()

    _mark_periods(fig, [p for _, p in activity_entries], activity_color, permin, permax,
                  linestyle=mark_linestyle, linewidth=mark_linewidth, alpha=mark_alpha)
    _mark_periods(fig, [p for _, p in planet_entries], planet_color, permin, permax,
                  linestyle=mark_linestyle, linewidth=mark_linewidth, alpha=mark_alpha)
    _mark_periods(fig, [p for _, p in other_entries], other_color, permin, permax,
                  linestyle=mark_linestyle, linewidth=mark_linewidth, alpha=mark_alpha)

    _build_combined_legend(
        fig, percentile, ['indigo', 'black', 'orchid'],
        [(activity_color, activity_entries), (planet_color, planet_entries), (other_color, other_entries)],
        linestyle=mark_linestyle, linewidth=mark_linewidth,
    )

    return fig


_RANGE_PATTERN = re.compile(
    r"Re-estimated period range:\s*from\s+([\d.eE+-]+)\s+to\s+([\d.eE+-]+)"
)


def w0_loop(wave, output_dir, w0_min=5.5, w0_max=196.0, w0_step=2.0,
            permin=1.0, permax=200.0, deltaj=0.05, percentile=DEFAULT_PERCENTILES,
            time_string=None, period_string=None, dashed_periods=None,
            activity_periods=None, planet_periods=None, other_periods=None,
            activity_color=DEFAULT_ACTIVITY_COLOR, planet_color=DEFAULT_PLANET_COLOR,
            other_color=DEFAULT_OTHER_COLOR, figsize=(24, 12),
            save_period_ranges=True, verbose=True):
    """
    w0 sweep: for each value, saves a scalogram PDF and, if
    `save_period_ranges`, captures the "Re-estimated period range" that
    wavepal prints, to keep a record of the reliable period range for
    each w0.

    Arguments:
    wave : wavepal.Wavepal
        The Wavepal object to analyze.
    output_dir : str
        Directory where the scalogram PDFs and period ranges will be saved.
    w0_min : float
        Minimum w0 value for the sweep.
    w0_max : float
        Maximum w0 value for the sweep.
    w0_step : float
        Step size for the w0 sweep.
    permin : float
        Minimum period for the analysis.
    permax : float
        Maximum period for the analysis.
    deltaj : float
        Scalogram resolution.
    percentile : tuple of floats
        Percentiles to compute for the scalogram.
    time_string : list of floats, optional
        Custom time ticks for the scalogram.
    period_string : list of floats, optional
        Custom period ticks for the scalogram.
    dashed_periods : list of floats, optional
        Periods to be highlighted with dashed lines on the scalogram (all
        drawn with wavepal's own single fixed style).
    activity_periods : float, list of floats, list of (label, period) pairs, or dict, optional
        Rotation-related periods (Prot, Prot/2, ...), marked in `activity_color`
        (default: orange). See `analyze_and_plot` for details.
    planet_periods : float, list of floats, list of (label, period) pairs, or dict, optional
        Known/candidate planet periods, marked in `planet_color`
        (default: navy). See `analyze_and_plot` for details.
    other_periods : float, list of floats, list of (label, period) pairs, or dict, optional
        Periods of unknown/unclear origin, marked in `other_color`
        (default: a soft/pale yellow). See `analyze_and_plot` for details.
    activity_color : str
        Matplotlib color used for `activity_periods`.
    planet_color : str
        Matplotlib color used for `planet_periods`.
    other_color : str
        Matplotlib color used for `other_periods`.
    figsize : tuple of floats
        Size of the figure to be created.
    save_period_ranges : bool
        If True, captures the "Re-estimated period range" from wavepal's output and saves it to a text file.
    verbose : bool
        If True, prints progress and warnings during the sweep.
    """
    os.makedirs(output_dir, exist_ok=True)
    w0_values = np.arange(w0_min, w0_max + w0_step / 2, w0_step)
    ranges_file = os.path.join(output_dir, "period_ranges.txt")

    results = []
    for w0 in w0_values:
        if save_period_ranges:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                fig = analyze_and_plot(
                    wave, w0=w0, permin=permin, permax=permax, deltaj=deltaj,
                    percentile=percentile, time_string=time_string,
                    period_string=period_string, dashed_periods=dashed_periods,
                    activity_periods=activity_periods, planet_periods=planet_periods,
                    other_periods=other_periods, activity_color=activity_color,
                    planet_color=planet_color, other_color=other_color,
                    figsize=figsize,
                )
            captured_text = buf.getvalue()
            matches = _RANGE_PATTERN.findall(captured_text)
            if matches:
                per_min, per_max = matches[-1]
                per_min, per_max = float(per_min), float(per_max)
            else:
                per_min, per_max = np.nan, np.nan
                if verbose:
                    print(f"[WARNING] Could not find 'Re-estimated period range' for w0={w0:.2f}")
            results.append((float(w0), per_min, per_max))
        else:
            fig = analyze_and_plot(
                wave, w0=w0, permin=permin, permax=permax, deltaj=deltaj,
                percentile=percentile, time_string=time_string,
                period_string=period_string, dashed_periods=dashed_periods,
                activity_periods=activity_periods, planet_periods=planet_periods,
                other_periods=other_periods, activity_color=activity_color,
                planet_color=planet_color, other_color=other_color,
                figsize=figsize,
            )

        output_file = os.path.join(output_dir, f"scalogram_w0_{w0:.2f}.pdf")
        fig.savefig(output_file, bbox_inches="tight")
        plt.close(fig)

        if save_period_ranges:
            with open(ranges_file, "w") as f:
                f.write("w0\tperiod_min\tperiod_max\n")
                for w0_i, pmin_i, pmax_i in results:
                    f.write(f"{w0_i:.4f}\t{pmin_i:.6f}\t{pmax_i:.6f}\n")

    if verbose and save_period_ranges:
        print(f"Period ranges saved to: {ranges_file}")

    return results