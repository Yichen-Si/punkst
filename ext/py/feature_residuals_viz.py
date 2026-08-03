#!/usr/bin/env python3
"""Generate an interactive HTML visualization for a `*.feature_residuals.tsv`
file produced by `transform_gampois.cpp` / `transform_lda.cpp` with
`--residuals` enabled (see docs/modules/feature_eval.md).

Usage:
    python3 ext/py/feature_residuals_viz.py --input FILE.feature_residuals.tsv --output out.html
"""

import argparse
import csv
import json
import math
import os
import random
import re
import sys
from html import escape as html_escape

# Columns dropped from the variable list entirely (raw moments used only to
# derive the log-ratio variables below, plus absDiff which is superseded by
# absDiffRate).
DROP_COLUMNS = {"F_w", "Qa_w", "Q0_w", "absDiff"}

# Columns whose values are normalized by dividing by totCount (N_w).
NORMALIZE_BY_TOTCOUNT = {"marginalDev", "conditionalDev", "factorDrift"}

VAR_LABELS = {
    "absDiffRate": "absDiffRate (mean absolute residual rate)",
    "totCount": "totCount (N_w, observed corpus total)",
    "nUnits": "nUnits (units expressing the feature)",
    "log2Gain": "log2Gain (log2 observed/predicted feature total)",
    "marginalDev": "marginalDev / N_w (feature-total mismatch deviance)",
    "conditionalDev": "conditionalDev / N_w (unit-level deviance after gain adj.)",
    "factorDrift": "factorDrift / N_w (topic-allocation deviance)",
    "deletionTV": "deletionTV (mean one-step deletion effect)",
    "topicInformation": "topicInformation (corpus-relative KL information)",
    "cofeatureCorroboration": "cofeatureCorroboration (mean positive cofeature lift)",
    "cofeatureConflict": "cofeatureConflict (mean negative cofeature lift)",
    "adjAbsDiffRate": "adjAbsDiffRate (gain-adjusted positive-count residual rate)",
    "pull": "pull (residual-weighted deleted-background leverage)",
    "EVES_w": "EVES_w (excess variance explained by structure)",
    "TVES_w": "TVES_w (total variance explained by structure)",
    "U_w": "U_w (LDA posterior-topic uncertainty)",
    "obsExcess": "observed excess over depth-only null (log2 F_w/Q0_w)",
    "modelExcess": "excess predicted by the model (log2 Qa_w/Q0_w)",
}

# Preferred display order; anything else present falls back to file order.
VAR_ORDER = [
    "totCount", "nUnits", "log2Gain", "absDiffRate", "adjAbsDiffRate", "pull",
    "marginalDev", "conditionalDev", "factorDrift", "deletionTV",
    "obsExcess", "modelExcess", "EVES_w", "TVES_w", "U_w",
    "topicInformation", "cofeatureCorroboration", "cofeatureConflict",
]

# Short names shown in the variable dropdowns, the colorbar title and the
# hover readouts; the full VAR_LABELS text is reserved for axis titles.
# Only variables whose display name differs from the column key need an entry.
VAR_NAMES = {
    "marginalDev": "marginalDev / N_w",
    "conditionalDev": "conditionalDev / N_w",
    "factorDrift": "factorDrift / N_w",
}

# Variables whose axis scale starts on log whenever they are selected. The
# color channel ignores this list: it always starts on the quantile scale.
LOG_SCALE_DEFAULT_VARS = ["totCount", "deletionTV"]

# Variables that start with a preset [min, max] range whenever they are
# selected. EVES_w is a fraction whose out-of-range values flag topic
# under-explanation or a negative dispersion estimate, so the default view
# restricts it to the interpretable interval.
VAR_DEFAULT_RANGES = {"EVES_w": [0.0, 1.0]}

# How many top points per axis get a Feature label (union of both axes, so at
# most twice this many labels per panel).
LABELS_PER_AXIS = 10

# Cap on Feature labels drawn for a box/lasso selection, so a selection that
# covers thousands of points does not paint the panel solid with text.
MAX_SELECTION_LABELS = 30

# Bin count for the marginal histograms drawn above and to the right of each
# scatter. Bins are uniform in the displayed space, i.e. in log10 units on a
# log axis.
MARGINAL_BINS = 40

GLOBAL_FILTER_VARS = ["totCount", "nUnits", "log2Gain"]

# Starting x / y / color variable for each panel, in panel order.
PANEL_DEFAULTS = [
    {"x": "deletionTV", "y": "cofeatureConflict", "color": "topicInformation"},
    {"x": "cofeatureCorroboration", "y": "pull", "color": "TVES_w"},
    {"x": "obsExcess", "y": "modelExcess", "color": "absDiffRate"},
    {"x": "log2Gain", "y": "factorDrift", "color": "cofeatureConflict"},
]

# Panels swapped when this column is present (full standalone-transform
# diagnostics), to put the gain/drift panel earlier.
SWAP_PANELS_IF_PRESENT = ("adjAbsDiffRate", 1, 3)


def panel_defaults(ordered_vars):
    panels = [dict(d) for d in PANEL_DEFAULTS]
    col, a, b = SWAP_PANELS_IF_PRESENT
    if col in ordered_vars:
        panels[a], panels[b] = panels[b], panels[a]
    return panels


def to_float(raw):
    """Parse a TSV cell to a finite float, or None for NA/inf/garbage."""
    if raw is None:
        return None
    s = raw.strip()
    if s == "" or s.upper() in ("NA", "NAN", "N/A", "NULL", "."):
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def safe_log2_ratio(numerator, denominator):
    if numerator is None or denominator is None:
        return None
    if numerator <= 0.0 or denominator <= 0.0:
        return None
    return math.log2(numerator / denominator)


def load_records(path):
    """Read and parse a TSV without retaining a second copy of every row."""
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter="\t")
        try:
            header = next(reader)
        except StopIteration:
            sys.exit("Input file is empty: " + path)
        return build_records(header, reader)


def build_records(header, rows):
    idx = {name: i for i, name in enumerate(header)}
    if "Feature" not in idx:
        sys.exit("Input file has no 'Feature' column; is this a feature_residuals.tsv?")

    numeric_cols = [c for c in header if c != "Feature"]
    raw_available = set(numeric_cols)

    derived_available = []
    if {"F_w", "Q0_w"} <= raw_available:
        derived_available.append("obsExcess")
    if {"Qa_w", "Q0_w"} <= raw_available:
        derived_available.append("modelExcess")

    kept_cols = [c for c in numeric_cols if c not in DROP_COLUMNS]
    available_vars = kept_cols + derived_available
    ordered_vars = [v for v in VAR_ORDER if v in available_vars]
    ordered_vars += [v for v in available_vars if v not in ordered_vars]

    records = []
    for row in rows:
        if len(row) <= idx["Feature"]:
            continue
        parsed = {c: to_float(row[idx[c]]) if idx[c] < len(row) else None
                  for c in numeric_cols}
        tot = parsed.get("totCount")
        for c in NORMALIZE_BY_TOTCOUNT:
            if c in parsed:
                v = parsed[c]
                parsed[c] = v / tot if (v is not None and tot is not None and tot > 0) else None
        if "obsExcess" in derived_available:
            parsed["obsExcess"] = safe_log2_ratio(parsed.get("F_w"), parsed.get("Q0_w"))
        if "modelExcess" in derived_available:
            parsed["modelExcess"] = safe_log2_ratio(parsed.get("Qa_w"), parsed.get("Q0_w"))

        rec = {"Feature": row[idx["Feature"]]}
        for v in ordered_vars:
            rec[v] = parsed.get(v)
        records.append(rec)

    return ordered_vars, records


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>__TITLE__</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  :root {
    --surface-1: #fcfcfb;
    --page: #f9f9f7;
    --ink-primary: #0b0b0b;
    --ink-secondary: #52514e;
    --ink-muted: #898781;
    --grid: #e1e0d9;
    --baseline: #c3c2b7;
    --accent: #eb6834;
    --border: rgba(11,11,11,0.10);
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    background: var(--page);
    color: var(--ink-primary);
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    font-size: 13px;
  }
  header {
    padding: 14px 20px 10px;
    border-bottom: 1px solid var(--border);
    background: var(--surface-1);
  }
  header h1 { font-size: 16px; margin: 0 0 2px; }
  header .sub { color: var(--ink-secondary); font-size: 12px; }

  #global-filters {
    display: flex; flex-wrap: wrap; gap: 10px 18px;
    padding: 10px 20px; background: var(--surface-1);
    border-bottom: 1px solid var(--border); align-items: flex-end;
  }
  .gf-group { display: flex; flex-direction: column; gap: 4px; }
  .gf-group label { font-size: 11px; color: var(--ink-secondary); }
  .gf-row { display: flex; gap: 6px; align-items: center; }
  .gf-row input[type=number] { width: 90px; }
  .gf-row input[type=text] { width: 200px; }
  .gf-row label.cb {
    display: flex; gap: 4px; align-items: center; font-size: 12px;
    color: var(--ink-primary); white-space: nowrap; margin-right: 8px;
  }
  #regex-status { font-size: 11px; color: var(--ink-muted); }
  #regex-status.bad { color: #d03b3b; }
  #reset-filters {
    padding: 5px 10px; border: 1px solid var(--border); background: #fff;
    border-radius: 4px; cursor: pointer; font-size: 12px; color: var(--ink-secondary);
  }
  #reset-filters:hover { background: var(--grid); }
  #match-count { margin-left: auto; font-size: 12px; color: var(--ink-secondary); align-self: center; }
  .filter-actions-row {
    flex: 0 0 100%; display: flex; flex-wrap: wrap;
    gap: 10px 18px; align-items: flex-end;
  }

  #plots {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    padding: 12px 20px;
  }
  .panel {
    background: var(--surface-1);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 10px;
  }
  .ctrl-row { display: flex; gap: 6px; margin-bottom: 6px; align-items: center; flex-wrap: wrap; }
  .ctrl-row label { font-size: 11px; color: var(--ink-muted); min-width: 34px; }
  .ctrl-row label.role { min-width: 46px; font-weight: 600; color: var(--ink-secondary); }
  .ctrl-row select { flex: 1; min-width: 0; font-size: 12px; }
  .ctrl-row input[type=number] { width: 72px; font-size: 12px; }
  .ctrl-row.tight { gap: 10px; }
  .radio-inline { display: flex; gap: 8px; font-size: 11px; color: var(--ink-secondary); }
  .radio-inline label { display: flex; gap: 3px; align-items: center; min-width: 0; }
  .sel-row {
    display: flex; align-items: center; gap: 8px; margin: 6px 0 2px;
    font-size: 11px; color: var(--ink-muted);
  }
  .sel-row .sel-status.active { color: var(--accent); }
  .sel-row button {
    margin-left: auto; padding: 2px 8px; font-size: 11px; cursor: pointer;
    border: 1px solid var(--border); background: #fff; border-radius: 4px;
    color: var(--ink-secondary);
  }
  .sel-row button:hover { background: var(--grid); }
  .plot-div { position: relative; width: 100%; height: 540px; }
  /* Plotly leaves more title-to-tick spacing than this compact layout needs. */
  .plot-div .xtitle { transform: translateY(-3px); }
  .selection-overlay {
    position: absolute; inset: 0; width: 100%; height: 100%;
    pointer-events: none; z-index: 20;
  }
  .ctrl-row .radio-inline { margin-left: 4px; }
  .var-block { border-top: 1px solid var(--grid); padding-top: 6px; margin-top: 6px; }
  .var-block:first-child { border-top: none; padding-top: 0; margin-top: 0; }

  #hover-info {
    padding: 8px 20px 16px; font-size: 12px; color: var(--ink-secondary);
    display: flex; flex-wrap: wrap; gap: 14px 26px;
  }
  #hover-info .hi-feature { font-weight: 700; color: var(--ink-primary); font-size: 13px; }
  #hover-info .hi-item span.k { color: var(--ink-muted); }
  #hover-info .placeholder { color: var(--ink-muted); font-style: italic; }

  /* Custom hover popup: Plotly's own hoverlabel cannot align columns, so the
     traces use hoverinfo:"none" and this table is positioned at the cursor. */
  #tooltip {
    position: fixed; z-index: 40; display: none; pointer-events: none;
    background: #ffffff; color: #000000;
    border: 1px solid rgba(11,11,11,0.35); border-radius: 4px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.18);
    padding: 6px 8px; font-size: 11px;
  }
  #tooltip .tt-feature { display: block; font-weight: 700; font-size: 12px; margin-bottom: 4px; }
  #tooltip table { border-collapse: collapse; }
  #tooltip th {
    font-size: 10px; font-weight: 600; color: #555; text-align: right;
    padding: 0 0 3px 12px; border-bottom: 1px solid #e1e0d9;
  }
  #tooltip td {
    text-align: right; padding: 1px 0 1px 12px; white-space: nowrap;
    font-variant-numeric: tabular-nums;
  }
  #tooltip th.k, #tooltip td.k { text-align: left; padding-left: 0; }
  #tooltip td.k { color: inherit; }
</style>
</head>
<body>
<header>
  <h1>__TITLE__</h1>
  <div class="sub">__SUBTITLE__</div>
</header>

<div id="global-filters">
  <div class="gf-group" data-gf="totCount" style="__TOTCOUNT_DISPLAY__">
    <label>totCount</label>
    <div class="gf-row">
      <input type="number" step="any" placeholder="min" data-role="min" autocomplete="off">
      <span>&ndash;</span>
      <input type="number" step="any" placeholder="max" data-role="max" autocomplete="off">
    </div>
  </div>
  <div class="gf-group" data-gf="nUnits" style="__NUNITS_DISPLAY__">
    <label>nUnits</label>
    <div class="gf-row">
      <input type="number" step="any" placeholder="min" data-role="min" autocomplete="off">
      <span>&ndash;</span>
      <input type="number" step="any" placeholder="max" data-role="max" autocomplete="off">
    </div>
  </div>
  <div class="gf-group" data-gf="log2Gain" style="__LOG2GAIN_DISPLAY__">
    <label>log2Gain</label>
    <div class="gf-row">
      <input type="number" step="any" placeholder="min" data-role="min" autocomplete="off">
      <span>&ndash;</span>
      <input type="number" step="any" placeholder="max" data-role="max" autocomplete="off">
    </div>
  </div>
  <div class="gf-group">
    <label>Feature name regex</label>
    <div class="gf-row">
      <input type="text" id="regex-input" placeholder="e.g. ^MT-|^RPL" autocomplete="off">
    </div>
    <div id="regex-status">&nbsp;</div>
  </div>
  <div id="match-count"></div>
  <div class="filter-actions-row">
    <div class="gf-group">
      <label>Labels</label>
      <div class="gf-row">
        <label class="cb"><input type="checkbox" id="label-toggle" autocomplete="off"> extreme points</label>
        <label class="cb"><input type="checkbox" id="annotate-toggle" checked autocomplete="off"> selected points</label>
      </div>
    </div>
    <div class="gf-group">
      <label>Selection</label>
      <div class="gf-row">
        <label class="cb"><input type="checkbox" id="mute-background-toggle" checked autocomplete="off"> mute background points</label>
      </div>
    </div>
    <button id="reset-filters">Reset filters</button>
  </div>
</div>

<!-- Panels are created by the script, one per entry in panelDefaults. -->
<div id="plots"></div>

<div id="hover-info"><span class="placeholder">Hover a point to see its values here.</span></div>
<div id="tooltip"></div>

<script id="feature-data" type="application/json">__DATA_JSON__</script>
<script id="var-meta" type="application/json">__VARMETA_JSON__</script>
<script>
(function () {
  "use strict";
  var DATA = JSON.parse(document.getElementById("feature-data").textContent);
  var FEATURES = DATA.features;
  var COLUMNS = DATA.columns;
  var NRECORDS = FEATURES.length;
  var VAR_META = JSON.parse(document.getElementById("var-meta").textContent);
  var VARS = VAR_META.vars;
  var LABELS = VAR_META.labels;   // long text, axis titles only
  var NAMES = VAR_META.names;     // short variable names, everywhere else
  var GLOBAL_VARS = VAR_META.globalVars;
  var LOG_DEFAULT_VARS = VAR_META.logDefaultVars;
  var DEFAULT_RANGES = VAR_META.defaultRanges;
  var LABELS_PER_AXIS = VAR_META.labelsPerAxis;
  var MAX_SELECTION_LABELS = VAR_META.maxSelectionLabels;
  var MARGINAL_BINS = VAR_META.marginalBins;

  function defaultScale(varKey, role) {
    if (role === "color") return "quantile";
    return LOG_DEFAULT_VARS.indexOf(varKey) !== -1 ? "log" : "linear";
  }

  function defaultRange(varKey) {
    var r = DEFAULT_RANGES[varKey];
    return r ? { min: r[0], max: r[1] } : { min: null, max: null };
  }

  // Per-variable empirical percentile over every record with a value:
  // PCT[v][i] = (# values <= COLUMNS[v][i]) / (# non-missing values), so the
  // maximum is exactly 1 and tied values share a percentile. Left undefined
  // where the record has no value. Computed once over the full dataset, so
  // percentiles and the quantile color scale do not shift as filters change.
  var PCT = {};
  VARS.forEach(function (v) {
    var values = COLUMNS[v];
    var order = [];
    for (var i = 0; i < NRECORDS; i++) {
      var val = values[i];
      if (val !== null && val !== undefined && !isNaN(val)) order.push(i);
    }
    order.sort(function (a, b) { return values[a] - values[b]; });
    var col = new Float32Array(NRECORDS);
    col.fill(NaN);
    var n = order.length;
    var s = 0;
    while (s < n) {
      var e = s;
      while (e + 1 < n && values[order[e + 1]] === values[order[s]]) e++;
      var pct = (e + 1) / n;
      for (var k = s; k <= e; k++) col[order[k]] = pct;
      s = e + 1;
    }
    PCT[v] = col;
  });

  var SEQ_COLORSCALE = [
    [0.00, "#cde2fb"], [0.14, "#b7d3f6"], [0.29, "#9ec5f4"], [0.43, "#86b6ef"],
    [0.57, "#6da7ec"], [0.71, "#5598e7"], [0.86, "#3987e5"], [1.00, "#0d366b"]
  ];
  var HIGHLIGHT_COLOR = "#eb6834";
  var HIGHLIGHT_HALF = "rgba(235,104,52,0.5)";
  var BACKGROUND_COLOR = "#b4b3ab";
  var MARGINAL_FILL = "#cde2fb";
  var MARGINAL_LINE = "#86b6ef";
  // Selected points are redrawn on top, bigger, mostly opaque and outlined.
  var SELECTED_SIZE = 8;
  var SELECTED_OPACITY = 0.8;
  // Everything else while a selection is active. Plotly's default dimming
  // multiplies the marker opacity (0.5 * 0.2 = 0.1), which is too faint to read
  // as context, so both point traces set it explicitly instead. Colors are left
  // alone, so the unselected points keep the colorscale.
  var DIM_OPACITY = 0.2;
  var SELECTED_OUTLINE = "#33322f";
  var SELECTED_BG_COLOR = "#7c7b75";
  // Axis domains: the scatter keeps the lower-left block, the marginal count
  // axes take the strip above and to the right.
  var SCATTER_DOMAIN = [0, 0.85];
  var MARGINAL_DOMAIN = [0.87, 1];
  // Horizontal colorbar under the scatter: ends with the scatter's x domain,
  // starts far enough right to leave the strip on its left for the title.
  var CBAR_LEN = 0.5;
  var CBAR_X = SCATTER_DOMAIN[1] - CBAR_LEN;
  var CBAR_Y = -0.13;

  var DEFAULTS = VAR_META.panelDefaults;
  var NPANELS = DEFAULTS.length;

  function pick(list, fallback, used) {
    for (var i = 0; i < list.length; i++) {
      if (VARS.indexOf(list[i]) !== -1) return list[i];
    }
    for (var j = 0; j < VARS.length; j++) {
      if (used.indexOf(VARS[j]) === -1) return VARS[j];
    }
    return VARS[0];
  }

  var state = {
    global: {
      regex: null, labels: false, annotateSelection: true,
      muteBackgroundHover: true
    },
    panels: []
  };
  GLOBAL_VARS.forEach(function (v) { state.global[v] = { min: null, max: null }; });

  for (var p = 0; p < NPANELS; p++) {
    var used = state.panels.map(function (s) { return s.x; })
      .concat(state.panels.map(function (s) { return s.y; }));
    var d = DEFAULTS[p];
    var st0 = {
      x: pick([d.x], VARS[0], used),
      y: pick([d.y], VARS[Math.min(1, VARS.length - 1)], used),
      color: pick([d.color], VARS[Math.min(2, VARS.length - 1)], used),
      xClip: false, yClip: false, colorClip: false,
      // Remembered so picking box/lasso select on the modebar is not silently
      // reverted to "zoom" by the next redraw.
      dragmode: "zoom"
    };
    st0.xScale = defaultScale(st0.x, "x");
    st0.yScale = defaultScale(st0.y, "y");
    st0.colorScale = defaultScale(st0.color, "color");
    ["x", "y", "color"].forEach(function (role) {
      var r = defaultRange(st0[role]);
      st0[role + "Min"] = r.min;
      st0[role + "Max"] = r.max;
    });
    state.panels.push(st0);
  }

  function fieldEl(roleLabel, scaleId, clipId, withQuantile) {
    var scales = ["linear", "log"];
    if (withQuantile) scales.push("quantile");
    return (
      '<div class="var-block">' +
        '<div class="ctrl-row"><label class="role">' + roleLabel + '</label>' +
          '<select data-role="var">' +
            VARS.map(function (v) {
              return '<option value="' + v + '">' + NAMES[v] + '</option>';
            }).join("") +
          '</select>' +
        '</div>' +
        '<div class="ctrl-row tight">' +
          '<div class="radio-inline">' +
            scales.map(function (s) {
              return '<label><input type="radio" data-role="scale" name="' + scaleId +
                '" value="' + s + '"> ' + s + '</label>';
            }).join("") +
          '</div>' +
        '</div>' +
        '<div class="ctrl-row"><label>range</label>' +
          '<input type="number" step="any" data-role="min" placeholder="min">' +
          '<input type="number" step="any" data-role="max" placeholder="max">' +
          '<div class="radio-inline">' +
            '<label><input type="radio" data-role="clip" name="' + clipId + '" value="clip"> clip</label>' +
            '<label><input type="radio" data-role="clip" name="' + clipId + '" value="remove" checked> remove</label>' +
          '</div>' +
        '</div>' +
      '</div>'
    );
  }

  var selStatusEls = [];
  var panelsRoot = document.getElementById("plots");
  for (p = 0; p < NPANELS; p++) {
    // Built here rather than in the markup, so the panel count follows
    // panelDefaults and the two cannot drift apart.
    var panel = document.createElement("div");
    panel.className = "panel";
    panel.setAttribute("data-panel", p);
    panelsRoot.appendChild(panel);
    panel.innerHTML =
      fieldEl("X axis", "xscale" + p, "xclip" + p, false) +
      fieldEl("Y axis", "yscale" + p, "yclip" + p, false) +
      fieldEl("Color", "cscale" + p, "cclip" + p, true) +
      '<div class="sel-row">' +
        '<span class="sel-status" data-role="sel-status">no selection</span>' +
        '<button type="button" data-role="sel-clear">clear selection</button>' +
      '</div>' +
      '<div class="plot-div" id="plotdiv' + p + '"></div>';

    var blocks = panel.querySelectorAll(".var-block");
    wireBlock(blocks[0], p, "x");
    wireBlock(blocks[1], p, "y");
    wireBlock(blocks[2], p, "color");
    selStatusEls.push(panel.querySelector('[data-role="sel-status"]'));
    panel.querySelector('[data-role="sel-clear"]')
      .addEventListener("click", function () { clearAllSelections(); });
    // set initial select/radio values from state
    var st = state.panels[p];
    blocks[0].querySelector('select').value = st.x;
    blocks[1].querySelector('select').value = st.y;
    blocks[2].querySelector('select').value = st.color;
    syncScaleRadios(blocks[0], st.xScale);
    syncScaleRadios(blocks[1], st.yScale);
    syncScaleRadios(blocks[2], st.colorScale);
    syncRangeInputs(blocks[0], st.xMin, st.xMax);
    syncRangeInputs(blocks[1], st.yMin, st.yMax);
    syncRangeInputs(blocks[2], st.colorMin, st.colorMax);
  }

  function syncScaleRadios(block, scale) {
    block.querySelectorAll('input[data-role="scale"]').forEach(function (r) {
      r.checked = (r.value === scale);
    });
  }

  function syncRangeInputs(block, min, max) {
    block.querySelector('input[data-role="min"]').value = (min === null ? "" : min);
    block.querySelector('input[data-role="max"]').value = (max === null ? "" : max);
  }

  function wireBlock(block, panelIdx, role) {
    var sel = block.querySelector('select[data-role="var"]');
    var minI = block.querySelector('input[data-role="min"]');
    var maxI = block.querySelector('input[data-role="max"]');
    var scaleRadios = block.querySelectorAll('input[data-role="scale"]');
    var clipRadios = block.querySelectorAll('input[data-role="clip"]');

    sel.addEventListener("change", function () {
      // Switching variable resets the scale and range to that variable's
      // defaults, so e.g. totCount always comes up on log and EVES_w on
      // [0, 1] regardless of the previous pick.
      var st = state.panels[panelIdx];
      st[role] = sel.value;
      st[role + "Scale"] = defaultScale(sel.value, role);
      var r = defaultRange(sel.value);
      st[role + "Min"] = r.min;
      st[role + "Max"] = r.max;
      syncScaleRadios(block, st[role + "Scale"]);
      syncRangeInputs(block, r.min, r.max);
      redrawPanelNow(panelIdx);
    });
    minI.addEventListener("input", function () {
      state.panels[panelIdx][role + "Min"] = minI.value === "" ? null : parseFloat(minI.value);
      schedulePanelRedraw(panelIdx);
    });
    maxI.addEventListener("input", function () {
      state.panels[panelIdx][role + "Max"] = maxI.value === "" ? null : parseFloat(maxI.value);
      schedulePanelRedraw(panelIdx);
    });
    scaleRadios.forEach(function (r) {
      r.addEventListener("change", function () {
        if (r.checked) { state.panels[panelIdx][role + "Scale"] = r.value; redrawPanelNow(panelIdx); }
      });
    });
    clipRadios.forEach(function (r) {
      r.addEventListener("change", function () {
        if (r.checked) {
          state.panels[panelIdx][role + "Clip"] = (r.value === "clip");
          redrawPanelNow(panelIdx);
        }
      });
    });
  }

  // ---- global filters ----
  document.querySelectorAll("#global-filters .gf-group[data-gf]").forEach(function (grp) {
    var v = grp.getAttribute("data-gf");
    if (GLOBAL_VARS.indexOf(v) === -1) return;
    var minI = grp.querySelector('input[data-role="min"]');
    var maxI = grp.querySelector('input[data-role="max"]');
    minI.addEventListener("input", function () {
      state.global[v].min = minI.value === "" ? null : parseFloat(minI.value);
      scheduleAllRedraw();
    });
    maxI.addEventListener("input", function () {
      state.global[v].max = maxI.value === "" ? null : parseFloat(maxI.value);
      scheduleAllRedraw();
    });
  });

  var regexInput = document.getElementById("regex-input");
  var regexStatus = document.getElementById("regex-status");
  function applyRegexInput() {
    var text = regexInput.value;
    if (text === "") {
      state.global.regex = null;
      regexStatus.textContent = " ";
      regexStatus.classList.remove("bad");
    } else {
      try {
        state.global.regex = new RegExp(text);
        regexStatus.textContent = " ";
        regexStatus.classList.remove("bad");
      } catch (e) {
        state.global.regex = null;
        regexStatus.textContent = "invalid regex: " + e.message;
        regexStatus.classList.add("bad");
      }
    }
  }

  regexInput.addEventListener("input", function () {
    applyRegexInput();
    scheduleAllRedraw();
  });

  var labelToggle = document.getElementById("label-toggle");
  labelToggle.addEventListener("change", function () {
    state.global.labels = labelToggle.checked;
    for (var pi = 0; pi < NPANELS; pi++) updateExtremeLabels(pi);
  });

  // Selected-point names can be turned off without giving up the selection: the
  // cross-panel highlight stays. No redraw, which would drop the selection.
  var annotateToggle = document.getElementById("annotate-toggle");
  annotateToggle.addEventListener("change", function () {
    state.global.annotateSelection = annotateToggle.checked;
    if (focalPanel !== null) syncSelectionLabels(focalPanel);
  });

  // With an active selection, background points are excluded from the custom
  // hover target set by default. This changes hover behavior only: it does not
  // redraw traces or alter the selected records.
  var muteBackgroundToggle = document.getElementById("mute-background-toggle");
  muteBackgroundToggle.addEventListener("change", function () {
    state.global.muteBackgroundHover = muteBackgroundToggle.checked;
    if (state.global.muteBackgroundHover && focalRecords &&
        lastHoverIdx !== null && !focalRecords[lastHoverIdx]) onUnhover();
  });

  document.getElementById("reset-filters").addEventListener("click", function () {
    GLOBAL_VARS.forEach(function (v) { state.global[v] = { min: null, max: null }; });
    state.global.regex = null;
    regexInput.value = "";
    regexStatus.textContent = " ";
    regexStatus.classList.remove("bad");
    document.querySelectorAll("#global-filters input[type=number]").forEach(function (i) { i.value = ""; });
    redrawAllNow();
  });

  // The global filters are panel-independent, so they are evaluated once per
  // redraw into a mask instead of once per panel plus once for the counter.
  // Only a global redraw refreshes it; panel-local redraws reuse the mask.
  var globalMask = new Uint8Array(NRECORDS);
  var globalPassCount = 0;

  function computeGlobalMask() {
    var n = 0;
    for (var i = 0; i < NRECORDS; i++) {
      var ok = passesGlobalFilters(i) ? 1 : 0;
      globalMask[i] = ok;
      n += ok;
    }
    globalPassCount = n;
  }

  function passesGlobalFilters(recIdx) {
    for (var i = 0; i < GLOBAL_VARS.length; i++) {
      var v = GLOBAL_VARS[i];
      var f = state.global[v];
      var val = COLUMNS[v][recIdx];
      if (f.min !== null && (val === null || val === undefined || isNaN(val) || val < f.min)) return false;
      if (f.max !== null && (val === null || val === undefined || isNaN(val) || val > f.max)) return false;
    }
    if (state.global.regex && !state.global.regex.test(FEATURES[recIdx])) return false;
    return true;
  }

  // Transform a raw value under a given scale/range/clip policy.
  // Returns null if the point should be dropped for this axis.
  function transformValue(raw, scale, min, max, clip) {
    if (raw === null || raw === undefined || isNaN(raw)) return null;
    var v = raw;
    if (scale === "log") {
      if (v <= 0) return null; // cannot be represented on a log axis
    }
    if (min !== null && v < min) {
      if (clip) v = min; else return null;
    }
    if (max !== null && v > max) {
      if (clip) v = max; else return null;
    }
    return v;
  }

  var plotDivs = [];
  for (p = 0; p < NPANELS; p++) plotDivs.push(document.getElementById("plotdiv" + p));

  // A canvas above each Plotly graph carries selected markers. This preserves
  // the guaranteed foreground layer without creating one SVG node per selected
  // point (which becomes very expensive for linked selections across panels).
  var selectionCanvases = new Array(NPANELS);
  var selectionOverlay = new Array(NPANELS);
  var selectionDrawPending = new Array(NPANELS);
  function emptySelectionOverlay() {
    return {
      bg: [], fg: [], hover: null,
      extreme: { xs: [], ys: [], text: [] },
      selectedLabels: { xs: [], ys: [], text: [] }
    };
  }
  for (p = 0; p < NPANELS; p++) {
    selectionOverlay[p] = emptySelectionOverlay();
    selectionDrawPending[p] = false;
  }

  function ensureSelectionCanvas(panelIdx) {
    var gd = plotDivs[panelIdx];
    var canvas = selectionCanvases[panelIdx];
    if (canvas && canvas.parentNode === gd) return canvas;
    canvas = document.createElement("canvas");
    canvas.className = "selection-overlay";
    canvas.setAttribute("aria-hidden", "true");
    gd.appendChild(canvas);
    selectionCanvases[panelIdx] = canvas;
    return canvas;
  }

  function hexRgb(hex) {
    return [parseInt(hex.slice(1, 3), 16), parseInt(hex.slice(3, 5), 16),
            parseInt(hex.slice(5, 7), 16)];
  }

  var SEQ_COLOR_RGB = SEQ_COLORSCALE.map(function (stop) {
    return [stop[0], hexRgb(stop[1])];
  });

  function scaleColor(t) {
    if (!isFinite(t)) t = 0.5;
    t = Math.max(0, Math.min(1, t));
    var hi = 1;
    while (hi < SEQ_COLOR_RGB.length - 1 && t > SEQ_COLOR_RGB[hi][0]) hi++;
    var a = SEQ_COLOR_RGB[hi - 1], b = SEQ_COLOR_RGB[hi];
    var span = b[0] - a[0];
    var f = span > 0 ? (t - a[0]) / span : 0;
    var rgb = [0, 1, 2].map(function (i) { return Math.round(a[1][i] + f * (b[1][i] - a[1][i])); });
    return "rgb(" + rgb.join(",") + ")";
  }

  function drawCirclePath(ctx, positions, source, xa, ya) {
    for (var i = 0; i < positions.length; i++) {
      var k = positions[i];
      var px = xa._offset + xa.c2p(source.xs[k]);
      var py = ya._offset + ya.c2p(source.ys[k]);
      if (!isFinite(px) || !isFinite(py)) continue;
      ctx.moveTo(px + SELECTED_SIZE / 2, py);
      ctx.arc(px, py, SELECTED_SIZE / 2, 0, Math.PI * 2);
    }
  }

  function canvasPoint(x, y, xa, ya) {
    return [xa._offset + xa.c2p(x), ya._offset + ya.c2p(y)];
  }

  function drawCanvasLabels(ctx, labels, xa, ya, color, position) {
    if (!labels || !labels.text.length) return;
    ctx.fillStyle = color;
    ctx.font = "9px system-ui, -apple-system, Segoe UI, sans-serif";
    ctx.textAlign = position === "top" ? "center" : "left";
    ctx.textBaseline = position === "top" ? "bottom" : "middle";
    for (var i = 0; i < labels.text.length; i++) {
      var pt = canvasPoint(labels.xs[i], labels.ys[i], xa, ya);
      if (!isFinite(pt[0]) || !isFinite(pt[1])) continue;
      ctx.fillText(labels.text[i], pt[0] + (position === "top" ? 0 : 4),
                   pt[1] + (position === "top" ? -4 : 0));
    }
  }

  function drawSelectionOverlay(panelIdx) {
    selectionDrawPending[panelIdx] = false;
    var gd = plotDivs[panelIdx], data = lastFiltered[panelIdx];
    if (!gd || !data || !gd._fullLayout) return;
    var canvas = ensureSelectionCanvas(panelIdx);
    var cssW = gd.clientWidth, cssH = gd.clientHeight;
    if (!cssW || !cssH) return;
    var dpr = window.devicePixelRatio || 1;
    var pixelW = Math.max(1, Math.round(cssW * dpr));
    var pixelH = Math.max(1, Math.round(cssH * dpr));
    if (canvas.width !== pixelW || canvas.height !== pixelH) {
      canvas.width = pixelW; canvas.height = pixelH;
    }
    var ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssW, cssH);

    var overlay = selectionOverlay[panelIdx];
    canvas.dataset.pointCount = overlay ? String(overlay.bg.length + overlay.fg.length) : "0";
    canvas.dataset.labelCount = overlay ?
      String(overlay.extreme.text.length + overlay.selectedLabels.text.length) : "0";
    canvas.dataset.hoverVisible = overlay && overlay.hover ? "1" : "0";
    if (!overlay || (!overlay.bg.length && !overlay.fg.length && !overlay.hover &&
        !overlay.extreme.text.length && !overlay.selectedLabels.text.length)) return;
    var xa = gd._fullLayout.xaxis, ya = gd._fullLayout.yaxis;
    if (!xa || !ya) return;
    ctx.save();
    ctx.beginPath();
    ctx.rect(xa._offset, ya._offset, xa._length, ya._length);
    ctx.clip();
    ctx.globalAlpha = SELECTED_OPACITY;
    ctx.lineWidth = 0.8;
    ctx.strokeStyle = SELECTED_OUTLINE;

    if (overlay.bg.length) {
      ctx.beginPath();
      drawCirclePath(ctx, overlay.bg, data.bg, xa, ya);
      ctx.fillStyle = SELECTED_BG_COLOR;
      ctx.fill(); ctx.stroke();
    }

    // Quantize only the canvas paint colors, not the underlying Plotly values.
    // This reduces thousands of fill calls to at most 128 batches while keeping
    // the continuous colorscale visually indistinguishable at marker size.
    if (overlay.fg.length) {
      var buckets = new Array(128);
      var lo = data.cLo, hi = data.cHi;
      for (var i = 0; i < overlay.fg.length; i++) {
        var k = overlay.fg[i];
        var t = (isFinite(lo) && isFinite(hi) && hi > lo) ?
          (data.fg.colors[k] - lo) / (hi - lo) : 0.5;
        var b = Math.max(0, Math.min(127, Math.round(t * 127)));
        if (!buckets[b]) buckets[b] = [];
        buckets[b].push(k);
      }
      for (var bi = 0; bi < buckets.length; bi++) {
        if (!buckets[bi]) continue;
        ctx.beginPath();
        drawCirclePath(ctx, buckets[bi], data.fg, xa, ya);
        ctx.fillStyle = scaleColor(bi / 127);
        ctx.fill(); ctx.stroke();
      }
    }
    ctx.restore();

    // Dynamic annotations also live on this top canvas so their ordering is
    // deterministic: selected markers, then hover ring, then text labels.
    if (overlay.hover) {
      var hp = canvasPoint(overlay.hover.x, overlay.hover.y, xa, ya);
      if (isFinite(hp[0]) && isFinite(hp[1])) {
        ctx.beginPath();
        ctx.arc(hp[0], hp[1], 7, 0, Math.PI * 2);
        ctx.strokeStyle = HIGHLIGHT_COLOR;
        ctx.lineWidth = 3;
        ctx.globalAlpha = 1;
        ctx.stroke();
      }
    }
    drawCanvasLabels(ctx, overlay.extreme, xa, ya, "#0b0b0b", "right");
    drawCanvasLabels(ctx, overlay.selectedLabels, xa, ya, HIGHLIGHT_COLOR, "top");
  }

  function scheduleSelectionOverlay(panelIdx) {
    if (selectionDrawPending[panelIdx]) return;
    selectionDrawPending[panelIdx] = true;
    requestAnimationFrame(function () { drawSelectionOverlay(panelIdx); });
  }

  var lastFiltered = [];   // per panel: {fg: {...}, bg: {...}}
  for (p = 0; p < NPANELS; p++) lastFiltered.push(null);

  // Points failing the global filters are kept as a gray background layer so
  // they still anchor the axis ranges; only the per-panel range policies and
  // missing values can drop a point from the plot entirely.
  function buildPanelData(panelIdx) {
    var st = state.panels[panelIdx];
    var fg = { xs: [], ys: [], colors: [], idxs: [] };
    var bg = { xs: [], ys: [], idxs: [] };
    var fgPos = new Int32Array(NRECORDS);
    var bgPos = new Int32Array(NRECORDS);
    fgPos.fill(-1);
    bgPos.fill(-1);
    for (var i = 0; i < NRECORDS; i++) {
      var x = transformValue(COLUMNS[st.x][i], st.xScale, st.xMin, st.xMax, st.xClip);
      if (x === null) continue;
      var y = transformValue(COLUMNS[st.y][i], st.yScale, st.yMin, st.yMax, st.yClip);
      if (y === null) continue;
      var c = transformValue(COLUMNS[st.color][i], st.colorScale, st.colorMin, st.colorMax, st.colorClip);
      if (c === null) continue;
      if (!globalMask[i]) {
        bgPos[i] = bg.xs.length;
        bg.xs.push(x); bg.ys.push(y); bg.idxs.push(i);
        continue;
      }
      var cv = st.colorScale === "log" ? Math.log10(c)
        : st.colorScale === "quantile" ? PCT[st.color][i]
        : c;
      if (cv === null || cv === undefined) continue;
      fgPos[i] = fg.xs.length;
      fg.xs.push(x); fg.ys.push(y); fg.colors.push(cv); fg.idxs.push(i);
    }
    return { fg: fg, bg: bg, fgPos: fgPos, bgPos: bgPos };
  }

  // Histogram outline for a marginal panel. Bins are uniform in the displayed
  // space (log10 units on a log axis, so a log-scaled marginal is not a single
  // spike), and the result is returned as a step polyline in data space: two
  // points per bin, at its left and right edge. Drawing the outline rather
  // than a histogram/bar trace keeps the bars correct under the log transform,
  // where a bar's width would be interpreted in linear data units.
  function marginalOutline(values, isLog, binSpec, targetCount) {
    var out = { pos: [], cnt: [], binSpec: binSpec || null };
    var n = values.length;
    if (!n) return out;
    var i;
    // Transform once: both the extent scan and the binning need the displayed
    // value, and on a log axis that is a log10 per point.
    var t = values;
    if (isLog) {
      t = new Float64Array(n);
      for (i = 0; i < n; i++) t[i] = Math.log10(values[i]);
    }
    var lo, hi, nb, edges;
    if (binSpec) {
      lo = binSpec.lo; hi = binSpec.hi; nb = binSpec.nb; edges = binSpec.edges;
    } else {
      lo = Infinity; hi = -Infinity;
      for (i = 0; i < n; i++) {
        if (t[i] < lo) lo = t[i];
        if (t[i] > hi) hi = t[i];
      }
      if (!(hi > lo)) hi = lo + 1;   // all values identical
      nb = MARGINAL_BINS;
    }
    var w = (hi - lo) / nb;
    var counts = new Array(nb);
    for (i = 0; i < nb; i++) counts[i] = 0;
    for (i = 0; i < n; i++) {
      var b = Math.floor((t[i] - lo) / w);
      if (b < 0) b = 0;
      if (b >= nb) b = nb - 1;
      counts[b]++;
    }
    // Materialize the edges once so a bin's right edge is bit-identical to the
    // next bin's left edge; recomputing them would leave float-rounding gaps.
    if (!edges) {
      edges = new Array(nb + 1);
      for (i = 0; i <= nb; i++) {
        var e = lo + i * w;
        edges[i] = isLog ? Math.pow(10, e) : e;
      }
      out.binSpec = { lo: lo, hi: hi, nb: nb, edges: edges };
    }
    var countScale = targetCount === undefined ? 1 : targetCount / n;
    for (i = 0; i < nb; i++) {
      out.pos.push(edges[i], edges[i + 1]);
      out.cnt.push(counts[i] * countScale, counts[i] * countScale);
    }
    return out;
  }

  // Feature labels for the extreme colored points: the top LABELS_PER_AXIS by
  // x united with the top LABELS_PER_AXIS by y, so at most twice that many.
  // Background (filtered-out) points are never labelled.
  function topKIndices(values, k) {
    var top = [];
    for (var i = 0; i < values.length; i++) {
      var pos = top.length;
      while (pos > 0 && values[i] > values[top[pos - 1]]) pos--;
      if (pos >= k) continue;
      top.splice(pos, 0, i);
      if (top.length > k) top.pop();
    }
    return top;
  }

  function extremeLabels(fg) {
    var n = fg.xs.length;
    var out = { xs: [], ys: [], text: [] };
    if (!n) return out;
    var take = Math.min(LABELS_PER_AXIS, n);
    var byX = topKIndices(fg.xs, take);
    var byY = topKIndices(fg.ys, take);
    var chosen = new Uint8Array(n);
    for (var j = 0; j < take; j++) { chosen[byX[j]] = 1; chosen[byY[j]] = 1; }
    for (var k = 0; k < n; k++) {
      if (!chosen[k]) continue;
      out.xs.push(fg.xs[k]);
      out.ys.push(fg.ys[k]);
      out.text.push(FEATURES[fg.idxs[k]]);
    }
    return out;
  }

  function fmt(v) {
    if (Math.abs(v) !== 0 && (Math.abs(v) < 1e-3 || Math.abs(v) >= 1e5)) return v.toExponential(3);
    return (Math.round(v * 1000) / 1000).toString();
  }

  function fmtPct(p) {
    if (p === null || p === undefined || isNaN(p)) return "NA";
    return p.toFixed(3);
  }

  function axisTitle(varKey, scale) {
    return LABELS[varKey] + (scale === "log" ? " (log10 axis)" : "");
  }

  function colorbarTitle(varKey, scale) {
    return NAMES[varKey] +
      (scale === "log" ? "<br>(log10)" :
       scale === "quantile" ? "<br>(percentile)" : "");
  }

  // ---- hover popup ----
  var tooltipEl = document.getElementById("tooltip");

  // Feature names come from the input file and are written into innerHTML below.
  var ESC = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" };
  function esc(s) {
    return String(s).replace(/[&<>"]/g, function (c) { return ESC[c]; });
  }

  // A compact approximation of matplotlib's coolwarm map. Tooltip rows use
  // their empirical percentile, independently of the panel color variables.
  var TOOLTIP_COOLWARM = [
    [0.00, [59, 76, 192]],
    [0.25, [141, 176, 254]],
    [0.50, [221, 220, 220]],
    [0.75, [244, 152, 122]],
    [1.00, [180, 4, 38]]
  ];

  function tooltipRowStyle(p) {
    var rgb;
    if (p === null || p === undefined || !isFinite(p)) {
      rgb = [238, 238, 235];
    } else {
      p = Math.max(0, Math.min(1, p));
      var hi = 1;
      while (hi < TOOLTIP_COOLWARM.length - 1 && p > TOOLTIP_COOLWARM[hi][0]) hi++;
      var a = TOOLTIP_COOLWARM[hi - 1], b = TOOLTIP_COOLWARM[hi];
      var span = b[0] - a[0];
      var f = span > 0 ? (p - a[0]) / span : 0;
      rgb = [0, 1, 2].map(function (i) {
        return Math.round(a[1][i] + f * (b[1][i] - a[1][i]));
      });
    }
    // Choose whichever of white or near-black gives the greater WCAG contrast
    // against the interpolated sRGB background.
    var linear = rgb.map(function (c) {
      c /= 255;
      return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
    });
    var luminance = 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2];
    var foreground = luminance < 0.179 ? "#ffffff" : "#111111";
    return "background-color:rgb(" + rgb.join(",") + ");color:" + foreground;
  }

  function tooltipHTML(recIdx) {
    var rows = VARS.map(function (v) {
      var val = COLUMNS[v][recIdx];
      var pct = PCT[v][recIdx];
      return '<tr style="' + tooltipRowStyle(pct) + '"><td class="k">' + NAMES[v] + '</td><td>' +
        (val === null || val === undefined ? "NA" : fmt(val)) + '</td><td>' +
        fmtPct(pct) + '</td></tr>';
    }).join("");
    return '<span class="tt-feature">' + esc(FEATURES[recIdx]) + '</span>' +
      '<table><thead><tr><th class="k">variable</th><th>value</th><th>pctl</th></tr></thead>' +
      '<tbody>' + rows + '</tbody></table>';
  }

  function showTooltip(recIdx, mouseEv) {
    tooltipEl.innerHTML = tooltipHTML(recIdx);
    tooltipEl.style.display = "block";
    if (mouseEv) moveTooltip(mouseEv);
  }

  function moveTooltip(mouseEv) {
    var pad = 14;
    var w = tooltipEl.offsetWidth, h = tooltipEl.offsetHeight;
    var x = mouseEv.clientX + pad, y = mouseEv.clientY + pad;
    if (x + w > window.innerWidth - 4) x = Math.max(4, mouseEv.clientX - pad - w);
    if (y + h > window.innerHeight - 4) y = Math.max(4, window.innerHeight - 4 - h);
    tooltipEl.style.left = x + "px";
    tooltipEl.style.top = y + "px";
  }

  function hideTooltip() { tooltipEl.style.display = "none"; }

  function drawPanel(panelIdx) {
    var st = state.panels[panelIdx];
    var data = buildPanelData(panelIdx);
    lastFiltered[panelIdx] = data;
    hiddenUnselectedPanels[panelIdx] = false;

    // Trace order matters. Plotly puts the WebGL canvas *behind* the SVG layer,
    // and inside the WebGL layer it paints every natively-selected point after
    // the whole trace loop (scene.draw: `select2d.draw(selectBatch)` last), so
    // no scattergl trace can sit reliably on top. Selected markers, hover ring,
    // and dynamic labels therefore use the canvas overlay above Plotly:
    //   0 background (gl), 1 colored (gl), 2-3 full marginals (SVG),
    //   4-5 normalized selected-point marginals (SVG).
    // Color limits are retained for the canvas overlay so its selected markers
    // map values to the same colors as the WebGL point cloud.
    var cLo = null, cHi = null;
    for (var ci = 0; ci < data.fg.colors.length; ci++) {
      var cv = data.fg.colors[ci];
      if (cLo === null || cv < cLo) cLo = cv;
      if (cHi === null || cv > cHi) cHi = cv;
    }
    if (cLo === null || cHi <= cLo) { cLo = undefined; cHi = undefined; }
    data.cLo = cLo;
    data.cHi = cHi;

    var bgTrace = {
      x: data.bg.xs, y: data.bg.ys,
      mode: "markers",
      type: "scattergl",
      marker: { color: BACKGROUND_COLOR, size: 5, opacity: 0.35, line: { width: 0 } },
      unselected: { marker: { opacity: DIM_OPACITY } },
      hoverinfo: "none",
      customdata: data.bg.idxs,
      showlegend: false,
      name: ""
    };
    var mainTrace = {
      x: data.fg.xs, y: data.fg.ys,
      mode: "markers",
      type: "scattergl",
      marker: {
        color: data.fg.colors,
        colorscale: SEQ_COLORSCALE,
        cmin: cLo, cmax: cHi,
        showscale: true,
        // Horizontal colorbar below the scatter, ending where the scatter's x
        // domain ends and leaving room to its left for the title. It carries no
        // title of its own: Plotly reserves margin for a colorbar title, so a
        // long one would shrink the plot. The title is drawn as an annotation
        // instead (see below).
        colorbar: {
          orientation: "h", thickness: 12,
          len: CBAR_LEN, x: CBAR_X, xanchor: "left",
          y: CBAR_Y, yanchor: "top",
          tickfont: { size: 9 }
        },
        size: 6,
        opacity: 0.5,
        line: { width: 0 }
      },
      unselected: { marker: { opacity: DIM_OPACITY } },
      hoverinfo: "none",
      customdata: data.fg.idxs,
      name: ""
    };
    var labels = state.global.labels ? extremeLabels(data.fg) : { xs: [], ys: [], text: [] };
    // Marginals: the x histogram shares the scatter's x axis and counts on y2,
    // the y histogram shares its y axis and counts on x2. Both summarize the
    // colored (filter-passing) points only.
    var xHist = marginalOutline(data.fg.xs, st.xScale === "log");
    var yHist = marginalOutline(data.fg.ys, st.yScale === "log");
    data.xHist = xHist;
    data.yHist = yHist;
    var xHistTrace = {
      x: xHist.pos, y: xHist.cnt,
      xaxis: "x", yaxis: "y2",
      type: "scatter", mode: "lines",
      line: { width: 1, color: MARGINAL_LINE },
      fill: "tozeroy", fillcolor: MARGINAL_FILL,
      hoverinfo: "skip", showlegend: false, name: ""
    };
    var yHistTrace = {
      x: yHist.cnt, y: yHist.pos,
      xaxis: "x2", yaxis: "y",
      type: "scatter", mode: "lines",
      line: { width: 1, color: MARGINAL_LINE },
      fill: "tozerox", fillcolor: MARGINAL_FILL,
      hoverinfo: "skip", showlegend: false, name: ""
    };
    var selXHistTrace = {
      x: [], y: [], xaxis: "x", yaxis: "y2",
      type: "scatter", mode: "lines", zorder: 10,
      line: { width: 1, color: HIGHLIGHT_HALF },
      fill: "tozeroy", fillcolor: HIGHLIGHT_HALF,
      hoverinfo: "skip", showlegend: false, name: ""
    };
    var selYHistTrace = {
      x: [], y: [], xaxis: "x2", yaxis: "y",
      type: "scatter", mode: "lines", zorder: 10,
      line: { width: 1, color: HIGHLIGHT_HALF },
      fill: "tozerox", fillcolor: HIGHLIGHT_HALF,
      hoverinfo: "skip", showlegend: false, name: ""
    };
    var layout = {
      xaxis: {
        title: { text: axisTitle(st.x, st.xScale), font: { size: 11 } },
        type: st.xScale === "log" ? "log" : "linear",
        autorange: true, domain: SCATTER_DOMAIN, anchor: "y",
        gridcolor: "#e1e0d9", zerolinecolor: "#c3c2b7", tickfont: { size: 10 }
      },
      yaxis: {
        title: { text: axisTitle(st.y, st.yScale), font: { size: 11 } },
        type: st.yScale === "log" ? "log" : "linear",
        autorange: true, domain: SCATTER_DOMAIN, anchor: "x",
        gridcolor: "#e1e0d9", zerolinecolor: "#c3c2b7", tickfont: { size: 10 }
      },
      // Count axes for the marginals: bare, so they read as profiles.
      xaxis2: {
        domain: MARGINAL_DOMAIN, anchor: "y",
        showticklabels: false, showgrid: false, zeroline: false, ticks: ""
      },
      yaxis2: {
        domain: MARGINAL_DOMAIN, anchor: "x",
        showticklabels: false, showgrid: false, zeroline: false, ticks: ""
      },
      annotations: [{
        // Colorbar title, right-aligned just left of the bar and centred on
        // its thickness. Kept an annotation rather than colorbar.title because
        // Plotly reserves margin for the latter, which would shrink the plot.
        text: colorbarTitle(st.color, st.colorScale),
        xref: "paper", yref: "paper",
        x: CBAR_X, xanchor: "right", xshift: -8,
        y: CBAR_Y, yanchor: "middle", yshift: -15,
        showarrow: false, align: "right",
        font: { size: 11, color: "#52514e" }
      }],
      dragmode: st.dragmode,
      margin: { l: 55, r: 10, t: 24, b: 92 },
      plot_bgcolor: "#fcfcfb",
      paper_bgcolor: "#fcfcfb",
      showlegend: false,
      font: { family: "system-ui, -apple-system, Segoe UI, sans-serif", color: "#0b0b0b" }
    };

    var plotDone = Plotly.react(plotDivs[panelIdx],
      [bgTrace, mainTrace, xHistTrace, yHistTrace, selXHistTrace, selYHistTrace], layout,
      { responsive: true, displaylogo: false });
    selectionOverlay[panelIdx] = emptySelectionOverlay();
    selectionOverlay[panelIdx].extreme = labels;
    ensureSelectionCanvas(panelIdx);
    scheduleSelectionOverlay(panelIdx);
    if (plotDone && plotDone.then) {
      plotDone.then(function () {
        ensureSelectionCanvas(panelIdx);
        scheduleSelectionOverlay(panelIdx);
      });
    }
    afterDraw(panelIdx);
    // react() discards Plotly's selection, so the labels start empty again.
    selSig[panelIdx] = "0|";
    setSelStatus(panelIdx, 0, 0);
    return plotDone;
  }

  // ---- box/lasso selection labels ----
  var SEL_X_HIST_TRACE = 4;
  var SEL_Y_HIST_TRACE = 5;
  var MIN_SELECTION_MARGINAL_POINTS = 50;
  var MAX_SELECTION_MARGINAL_FRACTION = 0.85;
  var HIDE_LINKED_UNSELECTED_FRACTION = 0.5;

  function updateExtremeLabels(panelIdx) {
    var data = lastFiltered[panelIdx];
    var gd = plotDivs[panelIdx];
    if (!data || !gd) return;
    var labels = state.global.labels ? extremeLabels(data.fg) : { xs: [], ys: [], text: [] };
    selectionOverlay[panelIdx].extreme = labels;
    scheduleSelectionOverlay(panelIdx);
  }

  // shown: number of names drawn, or null when annotation is off.
  function setSelStatus(panelIdx, total, shown) {
    var el = selStatusEls[panelIdx];
    if (!el) return;
    if (!total) {
      el.textContent = "no selection";
      el.classList.remove("active");
      return;
    }
    el.textContent = total + " selected" +
      (shown !== null && shown < total ? " · labelling " + shown : "");
    el.classList.add("active");
  }

  // Per panel: pending sync timer, a guard while our own restyle is in flight,
  // a request that arrived during that restyle, and a signature of the labels
  // last applied.
  var selPending = [], selApplying = [], selDirty = [], selSig = [];
  for (p = 0; p < NPANELS; p++) {
    selPending.push(null); selApplying.push(false); selDirty.push(false); selSig.push("");
  }

  // Selection state is read back from the traces rather than from the event
  // payload: Plotly records the hit points on each trace as `selectedpoints`,
  // which is authoritative for box and lasso alike and survives events that
  // arrive with no (or partial) point list.
  function selectedIndices(gd, curve) {
    var td = (gd._fullData && gd._fullData[curve]) || (gd.data && gd.data[curve]);
    var sp = td && td.selectedpoints;
    return (sp && sp.length) ? sp : null;
  }

  // Restyling from inside a plotly_selected handler races Plotly's own
  // teardown of the drag, which is what made selections work only sometimes.
  // The sync is deferred to the next tick, coalesced, and skipped when the
  // labels would not change.
  function scheduleSelectionSync(panelIdx) {
    // Busy applying an earlier sync: record the request rather than dropping it,
    // so a selection made inside that window still lands.
    if (selApplying[panelIdx]) { selDirty[panelIdx] = true; return; }
    // Already queued: that run reads live state, so it covers this request too.
    if (selPending[panelIdx]) return;
    selPending[panelIdx] = setTimeout(function () {
      selPending[panelIdx] = null;
      syncSelectionLabels(panelIdx);
    }, 0);
  }

  // At most one selection is active at a time. focalPanel is the panel it was
  // drawn in — it keeps Plotly's own selection styling and gets the labels —
  // and focalRecords is the set of selected records, used to carry the same
  // selection into the other panels.
  var focalPanel = null;
  var focalRecords = null;
  var focalShowMarginals = false;
  var focalHideLinkedUnselected = false;
  var hiddenUnselectedPanels = new Array(NPANELS);
  for (p = 0; p < NPANELS; p++) hiddenUnselectedPanels[p] = false;

  function syncSelectionLabels(panelIdx) {
    var gd = plotDivs[panelIdx];
    var data = lastFiltered[panelIdx];
    if (!gd || !data) return;
    // Point indices are per trace, and a record sits in exactly one of the two,
    // so the cached panel data resolves them without any deduplication.
    var sources = [
      { sp: selectedIndices(gd, 0), src: data.bg },
      { sp: selectedIndices(gd, 1), src: data.fg }
    ];
    var xs = [], ys = [], text = [], total = 0, recs = {};
    for (var s = 0; s < sources.length; s++) {
      var sp = sources[s].sp, src = sources[s].src;
      if (!sp) continue;
      for (var i = 0; i < sp.length; i++) {
        var k = sp[i];
        if (k === null || k === undefined || k >= src.xs.length) continue;
        total++;
        recs[src.idxs[k]] = true;          // every selected record, uncapped
        if (text.length >= MAX_SELECTION_LABELS) continue;
        xs.push(src.xs[k]); ys.push(src.ys[k]); text.push(FEATURES[src.idxs[k]]);
      }
    }

    if (!total) {
      // Only the panel that owns the selection may tear the shared state down;
      // a cross-dimmed panel carries the focal panel's points, not its own.
      if (focalPanel === panelIdx) clearAllSelections();
      return;
    }

    var annotate = state.global.annotateSelection;
    var sig = (annotate ? "1|" : "0|") + total + "|" + text.join(",");
    if (selSig[panelIdx] === sig && focalPanel === panelIdx) return;
    if (focalPanel !== null && focalPanel !== panelIdx) clearPanelSelection(focalPanel);
    focalPanel = panelIdx;
    focalRecords = recs;
    var selectableCount = data.bg.xs.length + data.fg.xs.length;
    focalShowMarginals = total >= MIN_SELECTION_MARGINAL_POINTS &&
      total < MAX_SELECTION_MARGINAL_FRACTION * selectableCount;
    focalHideLinkedUnselected = total >
      HIDE_LINKED_UNSELECTED_FRACTION * selectableCount;
    if (focalHideLinkedUnselected && lastHoverIdx !== null && !recs[lastHoverIdx]) {
      onUnhover();
    }
    selSig[panelIdx] = sig;

    selApplying[panelIdx] = true;
    var done = function () {
      selApplying[panelIdx] = false;
      if (selDirty[panelIdx]) { selDirty[panelIdx] = false; scheduleSelectionSync(panelIdx); }
    };
    try {
      selectionOverlay[panelIdx].selectedLabels = {
        xs: annotate ? xs : [], ys: annotate ? ys : [], text: annotate ? text : []
      };
      scheduleSelectionOverlay(panelIdx);
      Promise.resolve().then(function () {
        var links = [];
        for (var pi = 0; pi < NPANELS; pi++) {
          links.push(linkPanel(pi, recs, pi === panelIdx, focalShowMarginals,
                               focalHideLinkedUnselected));
        }
        return Promise.all(links);
      }).then(done, done);
    } catch (e) {
      done();   // never leave the panel stuck with selApplying set
    }
    setSelStatus(panelIdx, total, annotate ? text.length : null);
  }

  // Carry the selection into a panel. Normally the selected points are redrawn
  // on the top canvas (bigger, opaque and outlined). When more than half the
  // focal panel is selected, linked panels instead make their unselected native
  // markers transparent and leave selected markers unchanged; this avoids
  // repainting a large foreground overlay while retaining all trace data and
  // index mappings. Names are focal-only, and its selection outline is kept.
  function updateSelectionMarginals(pi, fgSel, showMarginals) {
    var gd = plotDivs[pi], data = lastFiltered[pi], st = state.panels[pi];
    var x = [[], []], y = [[], []];
    if (showMarginals && fgSel.length && data.xHist.binSpec && data.yHist.binSpec) {
      var selectedX = new Array(fgSel.length), selectedY = new Array(fgSel.length);
      for (var i = 0; i < fgSel.length; i++) {
        selectedX[i] = data.fg.xs[fgSel[i]];
        selectedY[i] = data.fg.ys[fgSel[i]];
      }
      var xHist = marginalOutline(selectedX, st.xScale === "log",
                                  data.xHist.binSpec, data.fg.xs.length);
      var yHist = marginalOutline(selectedY, st.yScale === "log",
                                  data.yHist.binSpec, data.fg.ys.length);
      x = [xHist.pos, yHist.cnt];
      y = [xHist.cnt, yHist.pos];
    }
    return Plotly.restyle(gd, { x: x, y: y }, [SEL_X_HIST_TRACE, SEL_Y_HIST_TRACE]);
  }

  function linkPanel(pi, recs, isFocal, showMarginals, hideLinkedUnselected) {
    var gd = plotDivs[pi], data = lastFiltered[pi];
    if (!gd || !data) return Promise.resolve();
    if (!recs) {
      hiddenUnselectedPanels[pi] = false;
      var clearNative = Plotly.restyle(gd, {
        selectedpoints: null,
        "unselected.marker.opacity": DIM_OPACITY
      }, [0, 1]);
      return clearNative.then(function () {
        selectionOverlay[pi].bg = [];
        selectionOverlay[pi].fg = [];
        scheduleSelectionOverlay(pi);
        return updateSelectionMarginals(pi, [], false);
      });
    }
    // selectedpoints covers the whole selection, so dimming/hiding is exact.
    var bgSel = [], fgSel = [];
    var recIds = Object.keys(recs);
    for (var i = 0; i < recIds.length; i++) {
      var recIdx = +recIds[i];
      var fk = data.fgPos[recIdx];
      if (fk >= 0) {
        fgSel.push(fk);
      }
      var bk = data.bgPos[recIdx];
      if (bk >= 0) {
        bgSel.push(bk);
      }
    }
    var hideUnselected = !isFocal && hideLinkedUnselected;
    hiddenUnselectedPanels[pi] = hideUnselected;
    var nativeUpdate = isFocal ?
      Plotly.restyle(gd, { "unselected.marker.opacity": DIM_OPACITY }, [0, 1]) :
      Plotly.restyle(gd, {
        selectedpoints: [bgSel, fgSel],
        "unselected.marker.opacity": hideUnselected ? 0 : DIM_OPACITY
      }, [0, 1]);
    return nativeUpdate.then(function () {
      // Native selected markers are already unobstructed when all other points
      // are transparent, so the large canvas copy is unnecessary in that mode.
      selectionOverlay[pi].bg = hideUnselected ? [] : bgSel;
      selectionOverlay[pi].fg = hideUnselected ? [] : fgSel;
      scheduleSelectionOverlay(pi);
      return updateSelectionMarginals(pi, fgSel, showMarginals);
    });
  }

  // A double-click clears a selection. If it lands on the focal panel the whole
  // linked selection goes away; on any other panel Plotly may have dropped the
  // borrowed selectedpoints, so restore that panel's dimming.
  function onDeselect(panelIdx) {
    if (focalPanel === panelIdx) clearAllSelections();
    else if (focalRecords) linkPanel(panelIdx, focalRecords, false, focalShowMarginals,
                                     focalHideLinkedUnselected);
  }

  // Drop one panel's labels, status and selection outline, leaving the shared
  // focal/cross-dim state alone.
  function clearPanelSelection(pi) {
    var gd = plotDivs[pi];
    if (!gd) return;
    selSig[pi] = "0|";
    selectionOverlay[pi].selectedLabels = { xs: [], ys: [], text: [] };
    scheduleSelectionOverlay(pi);
    Plotly.relayout(gd, { selections: [] });
    setSelStatus(pi, 0, 0);
  }

  // Since a selection is shared across panels, clearing is global: every
  // "clear selection" button undoes the one active selection everywhere.
  function clearAllSelections() {
    focalPanel = null;
    focalRecords = null;
    focalShowMarginals = false;
    focalHideLinkedUnselected = false;
    for (var pi = 0; pi < NPANELS; pi++) {
      clearPanelSelection(pi);
      linkPanel(pi, null, true, false, false); // restore points; clear overlays
    }
  }

  // Freeze the autoranged extents once the panel is drawn, preserving the
  // established stable-axis behavior across style-only Plotly updates. Each
  // redraw passes autorange:true again, so new data still rescales the axes.
  // Idempotent, so the resize handler can reuse it.
  function afterDraw(panelIdx) {
    var gd = plotDivs[panelIdx];
    var fl = gd._fullLayout;
    if (!fl) return;
    var upd = {};
    ["xaxis", "yaxis"].forEach(function (ax) {
      if (fl[ax] && fl[ax].autorange && fl[ax].range) {
        upd[ax + ".range"] = fl[ax].range.slice();
        upd[ax + ".autorange"] = false;
      }
    });
    if (Object.keys(upd).length) Plotly.relayout(gd, upd);
  }

  window.addEventListener("resize", function () {
    for (var pr = 0; pr < NPANELS; pr++) {
      afterDraw(pr);
      scheduleSelectionOverlay(pr);
    }
  });

  function updateMatchCount() {
    var drawn = lastFiltered.map(function (d) {
      return d ? d.fg.xs.length + d.bg.xs.length : 0;
    });
    document.getElementById("match-count").textContent =
      globalPassCount + " of " + NRECORDS + " features pass filters · drawn per panel: " +
      drawn.join(" · ");
  }

  // Plotly re-fires hover whenever the closest point changes, which in a dense
  // cloud is constant. Repositioning the popup is cheap; rebuilding it and
  // repainting the ring in every panel is not, so that work is skipped while
  // the hovered record stays the same.
  var lastHoverIdx = null;

  function onHover(idx, mouseEv) {
    if (idx === lastHoverIdx) {
      if (mouseEv) moveTooltip(mouseEv);
      return;
    }
    lastHoverIdx = idx;
    for (var p2 = 0; p2 < NPANELS; p2++) {
      var st = state.panels[p2];
      var x = transformValue(COLUMNS[st.x][idx], st.xScale, st.xMin, st.xMax, true);
      var y = transformValue(COLUMNS[st.y][idx], st.yScale, st.yMin, st.yMax, true);
      if (x === null || y === null) {
        selectionOverlay[p2].hover = null;
      } else {
        selectionOverlay[p2].hover = { x: x, y: y };
      }
      scheduleSelectionOverlay(p2);
    }
    showHoverInfo(idx);
    showTooltip(idx, mouseEv);
  }

  function onUnhover() {
    lastHoverIdx = null;
    for (var p2 = 0; p2 < NPANELS; p2++) {
      selectionOverlay[p2].hover = null;
      scheduleSelectionOverlay(p2);
    }
    hideTooltip();
  }

  function showHoverInfo(recIdx) {
    var el = document.getElementById("hover-info");
    var parts = ['<span class="hi-feature">' + esc(FEATURES[recIdx]) + '</span>'];
    VARS.forEach(function (v) {
      var val = COLUMNS[v][recIdx];
      parts.push('<span class="hi-item"><span class="k">' + NAMES[v] + ':</span> ' +
        (val === null || val === undefined ? "NA" : fmt(val)) + '</span>');
    });
    el.innerHTML = parts.join("");
  }

  var INPUT_DEBOUNCE_MS = 120;
  var allRedrawTimer = null;
  var panelRedrawTimers = new Array(NPANELS);

  function cancelPanelRedraws() {
    for (var i = 0; i < NPANELS; i++) {
      if (panelRedrawTimers[i]) clearTimeout(panelRedrawTimers[i]);
      panelRedrawTimers[i] = null;
    }
  }

  function redrawAll() {
    cancelPanelRedraws();
    // react() drops selections and selectedpoints in every panel, so the
    // linked-selection state goes with them. The hover ring goes too, so the
    // "same record" guard has to forget what was last hovered.
    focalPanel = null;
    focalRecords = null;
    focalShowMarginals = false;
    focalHideLinkedUnselected = false;
    lastHoverIdx = null;
    computeGlobalMask();
    for (var p3 = 0; p3 < NPANELS; p3++) drawPanel(p3);
    updateMatchCount();
  }

  function redrawPanel(panelIdx) {
    // A focal-panel change invalidates the box/lasso geometry, so retain the
    // established full redraw and selection clear there. A non-focal panel has
    // no selection geometry of its own: rebuild just that panel, then remap the
    // persistent selected record IDs into its new traces.
    if (focalPanel !== null || focalRecords) {
      if (panelIdx === focalPanel || !focalRecords) {
        redrawAll();
        return;
      }
      if (lastHoverIdx !== null) onUnhover();
      lastHoverIdx = null;
      var activeFocal = focalPanel;
      var activeRecords = focalRecords;
      var showMarginals = focalShowMarginals;
      var hideUnselected = focalHideLinkedUnselected;
      var plotDone = drawPanel(panelIdx);
      updateMatchCount();
      // A rapid new selection or clear may finish before react(); only relink
      // when this is still the selection that requested the panel redraw.
      return Promise.resolve(plotDone).then(function () {
        if (focalPanel !== activeFocal || focalRecords !== activeRecords) return;
        return linkPanel(panelIdx, activeRecords, false, showMarginals, hideUnselected);
      }).then(function () {}, function () {});
    }
    if (lastHoverIdx !== null) onUnhover();
    lastHoverIdx = null;
    drawPanel(panelIdx);
    updateMatchCount();
  }

  function redrawPanelNow(panelIdx) {
    if (allRedrawTimer) {
      clearTimeout(allRedrawTimer);
      allRedrawTimer = null;
      redrawAll();
      return;
    }
    if (panelRedrawTimers[panelIdx]) clearTimeout(panelRedrawTimers[panelIdx]);
    panelRedrawTimers[panelIdx] = null;
    redrawPanel(panelIdx);
  }

  function schedulePanelRedraw(panelIdx) {
    if (allRedrawTimer) return;
    if (panelRedrawTimers[panelIdx]) clearTimeout(panelRedrawTimers[panelIdx]);
    panelRedrawTimers[panelIdx] = setTimeout(function () {
      panelRedrawTimers[panelIdx] = null;
      redrawPanel(panelIdx);
    }, INPUT_DEBOUNCE_MS);
  }

  function redrawAllNow() {
    if (allRedrawTimer) clearTimeout(allRedrawTimer);
    allRedrawTimer = null;
    cancelPanelRedraws();
    redrawAll();
  }

  function scheduleAllRedraw() {
    cancelPanelRedraws();
    if (allRedrawTimer) clearTimeout(allRedrawTimer);
    allRedrawTimer = setTimeout(function () {
      allRedrawTimer = null;
      redrawAll();
    }, INPUT_DEBOUNCE_MS);
  }

  // Browsers restore form controls across a reload, so the markup's defaults are
  // not necessarily what the user is looking at. Seed the state from the DOM
  // rather than assuming, otherwise a restored checkbox reads one way and
  // behaves the other.
  function syncStateFromControls() {
    document.querySelectorAll("#global-filters .gf-group[data-gf]").forEach(function (grp) {
      var v = grp.getAttribute("data-gf");
      if (GLOBAL_VARS.indexOf(v) === -1) return;
      var minV = grp.querySelector('input[data-role="min"]').value;
      var maxV = grp.querySelector('input[data-role="max"]').value;
      state.global[v] = {
        min: minV === "" ? null : parseFloat(minV),
        max: maxV === "" ? null : parseFloat(maxV)
      };
    });
    state.global.labels = labelToggle.checked;
    state.global.annotateSelection = annotateToggle.checked;
    state.global.muteBackgroundHover = muteBackgroundToggle.checked;
    applyRegexInput();
  }

  syncStateFromControls();
  redrawAll();

  // Hover listeners are attached once; Plotly.react() reuses the same div
  // element on every redraw, so re-attaching here would stack duplicate
  // handlers on each filter/control change.
  for (var p4 = 0; p4 < NPANELS; p4++) {
    (function (panelIdx) {
      plotDivs[panelIdx].on("plotly_hover", function (ev) {
        if (!ev.points || !ev.points.length) return;
        // Prefer a colored point over a background point under the cursor.
        var pt = null;
        for (var pi = 0; pi < ev.points.length; pi++) {
          var cand = ev.points[pi];
          if (cand.curveNumber > 1) continue;      // marginal traces
          // Invisible points are never hover targets. When requested, apply
          // the same exclusion to visible-but-unselected background points.
          if (focalRecords && !focalRecords[cand.customdata] &&
              (state.global.muteBackgroundHover || hiddenUnselectedPanels[panelIdx])) continue;
          if (!pt || cand.curveNumber === 1) pt = cand;
        }
        if (!pt) { onUnhover(); return; }
        onHover(pt.customdata, ev.event);
      });
      plotDivs[panelIdx].on("plotly_unhover", function () { onUnhover(); });
      // A new selection re-reads the traces' selected points; a deselect is
      // handled separately so it can never take focus. Both are deferred off
      // the event, like the sync itself.
      plotDivs[panelIdx].on("plotly_selected", function () { scheduleSelectionSync(panelIdx); });
      plotDivs[panelIdx].on("plotly_deselect", function () {
        setTimeout(function () { onDeselect(panelIdx); }, 0);
      });
      // Remember the modebar's drag tool so a redraw does not revert it to zoom.
      plotDivs[panelIdx].on("plotly_relayout", function (ev) {
        if (ev && ev.dragmode) state.panels[panelIdx].dragmode = ev.dragmode;
        scheduleSelectionOverlay(panelIdx);
      });
    })(p4);
  }
})();
</script>
</body>
</html>
"""


def render_html(ordered_vars, records, title, subtitle):
    labels = {v: VAR_LABELS.get(v, v) for v in ordered_vars}
    names = {v: VAR_NAMES.get(v, v) for v in ordered_vars}
    global_vars = [v for v in GLOBAL_FILTER_VARS if v in ordered_vars]
    log_default_vars = [v for v in LOG_SCALE_DEFAULT_VARS if v in ordered_vars]
    default_ranges = {v: r for v, r in VAR_DEFAULT_RANGES.items() if v in ordered_vars}
    var_meta = {"vars": ordered_vars, "labels": labels, "names": names,
                "globalVars": global_vars, "logDefaultVars": log_default_vars,
                "defaultRanges": default_ranges, "labelsPerAxis": LABELS_PER_AXIS,
                "maxSelectionLabels": MAX_SELECTION_LABELS,
                "marginalBins": MARGINAL_BINS,
                "panelDefaults": panel_defaults(ordered_vars)}

    # Column-oriented data avoids repeating every variable name once per row in
    # the generated HTML and creates far fewer JavaScript objects at startup.
    data = {
        "features": [r["Feature"] for r in records],
        "columns": {v: [r[v] for r in records] for v in ordered_vars},
    }

    def dumps(obj):
        return json.dumps(obj, allow_nan=False, separators=(",", ":")).replace("</", "<\\/")

    def hidden(v):
        return "" if v in ordered_vars else "display:none"

    # Titles are escaped: they carry the input filename and --title verbatim.
    tokens = {
        "__TITLE__": html_escape(title),
        "__SUBTITLE__": html_escape(subtitle),
        "__DATA_JSON__": dumps(data),
        "__VARMETA_JSON__": dumps(var_meta),
        "__TOTCOUNT_DISPLAY__": hidden("totCount"),
        "__NUNITS_DISPLAY__": hidden("nUnits"),
        "__LOG2GAIN_DISPLAY__": hidden("log2Gain"),
    }
    # One pass, so no substituted value can be scanned for further tokens (a
    # feature name or a title containing "__DATA_JSON__" is just text).
    return re.sub(r"__[A-Z0-9][A-Z0-9_]*__",
                  lambda m: tokens.get(m.group(0), m.group(0)),
                  HTML_TEMPLATE)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", "-i", required=True, help="*.feature_residuals.tsv path")
    ap.add_argument("--output", "-o", default=None,
                    help="output HTML path (default: input path with .tsv replaced by .html)")
    ap.add_argument("--title", default=None, help="Title shown in the page header")
    ap.add_argument("--seed", type=int, default=0,
                    help="RNG seed for the plotting-order shuffle (default 0)")
    args = ap.parse_args()

    if args.output is None:
        root, ext = os.path.splitext(args.input)
        args.output = (root if ext.lower() == ".tsv" else args.input) + ".html"

    ordered_vars, records = load_records(args.input)
    if not records:
        sys.exit("No data rows parsed from " + args.input)

    # Shuffle so that no group of features is systematically drawn on top of
    # the others; plotting order follows this record order in the browser.
    random.Random(args.seed).shuffle(records)

    title = args.title or ("Feature diagnostics: " + os.path.basename(args.input))
    subtitle = ("{n} features · {m} variables · source: {src}"
                .format(n=len(records), m=len(ordered_vars), src=os.path.basename(args.input)))

    html = render_html(ordered_vars, records, title, subtitle)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        fh.write(html)
    print("Wrote {} ({} features, variables: {})".format(
        args.output, len(records), ", ".join(ordered_vars)))


if __name__ == "__main__":
    main()
