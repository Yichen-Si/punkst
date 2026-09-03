#!/usr/bin/env python3
"""Build a temporary, development-only hierarchy diagnostic HTML atlas.

This viewer consumes production artifacts, but its HTML and UI are not a
supported output contract.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from multires_diagnostics.build_multires_level0_html import (
    build_payload as build_level0_payload,
)

from multires_diagnostics.multires_report_common import (
    encode_array,
    json_for_html,
    load_factor_annotations,
    load_graph_context,
    load_metadata,
    load_theta,
    metadata_arguments,
    numeric_columns,
    read_tsv,
    require_artifact,
    resolve_pipeline,
)


PLOTLY_URL = "https://cdn.plot.ly/plotly-2.35.2.min.js"


def _categorical(values: list[str]) -> dict:
    names = sorted(set(values), key=lambda value: (
        0, int(value[1:])) if value.startswith("C") and value[1:].isdigit()
        else (1, value))
    positions = {name: index for index, name in enumerate(names)}
    return {"names": names, "values": [positions[value] for value in values]}


def _axis_details(path: Path,
                  factor_names: list[str]) -> tuple[list[str], np.ndarray]:
    header, rows = read_tsv(path)
    required = {"axis", "factor", "coefficient"}
    if not required.issubset(header):
        raise ValueError(f"Scene axis table has an invalid header: {path}")
    grouped: dict[int, list[tuple[float, float, str]]] = {}
    for row in rows:
        axis = int(row["axis"])
        coefficient = float(row["coefficient"])
        signed_weight = (
            float(row["positive_weight"]) - float(row["negative_weight"])
            if {"positive_weight", "negative_weight"}.issubset(header)
            else coefficient)
        grouped.setdefault(axis, []).append(
            (coefficient, signed_weight, row["factor"]))
    if sorted(grouped) != list(range(len(grouped))):
        raise ValueError(f"Scene axes are not consecutive: {path}")
    summaries = []
    factor_positions = {factor: index
                        for index, factor in enumerate(factor_names)}
    loadings = np.zeros((len(grouped), len(factor_names)), dtype=np.float32)
    for axis in range(len(grouped)):
        strongest = sorted(
            grouped[axis], key=lambda value: (-abs(value[0]), value[2]))[:4]
        terms = []
        for coefficient, _, factor in strongest:
            terms.append(f"{coefficient:+.3g} F{factor}")
        for _, signed_weight, factor in grouped[axis]:
            if factor in factor_positions:
                loadings[axis, factor_positions[factor]] = signed_weight
        summaries.append(" · ".join(terms))
    return summaries, loadings


def _load_view(root: Path, record: dict, name: str,
               id_positions: dict[str, int],
               factor_names: list[str]) -> dict:
    view = record[name]
    if not view.get("available"):
        return {"available": False, "reason": view.get("omitted_reason")}
    coordinate_path = root / view["coordinates"]
    header, rows = read_tsv(coordinate_path)
    axes = numeric_columns(header)
    if len(axes) != view["dimensions"] or "id" not in header:
        raise ValueError(f"Scene {name} coordinate schema does not match manifest")
    indices = []
    coordinates = np.empty((len(rows), len(axes)), dtype=np.float32)
    cores = []
    for index, row in enumerate(rows):
        if row["id"] not in id_positions:
            raise ValueError(f"Scene {name} contains an unknown member")
        indices.append(id_positions[row["id"]])
        cores.append(int(row["core"]))
        coordinates[index] = [float(row[column]) for column in axes]
    if len(set(indices)) != len(indices) or not np.isfinite(coordinates).all():
        raise ValueError(f"Scene {name} coordinates are invalid")
    if name == "diffusion":
        selected = view.get("selected_modes", [])
        alternate = view.get("alternate_modes", [])
        if not isinstance(alternate, list) or len(alternate) > 20:
            raise ValueError("Scene diffusion alternate modes are invalid")
        alternate_path = root / view["alternate_coordinates"]
        alternate_header, alternate_rows = read_tsv(alternate_path)
        alternate_columns = [f"mode_{mode}" for mode in alternate]
        if (alternate_header != ["id", *alternate_columns]
                or [row["id"] for row in alternate_rows]
                != [row["id"] for row in rows]):
            raise ValueError(
                "Scene diffusion alternate coordinates do not align")
        if alternate:
            extra = np.asarray([
                [float(row[column]) for column in alternate_columns]
                for row in alternate_rows], dtype=np.float32)
            if not np.isfinite(extra).all():
                raise ValueError("Scene diffusion alternates are non-finite")
            coordinates = np.column_stack((coordinates, extra))
        display_modes = [*selected, *alternate]
        summaries = [f"global diffusion mode {mode}"
                     for mode in display_modes]
        _, mode_rows = read_tsv(root / view["modes"])
        spectrum = []
        for row in mode_rows:
            residual = row["regression_residual"]
            spectrum.append({
                "mode": int(row["mode"]),
                "variance_fraction": float(row["variance_fraction"]),
                "regression_residual": (
                    float(residual) if residual != "." else None),
                "importance_rank": (
                    int(row["importance_rank"])
                    if row["importance_rank"] != "." else None),
                "selected": row["selected"] == "1",
            })
        loadings = None
    else:
        summaries, loadings = _axis_details(
            root / view["axes"], factor_names)
        spectrum = None
    output = {
        "available": True,
        "dimensions": len(axes),
        "rows": len(rows),
        "indices": indices,
        "core": cores,
        "coordinates_f32": encode_array(coordinates),
        "axis_summaries": summaries,
        "fit_rows": view.get("fit_rows"),
        "represented_groups": view.get("represented_groups"),
        "retained_subspace_variance_fraction":
            view.get("retained_subspace_variance_fraction"),
        "quartimax_objective": view.get("quartimax_objective"),
    }
    if loadings is not None:
        output["loadings_f32"] = encode_array(loadings)
    if spectrum is not None:
        output["spectrum"] = spectrum
        output["selected_modes"] = view.get("selected_modes", [])
        output["selected_dimensions"] = len(selected)
        output["display_modes"] = display_modes
        output["dimensions"] = len(display_modes)
        output["diffusion_time"] = view.get("diffusion_time")
        output["target_effective_rank"] = view.get("target_effective_rank")
        output["achieved_effective_rank"] = view.get(
            "achieved_effective_rank")
        output["basis_limited"] = view.get("basis_limited")
    return output


def build_payload(pipeline_path: Path, metadata_path: Path | None,
                  metadata_id_column: str | None,
                  cell_type_column: str | None,
                  umap_x_column: str | None,
                  umap_y_column: str | None,
                  de_path: Path | None) -> dict:
    _, pipeline, stages = resolve_pipeline(pipeline_path)
    required = {"graph", "scenes", "embeddings"}
    if not required.issubset(stages):
        raise ValueError(
            "Pipeline must be complete through the embeddings stage")
    graph = load_graph_context(stages["graph"])
    scenes_path, scenes_manifest = require_artifact(
        stages["scenes"], "punkst.multires.scenes")
    embeddings_path, embeddings = require_artifact(
        stages["embeddings"], "punkst.multires.embeddings")
    if (embeddings.get("source", {}).get("scenes_fingerprint")
            != scenes_manifest.get("fingerprint")):
        raise ValueError("Scene embeddings do not match the scene artifact")
    factors = graph["factor_names"]
    factor_annotations = load_factor_annotations(de_path, factors)
    loading_labels = []
    for factor in factors:
        annotation = factor_annotations.get(factor)
        prefix = f"F{factor}: "
        if annotation and annotation.startswith(prefix):
            genes = ",".join(
                value.strip()
                for value in annotation[len(prefix):].split(",")
                if value.strip())
            loading_labels.append(f"F{factor} - {genes}")
        else:
            loading_labels.append(f"F{factor}")
    metadata = load_metadata(
        metadata_path, id_column=metadata_id_column,
        cell_type_column=cell_type_column,
        umap_x_column=umap_x_column, umap_y_column=umap_y_column)
    global_payload = build_level0_payload(
        pipeline_path, metadata_path, metadata_id_column,
        cell_type_column, umap_x_column, umap_y_column, de_path,
        "partition")
    if global_payload["factor_names"] != factors:
        raise ValueError("Level-0 and scene factor sets differ")

    _, membership_rows = read_tsv(
        scenes_path.parent / scenes_manifest["tables"]["memberships"])
    memberships: dict[tuple[int, int], list[dict[str, str]]] = {}
    for row in membership_rows:
        memberships.setdefault(
            (int(row["level"]), int(row["scene"])), []).append(row)
    assignments: dict[int, dict[str, str]] = {}
    for level in scenes_manifest["levels"]:
        _, rows = read_tsv(scenes_path.parent / level["assignment_table"])
        assignments[int(level["level"])] = {
            row["id"]: row["partition_cluster"] for row in rows}
    _, node_rows = read_tsv(
        scenes_path.parent / scenes_manifest["tables"]["nodes"])
    _, edge_rows = read_tsv(
        scenes_path.parent / scenes_manifest["tables"]["edges"])
    edges = [{key: int(row[key]) for key in ("parent", "child")}
             for row in edge_rows]

    records = {
        (int(row["level"]), int(row["scene"])): row
        for row in embeddings.get("scenes", [])}
    if set(records) != set(memberships):
        raise ValueError("Embedding and membership scene sets differ")
    all_identifiers = list(dict.fromkeys(
        row["id"] for key in sorted(memberships)
        for row in memberships[key]))
    external_palette = _categorical([
        metadata.get(identifier, {}).get("cell_type", "Unannotated")
        for identifier in dict.fromkeys(
            [*global_payload["identifiers"], *all_identifiers])
    ])["names"] if metadata else []
    external_colors = {
        label: index for index, label in enumerate(external_palette)}
    global_external = global_payload["colorings"].get("cell-type")
    if global_external is not None:
        global_external["color_indices"] = [
            external_colors[label] for label in global_external["names"]]
    all_theta = load_theta(graph, all_identifiers)
    theta_by_id = {
        identifier: all_theta[index]
        for index, identifier in enumerate(all_identifiers)
    }
    payload_scenes = []
    for key in sorted(records):
        level, scene = key
        record = records[key]
        members = memberships[key]
        identifiers = [row["id"] for row in members]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError(f"Scene {level}/{scene} repeats a member")
        positions = {identifier: index
                     for index, identifier in enumerate(identifiers)}
        theta = np.asarray(
            [theta_by_id[identifier] for identifier in identifiers],
            dtype=np.float32)
        categories = {
            "membership": _categorical([
                "Core" if row["core"] == "1" else "Halo"
                for row in members]),
        }
        next_assignment = assignments.get(level + 1)
        if next_assignment is not None:
            categories["child-partition"] = _categorical([
                f"C{next_assignment[identifier]}"
                for identifier in identifiers])
        if metadata and any("cell_type" in metadata.get(identifier, {})
                            for identifier in identifiers):
            cell_types = _categorical([
                metadata.get(identifier, {}).get("cell_type", "Unannotated")
                for identifier in identifiers])
            cell_types["color_indices"] = [
                external_colors[label] for label in cell_types["names"]]
            categories["cell-type"] = cell_types
        default = ("child-partition" if "child-partition" in categories
                   else "cell-type" if "cell-type" in categories
                   else "membership")
        payload_scenes.append({
            "node": int(record["node"]),
            "level": level,
            "scene": scene,
            "members": len(identifiers),
            "core_members": int(record["core_members"]),
            "identifiers": identifiers,
            "membership_score": [float(row["score"]) for row in members],
            "theta_f32": encode_array(theta),
            "categories": categories,
            "default_color_by": default,
            "clues": record.get("clues"),
            "factor_selection": record.get("factor_selection"),
            "views": {
                name: _load_view(
                    embeddings_path.parent, record, name, positions,
                    factors)
                for name in ("diffusion", "supervised", "quartimax_pca")
            },
        })
    node_labels = {
        int(row["node"]): (
            "Overview" if int(row["level"]) == 0 else
            f"Level {row['level']} · scene {row['scene']}")
        for row in node_rows}
    return {
        "title": "Multiresolution scene embedding atlas",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline_fingerprint": pipeline.get("fingerprint"),
        "factor_names": factors,
        "factor_labels": [f"F{factor}" for factor in factors],
        "factor_loading_labels": loading_labels,
        "global": global_payload,
        "scenes": payload_scenes,
        "edges": edges,
        "node_labels": node_labels,
        "levels": scenes_manifest.get("levels", []),
    }


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title><script src="__PLOTLY_URL__"></script>
<style>
:root{--ink:#202124;--muted:#667085;--line:#d9dee7;--panel:#fff;--bg:#f3f5f8}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.4 system-ui,sans-serif}
header{padding:18px 24px 8px}h1{font-size:22px;margin:0 0 4px}.subtitle,.hint{color:var(--muted)}
.global-controls{display:flex;gap:14px;align-items:center;padding:6px 24px 2px;color:var(--muted);flex-wrap:wrap}
.global-controls label{display:flex;align-items:center;gap:5px}.selection-count{font-size:12px}
button{border:1px solid #bfc7d4;border-radius:6px;padding:6px 10px;background:white;color:var(--ink);cursor:pointer}
.grid{display:grid;grid-template-columns:minmax(480px,1fr) minmax(480px,1fr);gap:14px;padding:10px 18px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;overflow:hidden;min-width:0}
.wide{grid-column:1/-1}.title{padding:12px 14px 0;font-weight:650}.controls{display:flex;gap:10px;align-items:end;padding:10px 14px 4px;flex-wrap:wrap}
label{color:var(--muted);font-size:12px;display:grid;gap:3px}select{min-width:145px;border:1px solid #bfc7d4;border-radius:6px;padding:6px;background:white}
.check-label{display:flex;flex-direction:row;align-items:center;gap:5px;padding:6px 0;color:var(--ink);cursor:pointer}
.factor-picker{display:flex;gap:7px;align-items:center;flex-wrap:wrap;padding:5px 14px 8px;border-top:1px solid #eef0f4}
.factor-choice{display:inline-flex;align-items:center;gap:4px;border:0;padding:2px 3px;background:transparent;font-size:11px;color:var(--muted)}
.factor-choice:hover{color:var(--ink)}.factor-circle{width:13px;height:13px;border:2px solid #6b7280;border-radius:50%;background:white}
.factor-choice.is-selected{color:var(--ink)}.factor-choice.is-selected .factor-circle{border-color:#31688e;background:#35b779;box-shadow:0 0 0 2px #d8efe5}.factor-id.is-abundant{font-weight:750}
.plot{height:500px}.loading{height:390px}.hint{font-size:11px;padding:0 14px 10px}.scene-bar{padding:12px 18px;display:flex;gap:14px;align-items:end;flex-wrap:wrap}
.diffusion-grid{display:grid;grid-template-columns:minmax(480px,1fr) minmax(480px,1fr);gap:14px;min-width:0}
.weight-grid{display:grid;grid-template-columns:minmax(620px,1.4fr) minmax(380px,.6fr);gap:14px;min-width:0}
.stats{padding:12px 14px 18px}.stat-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:10px}.stat{border:1px solid var(--line);border-radius:7px;padding:9px}.stat b{display:block;margin-bottom:4px}
.level-heading{margin:6px 4px -4px;font-size:19px}.diagnostic-heading{font-size:16px;font-weight:650;margin:2px 4px -4px}.nav{display:flex;gap:6px;flex-wrap:wrap}
@media(max-width:1050px){.grid{grid-template-columns:1fr}.wide{grid-column:auto}.diffusion-grid,.weight-grid{grid-template-columns:1fr}.stat-grid{grid-template-columns:1fr 1fr}}
</style></head><body>
<header><h1>__TITLE__</h1><div class="subtitle">Current production artifacts · adaptively weighted scene diffusion modes · hard cores plus halos</div></header>
<div class="global-controls"><button id="clear-global" type="button">Clear selection</button><span id="global-selection-count" class="selection-count"></span></div>
<main class="grid">
<section class="card"><div class="title">Global diffusion projection</div><div class="controls">
<label>Color points by<select id="g-color"></select></label><label>X mode<select id="g-x"></select></label><label>Y mode<select id="g-y"></select></label><label class="check-label"><input id="hide-muted-global" type="checkbox"> Hide muted</label></div>
<div id="global-plot" class="plot"></div><div class="hint">Selected unweighted diffusion modes come first, followed by up to 20 unselected modes. The full mode range is fixed when initialized; grouping and lasso selections never recenter or rescale it. In external-label mode, use the matching legend in the UMAP reference at right.</div></section>
<section class="card" id="umap-card"><div class="title">Supplied UMAP reference</div><div class="controls"><label class="check-label"><input id="hide-muted-umap" type="checkbox"> Hide muted</label></div><div id="umap-plot" class="plot"></div>
<div class="hint">Cell types and UMAP are external display metadata and never enter fitting. First-row box/lasso and legend selections are synchronized.</div></section>
<h2 id="level-heading" class="wide level-heading">Scene visualization</h2>
<section class="card wide"><div class="scene-bar"><label>Scene<select id="scene"></select></label>
<label>Color local views by<select id="color-by"></select></label>
<button id="clear-local" type="button">Clear local selection</button><span id="local-selection-count" class="selection-count"></span>
<span id="scene-summary" class="subtitle"></span><span id="navigation" class="nav"></span></div></section>
<div class="wide diffusion-grid"><section class="card"><div class="title">Scene-restricted global modes · selected grouping</div><div class="controls">
<label>X mode<select id="d-x"></select></label><label>Y mode<select id="d-y"></select></label><label class="check-label"><input id="hide-muted-diffusion-group" type="checkbox"> Hide muted points</label></div>
<div id="diffusion-group-plot" class="plot"></div><div class="hint">Global eigenvectors are centered on scene cores and weighted at the adaptive scene time. Selected axes come first, followed by up to 20 unselected modes in scene-importance order. Halos do not enter fitting.</div></section>
<section class="card"><div class="title">Scene-restricted global modes · factor theta</div><div id="d-factor-picker" class="factor-picker"></div>
<label class="check-label plot-control"><input id="hide-muted-diffusion-factor" type="checkbox"> Hide muted points</label>
<div id="diffusion-factor-plot" class="plot"></div><div class="hint">Shares the modes and fixed viewport at left. Click one factor; Ctrl/Cmd/Shift-click combines factors by summing theta. Values below 0.05 remain faint.</div></section></div>
<section class="card"><div class="title">Supervised next-cluster separation</div><div class="controls">
<label>X axis<select id="s-x"></select></label><label>Y axis<select id="s-y"></select></label><label class="check-label"><input id="hide-muted-supervised" type="checkbox"> Hide muted points</label></div><div id="supervised-plot" class="plot"></div>
<div class="hint">Mean-separation axes use child-scene cores when available; terminal scenes use private resolution-1 Leiden clues. Clue assignments are not hierarchy output.</div></section>
<section class="card"><div class="title">Quartimax-rotated PCA</div><div class="controls">
<label>X axis<select id="p-x"></select></label><label>Y axis<select id="p-y"></select></label><label class="check-label"><input id="hide-muted-quartimax-pca" type="checkbox"> Hide muted points</label></div><div id="pca-plot" class="plot"></div>
<div class="hint">Leading core-only factor-space PCs, quartimax rotated to concentrate loading energy without changing the retained subspace; halos are transformed afterward.</div></section>
<div class="wide weight-grid"><section class="card"><div class="title">Supervised factor contrasts</div><div id="supervised-loading" class="loading"></div></section>
<section class="card"><div class="title">Quartimax-PCA factor contrasts</div><div id="pca-loading" class="loading"></div></section></div>
<div class="wide diagnostic-heading">Scene diagnostics</div>
<div class="wide diffusion-grid"><section class="card"><div class="title">Summary</div><div id="stats" class="stats stat-grid"></div></section>
<section class="card"><div class="title">Global-mode spectrum</div><div id="diffusion-spectrum" class="plot"></div>
<div class="hint">Bars show adaptive-time-weighted core variance fractions; residuals above 0.5 identify parsimonious directions. Orange bars are selected scene axes.</div></section></div>
</main><script>
"use strict";const P=__PAYLOAD__;
function decode(s){const r=atob(s),b=new Uint8Array(r.length);for(let i=0;i<r.length;i++)b[i]=r.charCodeAt(i);return new Float32Array(b.buffer)}
function fmt(v,d=3){return v===null||v===undefined||!Number.isFinite(Number(v))?"—":Number(v).toFixed(d)}
function quantile(values,p){const a=values.filter(Number.isFinite).sort((x,y)=>x-y);if(!a.length)return 1;const z=p*(a.length-1),i=Math.floor(z),j=Math.ceil(z);return a[i]+(z-i)*(a[j]-a[i])}
function paddedRange(values){let lo=Infinity,hi=-Infinity;for(const value of values)if(Number.isFinite(value)){lo=Math.min(lo,value);hi=Math.max(hi,value)}if(!Number.isFinite(lo))return[-1,1];const span=Math.max(hi-lo,Math.abs(lo),Math.abs(hi),1e-8),pad=.04*span;return[lo-pad,hi+pad]}
const colors=["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf","#393b79","#637939","#8c6d31","#843c39","#7b4173","#3182bd","#31a354","#756bb1","#636363","#e6550d"];
function groupColor(group,g){const index=group.color_indices?group.color_indices[g]:g;return colors[index%colors.length]}
const config={responsive:true,displaylogo:false,scrollZoom:true,modeBarButtonsToAdd:["pan2d","select2d","lasso2d","resetScale2d"]};
const margin={l:70,r:155,t:20,b:65},loadingMargin={l:105,r:15,t:25,b:78};

const G=P.global;G.coordinates=decode(G.coordinates_f32);G.theta=decode(G.theta_f32);G.umap=G.umap_f32?decode(G.umap_f32):null;
const globalColorBy=document.getElementById("g-color");
const globalKinds={};for(const key of Object.keys(G.colorings)){const c=G.colorings[key];c.members=Array.from({length:c.names.length},()=>[]);for(let i=0;i<G.rows;i++)c.members[c.values[i]].push(i);globalKinds[key]=c}
const projectionKind=globalKinds.partition?"partition":Object.keys(globalKinds)[0],umapKind=globalKinds["cell-type"]?"cell-type":projectionKind;
const globalState={selection:null,source:null,hideMuted:{global:false,umap:false},rendering:false,active:{}};
for(const key of Object.keys(globalKinds))globalState.active[key]=Array(globalKinds[key].names.length).fill(true);
function globalAxis(axis){return Array.from({length:G.rows},(_,i)=>G.coordinates[i*G.dimensions+axis])}
function globalModeLabel(axis){return`Mode ${G.display_modes[axis]+1}${axis<G.selected_dimensions?" (selected)":""}`}
function globalKind(kind){return kind==="umap"?umapKind:globalColorBy.value}
function globalKindLabel(kind){return kind==="partition"?"Partition":kind==="cell-type"?"External labels":kind.replaceAll("-"," ")}
function globalShowsLegend(kind){return !(kind==="global"&&globalKind(kind)==="cell-type")}
function globalCoords(kind){if(kind==="umap"){const x=Array.from({length:G.rows},(_,i)=>G.umap[2*i]),y=Array.from({length:G.rows},(_,i)=>G.umap[2*i+1]);return{x,y,xt:"UMAP1",yt:"UMAP2",ranges:[paddedRange(x),paddedRange(y)],revision:"umap"}}const xm=Number(document.getElementById("g-x").value),ym=Number(document.getElementById("g-y").value),x=globalAxis(xm),y=globalAxis(ym);return{x,y,xt:globalModeLabel(xm),yt:globalModeLabel(ym),ranges:[paddedRange(x),paddedRange(y)],revision:`global-${xm}-${ym}`}}
function topFactors(theta,i){const K=P.factor_names.length,order=Array.from({length:K},(_,factor)=>factor).sort((a,b)=>theta[i*K+b]-theta[i*K+a]),top=theta[i*K+order[0]],terms=[];for(const factor of order){const value=theta[i*K+factor];if(terms.length>=3||value<.2*top)break;terms.push(`${P.factor_labels[factor]} = ${value.toFixed(3)}`)}return`Top factors: ${terms.join(" · ")}`}
function globalExternalLabel(i){const c=globalKinds["cell-type"];return c?`External label: ${c.names[c.values[i]]}`:""}
function sceneExternalLabel(scene,i){const c=scene.categories["cell-type"];return c?`External label: ${c.names[c.values[i]]}`:""}
function globalLegendTrace(group,g){return{type:"scattergl",mode:"markers",x:[null],y:[null],name:group.names[g],legendgroup:`global-${g}`,legendrank:g,showlegend:true,marker:{size:9,color:groupColor(group,g),opacity:1},hoverinfo:"skip",meta:{group:g,legendOnly:true}}}
function globalTrace(indices,coords,group,g,muted,hideMuted){const x=[],y=[],custom=[];for(const i of indices){if(!Number.isFinite(coords.x[i])||!Number.isFinite(coords.y[i]))continue;x.push(coords.x[i]);y.push(coords.y[i]);custom.push([G.identifiers[i],i,group.names[g],topFactors(G.theta,i),globalExternalLabel(i)])}return{type:"scattergl",mode:"markers",x,y,customdata:custom,name:group.names[g],legendgroup:`global-${g}`,showlegend:false,visible:muted&&hideMuted?false:true,marker:{size:6,color:groupColor(group,g),opacity:muted?.025:.7},hoverinfo:muted?"skip":"all",hovertemplate:muted?null:"%{customdata[0]}<br>%{customdata[4]}<br>%{customdata[3]}<extra>%{customdata[2]}</extra>",meta:{group:g}}}
function globalTraces(kind){const key=globalKind(kind),group=globalKinds[key],coords=globalCoords(kind),traces=[],hideMuted=globalState.hideMuted[kind],showLegend=globalShowsLegend(kind);for(let g=0;g<group.names.length;g++){if(showLegend)traces.push(globalLegendTrace(group,g));const active=[],muted=[];for(const i of group.members[g])(globalState.selection===null||globalState.selection.has(i)?active:muted).push(i);if(muted.length)traces.push(globalTrace(muted,coords,group,g,true,hideMuted));if(active.length)traces.push(globalTrace(active,coords,group,g,false,hideMuted))}return traces}
function globalLayout(kind){const c=globalCoords(kind),showLegend=globalShowsLegend(kind);return{margin:{...margin,r:showLegend?margin.r:20},paper_bgcolor:"white",plot_bgcolor:"white",dragmode:"pan",hovermode:"closest",showlegend:showLegend,legend:{x:1.02,y:1,font:{size:11}},xaxis:{title:c.xt,range:c.ranges[0],autorange:false,zeroline:false},yaxis:{title:c.yt,range:c.ranges[1],autorange:false,zeroline:false},uirevision:c.revision}}
async function drawGlobal(){globalState.rendering=true;const jobs=[Plotly.react("global-plot",globalTraces("global"),globalLayout("global"),config)];if(G.umap)jobs.push(Plotly.react("umap-plot",globalTraces("umap"),globalLayout("umap"),config));await Promise.all(jobs);globalState.rendering=false;bindGlobal(document.getElementById("global-plot"),"global");if(G.umap)bindGlobal(document.getElementById("umap-plot"),"umap");const count=globalState.selection===null?G.rows:globalState.selection.size;document.getElementById("global-selection-count").textContent=count===G.rows?"All points active":`${count.toLocaleString()} / ${G.rows.toLocaleString()} active · ${globalState.source}`}
function setGlobalSelection(indices,source){globalState.selection=indices&&indices.size<G.rows?indices:null;globalState.source=globalState.selection?source:null;drawGlobal()}
function globalLegendSelection(kind){for(const key of Object.keys(globalState.active))if(key!==kind)globalState.active[key].fill(true);const active=globalState.active[kind],group=globalKinds[kind];if(active.every(Boolean))return setGlobalSelection(null,null);const chosen=new Set();for(let g=0;g<active.length;g++)if(active[g])for(const i of group.members[g])chosen.add(i);setGlobalSelection(chosen,`${kind} legend`)}
let globalLegendTimer=null;function bindGlobal(plot,plotKind){if(plot._selectionBound)return;plot._selectionBound=true;plot.on("plotly_selected",event=>{if(globalState.rendering||!event||!event.points)return;for(const key of Object.keys(globalState.active))globalState.active[key].fill(true);setGlobalSelection(new Set(event.points.map(point=>Number(point.customdata[1]))),"box/lasso selection")});plot.on("plotly_legendclick",event=>{clearTimeout(globalLegendTimer);const kind=globalKind(plotKind),g=Number(plot.data[event.curveNumber].meta.group);globalLegendTimer=setTimeout(()=>{globalState.active[kind][g]=!globalState.active[kind][g];globalLegendSelection(kind)},240);return false});plot.on("plotly_legenddoubleclick",event=>{clearTimeout(globalLegendTimer);const kind=globalKind(plotKind),g=Number(plot.data[event.curveNumber].meta.group),active=globalState.active[kind],isolated=active[g]&&active.filter(Boolean).length===1;active.fill(isolated);if(!isolated)active[g]=true;globalLegendSelection(kind);return false})}

for(const scene of P.scenes){scene.theta=decode(scene.theta_f32);for(const view of Object.values(scene.views))if(view.available){view.coordinates=decode(view.coordinates_f32);if(view.loadings_f32)view.loadings=decode(view.loadings_f32)}}
const sceneSelect=document.getElementById("scene"),colorBy=document.getElementById("color-by");
P.scenes.forEach((scene,index)=>sceneSelect.add(new Option(`L${scene.level} / C${scene.scene} · ${scene.core_members} core · ${scene.members} scene`,index)));
function current(){return P.scenes[Number(sceneSelect.value)]}
const localState={selection:null,source:null,hideMuted:{diffusion_group:false,diffusion_factor:false,supervised:false,quartimax_pca:false},active:[],groupKey:null,rendering:false,factors:new Set([0])};
function groups(){return current().categories[colorBy.value]}
function ensureGroups(){const key=`${sceneSelect.value}-${colorBy.value}`,group=groups();if(localState.groupKey!==key){localState.groupKey=key;localState.active=Array(group.names.length).fill(true);localState.selection=null;localState.source=null}return group}
function fixedViewRange(view,axis){const values=[];for(let row=0;row<view.rows;row++)values.push(view.coordinates[row*view.dimensions+axis]);return paddedRange(values)}
function legendRanks(group){const counts=Array(group.names.length).fill(0);for(const value of group.values)counts[value]++;const order=Array.from({length:group.names.length},(_,g)=>g).sort((a,b)=>counts[b]-counts[a]||a-b),ranks=Array(group.names.length);for(let rank=0;rank<order.length;rank++)ranks[order[rank]]=rank;return ranks}
function legendTrace(group,g,rank){return{type:"scattergl",mode:"markers",x:[null],y:[null],name:group.names[g],legendgroup:`local-${g}`,legendrank:rank,showlegend:true,marker:{size:9,color:groupColor(group,g),opacity:1},hoverinfo:"skip",meta:{group:g,legendOnly:true}}}
function localTrace(rows,scene,view,xm,ym,group,g,core,muted,hideMuted){const x=[],y=[],custom=[];for(const row of rows){const i=view.indices[row];x.push(view.coordinates[row*view.dimensions+xm]);y.push(view.coordinates[row*view.dimensions+ym]);custom.push([scene.identifiers[i],i,group.names[g],core?"Core":"Halo",scene.membership_score[i],topFactors(scene.theta,i),sceneExternalLabel(scene,i)])}return{type:"scattergl",mode:"markers",x,y,customdata:custom,name:group.names[g],legendgroup:`local-${g}`,showlegend:false,visible:muted&&hideMuted?false:true,marker:{size:core?6:5,color:groupColor(group,g),opacity:muted?.025:(core?.72:.20),symbol:core?"circle":"circle-open"},hoverinfo:muted?"skip":"all",hovertemplate:muted?null:"%{customdata[0]}<br>%{customdata[6]}<br>%{customdata[5]}<br>%{customdata[3]}; membership score=%{customdata[4]:.3f}<extra>%{customdata[2]}</extra>",meta:{group:g,muted}}}
function localTraces(view,name){const scene=current(),group=ensureGroups(),traces=[],ranks=legendRanks(group),hideMuted=localState.hideMuted[name==="diffusion"?"diffusion_group":name];for(let g=0;g<group.names.length;g++){traces.push(legendTrace(group,g,ranks[g]));for(const muted of [true,false])for(const core of [0,1]){const rows=[];for(let row=0;row<view.rows;row++){const i=view.indices[row];if(group.values[i]===g&&view.core[row]===core&&(localState.selection!==null&&!localState.selection.has(i))===muted)rows.push(row)}if(rows.length)traces.push(localTrace(rows,scene,view,currentAxes(view).x,currentAxes(view).y,group,g,core,muted,hideMuted))}}return traces}
function selectorPrefix(view){return view===current().views.diffusion?"d":view===current().views.supervised?"s":"p"}
function currentAxes(view){const prefix=selectorPrefix(view);return{x:Number(document.getElementById(`${prefix}-x`).value),y:Number(document.getElementById(`${prefix}-y`).value)}}
function localModeLabel(view,axis){return`Mode ${view.display_modes[axis]+1}${axis<view.selected_dimensions?" (selected)":""}`}
function axisTitle(view,axis){if(!view.loadings)return localModeLabel(view,axis);const terms=[];for(let factor=0;factor<P.factor_names.length;factor++){const weight=view.loadings[axis*P.factor_names.length+factor];if(Math.abs(weight)>1e-10)terms.push({factor,weight})}terms.sort((a,b)=>Math.abs(b.weight)-Math.abs(a.weight));return`Axis ${axis+1}: `+terms.slice(0,4).map(v=>`${v.weight>=0?"+":"−"}${fmt(Math.abs(v.weight),2)}F${P.factor_names[v.factor]}`).join(" ")}
function viewLayout(view,name){const a=currentAxes(view);return{margin,paper_bgcolor:"white",plot_bgcolor:"white",dragmode:"pan",hovermode:"closest",legend:{x:1.02,y:1,font:{size:11}},xaxis:{title:axisTitle(view,a.x),range:fixedViewRange(view,a.x),autorange:false,zeroline:false},yaxis:{title:axisTitle(view,a.y),range:fixedViewRange(view,a.y),autorange:false,zeroline:false},uirevision:`${current().node}-${name}-${a.x}-${a.y}`}}
function omitted(target,reason){return Plotly.react(target,[],{annotations:[{text:reason||"Projection omitted",showarrow:false}],xaxis:{visible:false},yaxis:{visible:false}},config)}
function drawGroupedView(name,target){const view=current().views[name];if(!view.available)return omitted(target,view.reason);return Plotly.react(target,localTraces(view,name),viewLayout(view,name),config)}
function factorUpper(scene,view){const factors=Array.from(localState.factors),values=[];for(let row=0;row<view.rows;row++){const i=view.indices[row];let value=0;for(const factor of factors)value+=scene.theta[i*P.factor_names.length+factor];if(value>=.05)values.push(value)}return Math.max(.05,quantile(values,.95))}
function factorTraces(){const scene=current(),view=scene.views.diffusion;if(!view.available)return[];const a=currentAxes(view),factors=Array.from(localState.factors),upper=factorUpper(scene,view),low=[],high=[];for(const muted of [true,false])for(const core of [0,1])for(const lowValue of [true,false]){const x=[],y=[],values=[],custom=[];for(let row=0;row<view.rows;row++){const i=view.indices[row],isMuted=localState.selection!==null&&!localState.selection.has(i);if(isMuted!==muted||view.core[row]!==core)continue;let value=0;for(const factor of factors)value+=scene.theta[i*P.factor_names.length+factor];if((value<.05)!==lowValue)continue;x.push(view.coordinates[row*view.dimensions+a.x]);y.push(view.coordinates[row*view.dimensions+a.y]);values.push(value);custom.push([scene.identifiers[i],i,value,core?"Core":"Halo",scene.membership_score[i],topFactors(scene.theta,i),sceneExternalLabel(scene,i)])}if(!x.length)continue;const trace={type:"scattergl",mode:"markers",x,y,customdata:custom,showlegend:false,visible:muted&&localState.hideMuted.diffusion_factor?false:true,marker:{size:core?6:5,color:values,colorscale:"Viridis",cmin:0,cmax:upper,showscale:false,opacity:lowValue?(muted?.01:.06):(muted?.025:(core?.72:.20)),symbol:core?"circle":"circle-open"},hoverinfo:muted?"skip":"all",hovertemplate:muted?null:"%{customdata[0]}<br>%{customdata[6]}<br>%{customdata[5]}<br>Selected theta = %{customdata[2]:.4f}<br>%{customdata[3]}; membership score=%{customdata[4]:.3f}<extra></extra>",meta:{muted,lowValue}};(lowValue?low:high).push(trace)}const scale=high.find(trace=>!trace.meta.muted)||high[0];if(scale){scale.marker.showscale=true;scale.marker.colorbar={title:{text:`Theta<br>q95=${fmt(upper)}`},thickness:14,len:.66}}return low.concat(high)}
function drawFactor(){const view=current().views.diffusion;if(!view.available)return omitted("diffusion-factor-plot",view.reason);return Plotly.react("diffusion-factor-plot",factorTraces(),viewLayout(view,"diffusion-factor"),config)}
async function drawLocal(){localState.rendering=true;await Promise.all([drawGroupedView("diffusion","diffusion-group-plot"),drawFactor(),drawGroupedView("supervised","supervised-plot"),drawGroupedView("quartimax_pca","pca-plot")]);localState.rendering=false;for(const id of ["diffusion-group-plot","supervised-plot","pca-plot"])bindLocal(document.getElementById(id));bindLocalSelection(document.getElementById("diffusion-factor-plot"));bindDiffusionViewport();updateLocalSummary()}
function updateLocalSummary(){const scene=current(),count=localState.selection===null?scene.members:localState.selection.size;document.getElementById("local-selection-count").textContent=count===scene.members?"All scene points active":`${count.toLocaleString()} / ${scene.members.toLocaleString()} active · ${localState.source}`}
function setLocalSelection(indices,source){localState.selection=indices&&indices.size<current().members?indices:null;localState.source=localState.selection?source:null;drawLocal()}
function selectionFromLegend(){const scene=current(),group=ensureGroups();if(localState.active.every(Boolean))return setLocalSelection(null,null);const chosen=new Set();for(let i=0;i<scene.members;i++)if(localState.active[group.values[i]])chosen.add(i);setLocalSelection(chosen,`${colorBy.options[colorBy.selectedIndex].text} legend`)}
let localLegendTimer=null;function bindLocal(plot){bindLocalSelection(plot);if(plot._legendBound)return;plot._legendBound=true;plot.on("plotly_legendclick",event=>{const trace=plot.data[event.curveNumber];if(!trace||!trace.meta)return false;clearTimeout(localLegendTimer);localLegendTimer=setTimeout(()=>{localState.active[Number(trace.meta.group)]=!localState.active[Number(trace.meta.group)];selectionFromLegend()},240);return false});plot.on("plotly_legenddoubleclick",event=>{clearTimeout(localLegendTimer);const trace=plot.data[event.curveNumber];if(!trace||!trace.meta)return false;const g=Number(trace.meta.group),isolated=localState.active[g]&&localState.active.filter(Boolean).length===1;localState.active.fill(isolated);if(!isolated)localState.active[g]=true;selectionFromLegend();return false})}
function bindLocalSelection(plot){if(plot._localSelectionBound)return;plot._localSelectionBound=true;plot.on("plotly_selected",event=>{if(localState.rendering||!event||!event.points)return;localState.active.fill(true);setLocalSelection(new Set(event.points.map(point=>Number(point.customdata[1]))),"box/lasso selection")})}
let viewportSync=false;function bindDiffusionViewport(){const first=document.getElementById("diffusion-group-plot"),second=document.getElementById("diffusion-factor-plot");if(first._viewportBound)return;first._viewportBound=second._viewportBound=true;for(const [source,target] of [[first,second],[second,first]])source.on("plotly_relayout",event=>{if(localState.rendering||viewportSync)return;const update={};for(const key of ["xaxis.range[0]","xaxis.range[1]","yaxis.range[0]","yaxis.range[1]","xaxis.autorange","yaxis.autorange"])if(Object.hasOwn(event,key))update[key]=event[key];if(!Object.keys(update).length)return;viewportSync=true;Plotly.relayout(target,update).finally(()=>{viewportSync=false})})}
function fillAxes(name,prefix){const view=current().views[name],x=document.getElementById(`${prefix}-x`),y=document.getElementById(`${prefix}-y`);x.innerHTML="";y.innerHTML="";if(!view.available){x.disabled=y.disabled=true;return}x.disabled=y.disabled=false;for(let axis=0;axis<view.dimensions;axis++){const label=name==="diffusion"?localModeLabel(view,axis):`Axis ${axis+1}`;x.add(new Option(label,axis));y.add(new Option(label,axis))}x.value="0";y.value=String(Math.min(1,view.dimensions-1))}
function sceneFactorAbundance(scene){const total=Array(P.factor_names.length).fill(0);let mass=0;for(let row=0;row<scene.members;row++)for(let factor=0;factor<P.factor_names.length;factor++){const value=scene.theta[row*P.factor_names.length+factor];total[factor]+=value;mass+=value}return total.map(value=>mass>0?value/mass:0)}
function factorPicker(){const picker=document.getElementById("d-factor-picker"),abundance=sceneFactorAbundance(current());picker.innerHTML="";for(let factor=0;factor<P.factor_names.length;factor++){const selected=localState.factors.has(factor),button=document.createElement("button");button.type="button";button.className=`factor-choice${selected?" is-selected":""}`;button.title=`${P.factor_labels[factor]} · ${(100*abundance[factor]).toFixed(2)}% of scene theta · click to select alone; Ctrl/Cmd/Shift-click to toggle`;button.setAttribute("aria-pressed",selected?"true":"false");const circle=document.createElement("span");circle.className="factor-circle";button.appendChild(circle);const text=document.createElement("span");text.className=`factor-id${abundance[factor]>=.02?" is-abundant":""}`;text.textContent=`F${P.factor_names[factor]}`;button.appendChild(text);button.onclick=event=>{const additive=event.ctrlKey||event.metaKey||event.shiftKey;if(additive){if(selected&&localState.factors.size>1)localState.factors.delete(factor);else if(!selected)localState.factors.add(factor)}else{localState.factors.clear();localState.factors.add(factor)}factorPicker();drawFactor()};picker.appendChild(button)}}
function drawLoading(name,target){const view=current().views[name];if(!view.available||!view.loadings)return omitted(target,view.reason);const prefix=name==="supervised"?"s":"p",axes=[Number(document.getElementById(`${prefix}-x`).value),Number(document.getElementById(`${prefix}-y`).value)],tick=P.factor_names.map(f=>`F${f}`),traces=axes.map((axis,index)=>({type:"bar",orientation:"h",y:tick,x:P.factor_names.map((_,factor)=>view.loadings[axis*P.factor_names.length+factor]),name:`Axis ${axis+1}`,marker:{color:index?"#d95f02":"#1b9e77"},customdata:tick,hovertemplate:"%{customdata}<br>signed weight=%{x:.4g}<extra></extra>"})),ticktext=name==="supervised"?P.factor_loading_labels:tick;return Plotly.react(target,traces,{margin:loadingMargin,barmode:"group",paper_bgcolor:"white",plot_bgcolor:"white",xaxis:{title:"normalized signed factor weight",zeroline:true},yaxis:{automargin:true,tickmode:"array",tickvals:tick,ticktext,categoryorder:"array",categoryarray:tick,range:[tick.length-.5,-.5]},legend:{orientation:"h",x:0,y:-.18},uirevision:`loading-${current().node}-${name}-${axes.join("-")}`},config)}
function drawSpectrum(){const view=current().views.diffusion;if(!view.available||!view.spectrum)return omitted("diffusion-spectrum",view.reason);let rows=view.spectrum.filter(row=>row.importance_rank!==null);if(!rows.length)rows=view.spectrum;rows=rows.slice().sort((a,b)=>(a.importance_rank??1e9)-(b.importance_rank??1e9)||a.mode-b.mode);const x=rows.map((_,i)=>i+1),residual=rows.map(row=>row.regression_residual),traces=[{type:"bar",x,y:rows.map(row=>row.variance_fraction),name:"core variance fraction",marker:{color:rows.map(row=>row.selected?"#d95f02":"#9eb3cf")},customdata:rows.map(row=>row.mode),hovertemplate:"global mode %{customdata}<br>variance=%{y:.4f}<extra></extra>"},{type:"scatter",mode:"lines+markers",x,y:residual,name:"regression residual",yaxis:"y2",line:{color:"#1b9e77"},customdata:rows.map(row=>row.mode),hovertemplate:"global mode %{customdata}<br>residual=%{y:.3f}<extra></extra>"}];return Plotly.react("diffusion-spectrum",traces,{margin:{l:58,r:55,t:30,b:75},paper_bgcolor:"white",plot_bgcolor:"white",legend:{orientation:"h",x:0,y:1.12},xaxis:{title:"scene importance rank",tickmode:"array",tickvals:x,ticktext:rows.map(row=>`m${row.mode}`),tickangle:-45},yaxis:{title:"core variance fraction",rangemode:"tozero"},yaxis2:{title:"residual",overlaying:"y",side:"right",range:[0,Math.max(1.05,...residual.filter(Number.isFinite).map(value=>1.05*value))]},shapes:[{type:"line",xref:"paper",x0:0,x1:1,yref:"y2",y0:.5,y1:.5,line:{color:"#1b9e77",dash:"dot",width:1}}],uirevision:`spectrum-${current().node}`},config)}
function drawStats(){const scene=current(),d=scene.views.diffusion,s=scene.views.supervised,p=scene.views.quartimax_pca,clue=scene.clues||{},factor=scene.factor_selection||{},parents=P.edges.filter(e=>e.child===scene.node).length,children=P.edges.filter(e=>e.parent===scene.node).length;document.getElementById("stats").innerHTML=`<div class="stat"><b>Scene membership</b>${scene.core_members.toLocaleString()} core + ${(scene.members-scene.core_members).toLocaleString()} halo = ${scene.members.toLocaleString()} · ${parents} parent edge(s) · ${children} child edge(s)</div><div class="stat"><b>Global-mode view</b>${d.available?`${d.fit_rows} representative fit rows · ${d.selected_dimensions} selected axes · adaptive t=${fmt(d.diffusion_time)} · effective rank ${fmt(d.achieved_effective_rank)} / ${fmt(d.target_effective_rank)}${d.basis_limited?" · basis limited":""}`:d.reason}</div><div class="stat"><b>Supervised clues</b>${clue.source||"—"} · ${clue.groups??"—"} retained groups · ${clue.fit_rows??"—"} fit rows · ${clue.excluded_rows??"—"} excluded</div><div class="stat"><b>Supervised view</b>${s.available?`${s.dimensions} axes · ${s.fit_rows} fit rows · ${s.represented_groups} groups`:s.reason}</div><div class="stat"><b>Quartimax-rotated PCA</b>${p.available?`${p.dimensions} axes · retained variance ${fmt(100*p.retained_subspace_variance_fraction,1)}% · quartimax ${fmt(p.quartimax_objective)}`:p.reason}</div><div class="stat"><b>Factor subspace</b>${factor.retained_factors?factor.retained_factors.length:"—"} factors · retained core mass ${factor.retained_core_mass_proportion?fmt(100*factor.retained_core_mass_proportion,2)+"%":"—"}</div>`}
function navigation(){const scene=current(),box=document.getElementById("navigation");box.innerHTML="";for(const [label,nodes] of [["Parent",P.edges.filter(e=>e.child===scene.node).map(e=>e.parent)],["Child",P.edges.filter(e=>e.parent===scene.node).map(e=>e.child)]])for(const node of nodes){if(node===0)continue;const index=P.scenes.findIndex(value=>value.node===node);if(index<0)continue;const button=document.createElement("button");button.textContent=`${label}: ${P.node_labels[String(node)]}`;button.onclick=()=>{sceneSelect.value=String(index);drawScene()};box.appendChild(button)}}
async function drawScene(){const scene=current();document.getElementById("level-heading").textContent=`Level-${scene.level} visualization`;document.getElementById("scene-summary").textContent=`${scene.core_members.toLocaleString()} core + ${(scene.members-scene.core_members).toLocaleString()} halo`;colorBy.innerHTML="";for(const key of Object.keys(scene.categories)){const label=key==="child-partition"?"Next-level cluster":key==="cell-type"?"Cell type":"Core / halo";colorBy.add(new Option(label,key))}colorBy.value=scene.default_color_by;localState.groupKey=null;fillAxes("diffusion","d");fillAxes("supervised","s");fillAxes("quartimax_pca","p");factorPicker();navigation();drawStats();drawSpectrum();drawLoading("supervised","supervised-loading");drawLoading("quartimax_pca","pca-loading");await drawLocal()}
globalColorBy.add(new Option(globalKindLabel(projectionKind),projectionKind));if(globalKinds["cell-type"]&&projectionKind!=="cell-type")globalColorBy.add(new Option(globalKindLabel("cell-type"),"cell-type"));globalColorBy.value=projectionKind;
for(let axis=0;axis<G.dimensions;axis++){document.getElementById("g-x").add(new Option(globalModeLabel(axis),axis));document.getElementById("g-y").add(new Option(globalModeLabel(axis),axis))}document.getElementById("g-x").value="0";document.getElementById("g-y").value="1";
document.getElementById("umap-card").hidden=!G.umap;globalColorBy.onchange=drawGlobal;document.getElementById("g-x").onchange=drawGlobal;document.getElementById("g-y").onchange=drawGlobal;for(const kind of ["global","umap"])document.getElementById(`hide-muted-${kind}`).onchange=event=>{globalState.hideMuted[kind]=event.target.checked;drawGlobal()};document.getElementById("clear-global").onclick=()=>{for(const key of Object.keys(globalState.active))globalState.active[key].fill(true);setGlobalSelection(null,null)};
sceneSelect.onchange=drawScene;colorBy.onchange=()=>{localState.groupKey=null;drawLocal()};for(const [id,key] of [["hide-muted-diffusion-group","diffusion_group"],["hide-muted-diffusion-factor","diffusion_factor"],["hide-muted-supervised","supervised"],["hide-muted-quartimax-pca","quartimax_pca"]])document.getElementById(id).onchange=event=>{localState.hideMuted[key]=event.target.checked;drawLocal()};document.getElementById("clear-local").onclick=()=>{ensureGroups();localState.active.fill(true);setLocalSelection(null,null)};
for(const [id,name,target] of [["d-x","diffusion",null],["d-y","diffusion",null],["s-x","supervised","supervised-loading"],["s-y","supervised","supervised-loading"],["p-x","quartimax_pca","pca-loading"],["p-y","quartimax_pca","pca-loading"]])document.getElementById(id).onchange=()=>{if(name==="diffusion"){drawGroupedView("diffusion","diffusion-group-plot");drawFactor()}else{drawGroupedView(name,name==="supervised"?"supervised-plot":"pca-plot");drawLoading(name,target)}};
sceneSelect.value="0";drawGlobal();drawScene();
</script></body></html>"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline", type=Path, required=True,
                        help="pipeline directory or manifest.json")
    parser.add_argument("--out-html", type=Path, required=True)
    metadata_arguments(parser)
    args = parser.parse_args()
    payload = build_payload(
        args.pipeline, args.metadata, args.metadata_id_column,
        args.cell_type_column, args.umap_x_column, args.umap_y_column,
        args.de)
    html = HTML_TEMPLATE.replace("__TITLE__", payload["title"]) \
        .replace("__PLOTLY_URL__", PLOTLY_URL) \
        .replace("__PAYLOAD__", json_for_html(payload))
    args.out_html.parent.mkdir(parents=True, exist_ok=True)
    args.out_html.write_text(html, encoding="utf-8")
    print(f"Wrote {args.out_html} ({len(payload['scenes'])} scenes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
