#!/usr/bin/env python3
"""Build a temporary, development-only Level-0 diagnostic HTML report.

This viewer consumes production artifacts, but its HTML and UI are not a
supported output contract.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from multires_diagnostics.multires_report_common import (
    encode_array,
    json_for_html,
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


def categorical(values: list[str]) -> dict:
    names = sorted(set(values), key=lambda value: (
        0, int(value[1:])) if value.startswith("C") and value[1:].isdigit()
        else (1, value))
    index = {name: position for position, name in enumerate(names)}
    return {"names": names, "values": [index[value] for value in values]}


def build_payload(pipeline_path: Path, metadata_path: Path | None,
                  metadata_id_column: str | None,
                  cell_type_column: str | None,
                  umap_x_column: str | None,
                  umap_y_column: str | None,
                  de_path: Path | None, default_color_by: str) -> dict:
    _, pipeline, stages = resolve_pipeline(pipeline_path)
    if "graph" not in stages or "level0" not in stages:
        raise ValueError("Pipeline must contain graph and Level-0 stages")
    graph = load_graph_context(stages["graph"])
    level0_path, level0 = require_artifact(
        stages["level0"], "punkst.multires.level0")
    public = level0.get("public_tables", {})
    embedding_path = level0_path.parent / public.get("embedding", "")
    header, rows = read_tsv(embedding_path)
    selected_axes = numeric_columns(header)
    if len(selected_axes) < 2:
        raise ValueError("Level-0 embedding has fewer than two axes")
    population = level0.get("representation", {}).get("population")
    id_column = "id" if population == "points" else "representative_id"
    if id_column not in header:
        raise ValueError("Level-0 identifier column does not match population")
    identifiers = [row[id_column] for row in rows]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Level-0 identifiers are not unique")
    coordinates = np.asarray(
        [[float(row[column]) for column in selected_axes] for row in rows],
        dtype=np.float32)
    if not np.isfinite(coordinates).all():
        raise ValueError("Level-0 coordinates must be finite")
    theta = load_theta(graph, identifiers)
    factors = graph["factor_names"]
    axes_header, axes_rows = read_tsv(level0_path.parent / public["axes"])
    if not {"axis", "mode"}.issubset(axes_header):
        raise ValueError("Level-0 axis table has an invalid header")
    selected_modes = [int(row["mode"]) for row in axes_rows]
    if len(selected_modes) != len(selected_axes):
        raise ValueError("Level-0 selected modes do not match embedding axes")
    alternate_name = public.get("alternate_modes")
    alternate_modes: list[int] = []
    if alternate_name:
        alternate_header, alternate_rows = read_tsv(
            level0_path.parent / alternate_name)
        alternate_columns = alternate_header[1:]
        if (not alternate_header or alternate_header[0] != "id"
                or any(not column.startswith("mode_")
                       for column in alternate_columns)
                or [row["id"] for row in alternate_rows] != identifiers):
            raise ValueError("Level-0 alternate modes do not align")
        alternate_modes = [int(column.removeprefix("mode_"))
                           for column in alternate_columns]
        if len(alternate_modes) > 20:
            raise ValueError("Level-0 exposes more than 20 alternate modes")
        if alternate_modes:
            alternate_coordinates = np.asarray([
                [float(row[column]) for column in alternate_columns]
                for row in alternate_rows], dtype=np.float32)
            if not np.isfinite(alternate_coordinates).all():
                raise ValueError("Level-0 alternate coordinates must be finite")
            coordinates = np.column_stack((coordinates, alternate_coordinates))
    display_modes = [*selected_modes, *alternate_modes]
    metadata = load_metadata(
        metadata_path, id_column=metadata_id_column,
        cell_type_column=cell_type_column,
        umap_x_column=umap_x_column, umap_y_column=umap_y_column)

    colorings: dict[str, dict] = {}
    if "selection" in stages:
        selection_path, selection = require_artifact(
            stages["selection"], "punkst.multires.selection")
        levels = selection.get("levels", [])
        level1 = next((row for row in levels if row.get("level") == 1), None)
        if level1 is not None:
            partition_path = selection_path.parent / level1["partition_table"]
            partition_header, partition_rows = read_tsv(partition_path)
            if partition_header != ["id", "cluster"]:
                raise ValueError("Level-1 partition table has an invalid header")
            lookup = {row["id"]: row["cluster"] for row in partition_rows}
            missing = [value for value in identifiers if value not in lookup]
            if missing:
                raise ValueError("Level-1 partition does not cover Level-0 IDs")
            colorings["partition"] = categorical([
                f"C{lookup[value]}" for value in identifiers])
    if metadata and any("cell_type" in metadata.get(value, {})
                        for value in identifiers):
        colorings["cell-type"] = categorical([
            metadata.get(value, {}).get("cell_type", "Unannotated")
            for value in identifiers])
    if not colorings:
        colorings["population"] = categorical(["All points"] * len(rows))
    if default_color_by not in colorings:
        default_color_by = next(iter(colorings))

    has_umap = any("umap" in metadata.get(value, {}) for value in identifiers)
    umap = np.full((len(rows), 2), np.nan, dtype=np.float32)
    if has_umap:
        for row, identifier in enumerate(identifiers):
            value = metadata.get(identifier, {}).get("umap")
            if value is not None:
                umap[row] = value
    return {
        "title": "Multiresolution Level-0 embedding",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline_fingerprint": pipeline.get("fingerprint"),
        "population": population,
        "rows": len(rows),
        "identifiers": identifiers,
        "dimensions": len(display_modes),
        "selected_dimensions": len(selected_modes),
        "display_modes": display_modes,
        "coordinates_f32": encode_array(coordinates),
        "factor_names": factors,
        "factor_labels": [f"F{value}" for value in factors],
        "theta_f32": encode_array(theta),
        "colorings": colorings,
        "default_color_by": default_color_by,
        "umap_f32": encode_array(umap) if has_umap else None,
    }


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title><script src="__PLOTLY_URL__"></script>
<style>
:root{--bg:#f4f6f8;--panel:#fff;--line:#d8dee8;--ink:#1d2433;--muted:#687386}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.4 system-ui,sans-serif}
header{padding:18px 22px 8px}h1{font-size:22px;margin:0 0 3px}.sub{color:var(--muted)}
.controls{display:flex;gap:14px;flex-wrap:wrap;align-items:center;padding:8px 22px 14px}
label{display:flex;gap:6px;align-items:center;color:var(--muted)}select{padding:5px 8px;max-width:440px}
.factor-picker{display:flex;gap:7px;align-items:center;flex-wrap:wrap;padding:9px 14px 0}
.factor-choice{display:inline-flex;align-items:center;gap:4px;border:0;background:transparent;color:var(--muted);padding:2px 3px;cursor:pointer}
.factor-choice:hover{color:var(--ink)}.factor-circle{width:13px;height:13px;border:2px solid #6b7280;border-radius:50%;background:white}
.factor-choice.is-selected{color:var(--ink);font-weight:650}.factor-choice.is-selected .factor-circle{border-color:#31688e;background:#35b779;box-shadow:0 0 0 2px #d8efe5}
.hint{color:var(--muted);font-size:11px;padding:0 14px 10px}.selection{color:var(--muted)}
.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;padding:0 14px 14px}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:8px;min-width:0;overflow:hidden}
.panel h2{font-size:15px;margin:10px 14px 0}.plot{height:520px}
.plot-control{padding:5px 14px 0;font-size:12px}
@media(max-width:900px){.grid{grid-template-columns:1fr}}
</style></head><body>
<header><h1>__TITLE__</h1><div class="sub" id="summary"></div></header>
<div class="controls">
<label>Color <select id="color"></select></label>
<label>X axis <select id="xaxis"></select></label>
<label>Y axis <select id="yaxis"></select></label>
<span id="selection" class="selection"></span>
</div>
<main class="grid">
<section class="panel"><h2>Diffusion embedding · groups</h2><label class="plot-control"><input id="hide-muted-groups" type="checkbox"> Hide muted</label><div id="embedding-groups" class="plot"></div></section>
<section class="panel"><h2>Diffusion embedding · factor highlight</h2><label class="plot-control"><input id="hide-muted-factor" type="checkbox"> Hide muted</label><div id="factor-picker" class="factor-picker"></div><div id="embedding-factor" class="plot"></div><div class="hint">Click a factor to select it alone; Ctrl/Cmd/Shift-click toggles factors. Selected theta values are summed. The adaptive upper color limit is the whole-population 95th percentile among values at least 0.05.</div></section>
<section class="panel" id="umap-panel"><h2>Externally supplied UMAP</h2><label class="plot-control"><input id="hide-muted-umap" type="checkbox"> Hide muted</label><div id="umap" class="plot"></div></section>
</main>
<script>const P=__PAYLOAD__;
function f32(encoded){const s=atob(encoded),b=new Uint8Array(s.length);for(let i=0;i<s.length;i++)b[i]=s.charCodeAt(i);return new Float32Array(b.buffer)}
const Z=f32(P.coordinates_f32),T=f32(P.theta_f32),U=P.umap_f32?f32(P.umap_f32):null;
const colors=["#4477aa","#ee6677","#228833","#ccbb44","#66ccee","#aa3377","#bbbbbb","#332288","#88ccee","#44aa99","#117733","#999933","#ddcc77","#cc6677","#882255","#661100"];
const config={responsive:true,displaylogo:false,scrollZoom:true};
const margin={l:58,r:18,t:18,b:52};
const state={active:[],factors:new Set([0]),rendering:false,hideMuted:{groups:false,factor:false,umap:false}};
function q(v,p){const a=v.filter(Number.isFinite).sort((x,y)=>x-y);if(!a.length)return 1;const z=p*(a.length-1),i=Math.floor(z),j=Math.ceil(z);return a[i]+(z-i)*(a[j]-a[i])}
function axis(a){const out=[];for(let i=0;i<P.rows;i++)out.push(Z[i*P.dimensions+a]);return out}
function fixedRange(values){let lo=Infinity,hi=-Infinity;for(const v of values)if(Number.isFinite(v)){lo=Math.min(lo,v);hi=Math.max(hi,v)}if(!Number.isFinite(lo))return[-1,1];if(lo===hi){const d=Math.max(1,Math.abs(lo)*.05);return[lo-d,hi+d]}const pad=.04*(hi-lo);return[lo-pad,hi+pad]}
function externalLabel(i){const c=P.colorings["cell-type"];return c?`External label: ${c.names[c.values[i]]}`:""}
function topFactors(i){const K=P.factor_names.length,order=Array.from({length:K},(_,factor)=>factor).sort((a,b)=>T[i*K+b]-T[i*K+a]),top=T[i*K+order[0]],terms=[];for(const factor of order){const value=T[i*K+factor];if(terms.length>=3||value<.2*top)break;terms.push(`${P.factor_labels[factor]} = ${value.toFixed(3)}`)}return`Top factors: ${terms.join(" · ")}`}
function pointHover(i){const label=externalLabel(i);return`${P.identifiers[i]}${label?`<br>${label}`:""}<br>${topFactors(i)}`}
function colorLegendTrace(c,g){return{type:"scattergl",mode:"markers",name:c.names[g],x:[null],y:[null],legendgroup:`color-${g}`,showlegend:true,marker:{size:12,color:colors[g%colors.length],opacity:state.active[g]?1:.2},hoverinfo:"skip",meta:{group:g,legendOnly:true}}}
function groupTraces(x,y,source){const c=P.colorings[source],tr=[];for(let g=0;g<c.names.length;g++){const xx=[],yy=[],custom=[],active=state.active[g];for(let i=0;i<P.rows;i++)if(c.values[i]===g){xx.push(x[i]);yy.push(y[i]);custom.push(pointHover(i))}tr.push(colorLegendTrace(c,g));tr.push({type:"scattergl",mode:"markers",name:c.names[g],legendgroup:`color-${g}`,showlegend:false,x:xx,y:yy,customdata:custom,meta:{group:g},visible:!active&&state.hideMuted.groups?false:true,marker:{size:6,color:colors[g%colors.length],opacity:active?.72:.045},hoverinfo:active?"all":"skip",hovertemplate:active?"%{customdata}<extra>%{fullData.name}</extra>":null})}return tr}
function factorValues(){const factors=Array.from(state.factors),values=[];for(let i=0;i<P.rows;i++){let value=0;for(const factor of factors)value+=T[i*P.factor_names.length+factor];values.push(value)}return values}
function factorTraces(x,y){const values=factorValues(),c=P.colorings[color.value],hi=Math.max(.05,q(values.filter(v=>v>=.05),.95)),tr=[];for(const kind of ["muted","low","high"]){const xx=[],yy=[],v=[],custom=[];for(let i=0;i<P.rows;i++){const active=state.active[c.values[i]],match=kind==="muted"?!active:kind==="low"?active&&values[i]<.05:active&&values[i]>=.05;if(match){xx.push(x[i]);yy.push(y[i]);v.push(values[i]);custom.push([pointHover(i),values[i]])}}if(!xx.length)continue;const high=kind==="high",muted=kind==="muted";tr.push({type:"scattergl",mode:"markers",x:xx,y:yy,customdata:custom,showlegend:false,visible:muted&&state.hideMuted.factor?false:true,marker:{size:6,color:high?v:"#b8bec8",colorscale:"Viridis",cmin:0,cmax:hi,showscale:high,colorbar:{title:{text:`Theta<br>q95=${hi.toFixed(3)}`},thickness:14,len:.615},opacity:high?.78:kind==="low"?.10:.025},hoverinfo:muted?"skip":"all",hovertemplate:muted?null:"%{customdata[0]}<br>Selected theta = %{customdata[1]:.4f}<extra></extra>"})}return tr}
function layout(xname,yname,key,x,y){return {margin,paper_bgcolor:"white",plot_bgcolor:"white",dragmode:"pan",hovermode:"closest",xaxis:{title:xname,zeroline:false,autorange:false,range:fixedRange(x)},yaxis:{title:yname,zeroline:false,autorange:false,range:fixedRange(y)},legend:{x:1.01,y:1,font:{size:10}},uirevision:key}}
function updateSelection(){const c=P.colorings[color.value],active=c.values.reduce((sum,g)=>sum+(state.active[g]?1:0),0);document.getElementById("selection").textContent=`${active.toLocaleString()} / ${P.rows.toLocaleString()} points active`}
function modeLabel(axis){return`Mode ${P.display_modes[axis]+1}${axis<P.selected_dimensions?" (selected)":""}`}
async function drawEmbedding(){const xa=Number(xaxis.value),ya=Number(yaxis.value),x=axis(xa),y=axis(ya),source=color.value;state.rendering=true;await Promise.all([Plotly.react("embedding-groups",groupTraces(x,y,source),layout(modeLabel(xa),modeLabel(ya),`g-${xa}-${ya}-${source}`,x,y),config),Plotly.react("embedding-factor",factorTraces(x,y),layout(modeLabel(xa),modeLabel(ya),`f-${xa}-${ya}`,x,y),config)]);state.rendering=false;bindLegend(document.getElementById("embedding-groups"));bindViewportSync();updateSelection()}
async function drawUmap(){const panel=document.getElementById("umap-panel");if(!U){panel.hidden=true;return}const x=[],y=[],valid=[];for(let i=0;i<P.rows;i++)if(Number.isFinite(U[2*i])&&Number.isFinite(U[2*i+1])){x.push(U[2*i]);y.push(U[2*i+1]);valid.push(i)}const c=P.colorings[color.value],tr=[];for(let g=0;g<c.names.length;g++){const xx=[],yy=[],custom=[],active=state.active[g];for(let j=0;j<valid.length;j++){const i=valid[j];if(c.values[i]===g){xx.push(x[j]);yy.push(y[j]);custom.push(pointHover(i))}}tr.push(colorLegendTrace(c,g));tr.push({type:"scattergl",mode:"markers",name:c.names[g],legendgroup:`color-${g}`,showlegend:false,x:xx,y:yy,customdata:custom,meta:{group:g},visible:!active&&state.hideMuted.umap?false:true,marker:{size:6,color:colors[g%colors.length],opacity:active?.72:.045},hoverinfo:active?"all":"skip",hovertemplate:active?"%{customdata}<extra>%{fullData.name}</extra>":null})}await Plotly.react("umap",tr,layout("UMAP 1","UMAP 2",`u-${color.value}`,x,y),config);bindLegend(document.getElementById("umap"))}
function resetGroups(){state.active=Array(P.colorings[color.value].names.length).fill(true)}
let legendTimer=null;function bindLegend(plot){if(plot._groupSelectionBound)return;plot._groupSelectionBound=true;plot.on("plotly_legendclick",event=>{if(state.rendering)return false;const group=Number(plot.data[event.curveNumber].meta.group);clearTimeout(legendTimer);legendTimer=setTimeout(()=>{state.active[group]=!state.active[group];drawEmbedding();drawUmap()},240);return false});plot.on("plotly_legenddoubleclick",event=>{if(state.rendering)return false;clearTimeout(legendTimer);const group=Number(plot.data[event.curveNumber].meta.group),isolated=state.active[group]&&state.active.filter(Boolean).length===1;state.active.fill(isolated);if(!isolated)state.active[group]=true;drawEmbedding();drawUmap();return false})}
let syncing=false;function bindViewportSync(){const first=document.getElementById("embedding-groups"),second=document.getElementById("embedding-factor");if(first._viewportBound)return;first._viewportBound=second._viewportBound=true;for(const [source,target] of [[first,second],[second,first]])source.on("plotly_relayout",event=>{if(state.rendering||syncing)return;const update={};for(const key of ["xaxis.range[0]","xaxis.range[1]","yaxis.range[0]","yaxis.range[1]","xaxis.autorange","yaxis.autorange"])if(Object.hasOwn(event,key))update[key]=event[key];if(!Object.keys(update).length)return;syncing=true;Plotly.relayout(target,update).finally(()=>{syncing=false})})}
function factorPicker(){const picker=document.getElementById("factor-picker");picker.innerHTML="";for(let factor=0;factor<P.factor_names.length;factor++){const selected=state.factors.has(factor),button=document.createElement("button");button.type="button";button.className=`factor-choice${selected?" is-selected":""}`;button.title=`${P.factor_labels[factor]} · click to select alone; Ctrl/Cmd/Shift-click to toggle`;const circle=document.createElement("span");circle.className="factor-circle";button.appendChild(circle);const label=document.createElement("span");label.textContent=P.factor_labels[factor];button.appendChild(label);button.onclick=event=>{const additive=event.ctrlKey||event.metaKey||event.shiftKey;if(additive){if(selected&&state.factors.size>1)state.factors.delete(factor);else if(!selected)state.factors.add(factor)}else{state.factors.clear();state.factors.add(factor)}factorPicker();drawEmbedding()};picker.appendChild(button)}}
const color=document.getElementById("color"),xaxis=document.getElementById("xaxis"),yaxis=document.getElementById("yaxis");
for(const key of Object.keys(P.colorings))color.add(new Option(key,key));color.value=P.default_color_by;
for(let i=0;i<P.dimensions;i++){xaxis.add(new Option(modeLabel(i),i));yaxis.add(new Option(modeLabel(i),i))}xaxis.value="0";yaxis.value="1";
document.getElementById("summary").textContent=`${P.rows.toLocaleString()} ${P.population.replaceAll("_"," ")} · ${P.selected_dimensions} selected modes + ${P.dimensions-P.selected_dimensions} alternatives · generated ${P.generated_utc}`;
color.onchange=()=>{resetGroups();drawEmbedding();drawUmap()};xaxis.onchange=drawEmbedding;yaxis.onchange=drawEmbedding;
document.getElementById("hide-muted-groups").onchange=event=>{state.hideMuted.groups=event.target.checked;drawEmbedding()};
document.getElementById("hide-muted-factor").onchange=event=>{state.hideMuted.factor=event.target.checked;drawEmbedding()};
document.getElementById("hide-muted-umap").onchange=event=>{state.hideMuted.umap=event.target.checked;drawUmap()};
resetGroups();factorPicker();drawEmbedding();drawUmap();
</script></body></html>"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline", type=Path, required=True,
                        help="pipeline directory or manifest.json")
    parser.add_argument("--out-html", type=Path, required=True)
    parser.add_argument("--default-color-by",
                        choices=("partition", "cell-type"),
                        default="partition")
    metadata_arguments(parser)
    args = parser.parse_args()
    payload = build_payload(
        args.pipeline, args.metadata, args.metadata_id_column,
        args.cell_type_column, args.umap_x_column, args.umap_y_column,
        args.de, args.default_color_by)
    html = HTML_TEMPLATE.replace("__TITLE__", payload["title"]) \
        .replace("__PLOTLY_URL__", PLOTLY_URL) \
        .replace("__PAYLOAD__", json_for_html(payload))
    args.out_html.parent.mkdir(parents=True, exist_ok=True)
    args.out_html.write_text(html, encoding="utf-8")
    print(f"Wrote {args.out_html} ({payload['rows']} points)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
