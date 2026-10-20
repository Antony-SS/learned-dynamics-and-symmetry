"""Create an interactive, self-contained five-second rollout visualization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from quadrotor.canonical import (
    absolute_dim,
    canonical_dim,
    load_prepared,
    output_dim,
)
from quadrotor.idsia_data import load_idsia_prepared
from quadrotor.rollout import open_loop
from quadrotor.sequence import MotorNet


class AbsoluteModel(MotorNet):
    reads_pose = True

    def __init__(self) -> None:
        super().__init__(absolute_dim(), output_dim())


class CanonicalModel(MotorNet):
    def __init__(self) -> None:
        super().__init__(canonical_dim(), output_dim())


class SymmetricModel(MotorNet):
    def __init__(self) -> None:
        super().__init__(canonical_dim(), output_dim(), mirror=True)


def _rounded(values: np.ndarray) -> list[list[float]]:
    return np.round(values.astype(float), 5).tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=Path("runs/quadrotor_gru_20ep"))
    parser.add_argument(
        "--output", type=Path, default=Path("reports/quadrotor_5s_rollout.html")
    )
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--idsia", action="store_true")
    args = parser.parse_args()

    _train, _val, test = (
        load_idsia_prepared() if args.idsia else load_prepared()
    )
    required = int(round(args.seconds / 0.01)) + 1
    candidates = [flight for flight in test if flight.n_poses >= required]
    flight = sorted(candidates, key=lambda item: item.n_poses)[len(candidates) // 2]
    constructors = {
        "absolute": AbsoluteModel,
        "canonical": CanonicalModel,
        "symmetric": SymmetricModel,
    }
    predictions = {}
    summaries = {}
    for name, constructor in constructors.items():
        model = constructor()
        model.load_state_dict(
            torch.load(args.run / f"{name}.pt", map_location="cpu", weights_only=True)
        )
        path, _rmse = open_loop(model, flight)
        path = path[:required]
        error = np.linalg.vector_norm(path - flight.position[:required], axis=1)
        predictions[name] = _rounded(path)
        summaries[name] = {
            "rmse": float(np.sqrt(np.mean(error**2))),
            "final": float(error[-1]),
            "max": float(error.max()),
        }

    data = {
        "flight": flight.name,
        "dt": 0.01,
        "seconds": args.seconds,
        "truth": _rounded(flight.position[:required]),
        "models": predictions,
        "summary": summaries,
    }
    cards = "".join(
        f"""<section class="model" data-model="{name}">
<div class="model-head"><div><h2>{name.capitalize()}</h2>
<span>real versus estimated, one initial Vicon seed</span></div>
<div class="stats"><b>{summary['rmse']:.3f} m</b> 5 s RMSE<br>
<b>{summary['final']:.3f} m</b> final error</div></div>
<div class="plots"><div><canvas class="xy" width="560" height="360"></canvas>
<div class="caption">XY trajectory (m)</div></div>
<div><canvas class="traces" width="700" height="360"></canvas>
<div class="caption">World position traces: solid = Vicon, dashed = estimate</div></div></div>
</section>"""
        for name, summary in summaries.items()
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Five-second quadrotor rollouts</title>
<style>
body{{margin:0;background:#f5f5f2;color:#171717;font:14px/1.45 system-ui,sans-serif}}
main{{max-width:1320px;margin:auto;padding:34px 22px 70px}}
h1{{font-size:30px;margin:0 0 5px}}h2{{margin:0;font-size:21px}}
.subtle,.caption,.model-head span{{color:#666}}.controls{{position:sticky;top:0;z-index:3;
background:#f5f5f2;border-bottom:1px solid #ccc;padding:16px 0;display:flex;gap:14px;align-items:center}}
button{{background:#171717;color:white;border:0;border-radius:5px;padding:8px 18px;font-weight:650}}
input{{flex:1}}output{{width:60px;font-variant-numeric:tabular-nums}}
.model{{background:white;border:1px solid #d8d8d8;border-radius:8px;padding:18px;margin-top:18px}}
.model-head{{display:flex;justify-content:space-between;align-items:start;margin-bottom:12px}}
.stats{{text-align:right;color:#555}}.stats b{{color:#5b21b6}}
.plots{{display:grid;grid-template-columns:4fr 5fr;gap:16px}}canvas{{width:100%;height:auto;background:#fff}}
.caption{{text-align:center;font-size:12px}}
@media(max-width:850px){{.plots{{grid-template-columns:1fr}}}}
</style></head><body><main>
<header><h1>Five-second open-loop rollout</h1>
<div class="subtle">Category B flight <b>{flight.name}</b> · predictions receive motor commands only after t=0</div></header>
<div class="controls"><button id="play">Play</button><input id="time" type="range" min="0" max="{required - 1}" value="0"><output id="clock">0.00 s</output></div>
{cards}
</main><script>
const D={json.dumps(data,separators=(",", ":"))};
const colors={{x:"#b91c1c",y:"#2563eb",z:"#15803d"}};
let playing=false,timer=null;
const slider=document.querySelector("#time"), clock=document.querySelector("#clock"), play=document.querySelector("#play");
function extent(rows,a,b){{let v=[];for(const r of rows){{v.push(r[a],r[b])}}return [Math.min(...v),Math.max(...v)]}}
function frame(ctx,w,h){{ctx.clearRect(0,0,w,h);ctx.strokeStyle="#ddd";ctx.lineWidth=1;ctx.strokeRect(42,15,w-57,h-50)}}
function line(ctx,rows,xfn,yfn,stroke,dash,stop=rows.length-1){{ctx.beginPath();ctx.strokeStyle=stroke;ctx.lineWidth=2;ctx.setLineDash(dash);
for(let i=0;i<=stop;i++){{const x=xfn(rows[i],i),y=yfn(rows[i],i);i?ctx.lineTo(x,y):ctx.moveTo(x,y)}}ctx.stroke();ctx.setLineDash([])}}
function drawXY(canvas,pred,k){{const ctx=canvas.getContext("2d"),w=canvas.width,h=canvas.height,all=D.truth.concat(pred);
const ex=extent(all,0,1),pad=.08*Math.max(ex[1]-ex[0],.1);let xmin=ex[0]-pad,xmax=ex[1]+pad;
let ys=all.flatMap(r=>[r[1]]),ymin=Math.min(...ys)-pad,ymax=Math.max(...ys)+pad;
const X=r=>42+(r[0]-xmin)/(xmax-xmin)*(w-57),Y=r=>h-35-(r[1]-ymin)/(ymax-ymin)*(h-50);
frame(ctx,w,h);line(ctx,D.truth,X,Y,"#111",[],k);line(ctx,pred,X,Y,"#7c3aed",[7,5],k);
for(const [rows,c] of [[D.truth,"#111"],[pred,"#7c3aed"]]){{ctx.fillStyle=c;ctx.beginPath();ctx.arc(X(rows[k]),Y(rows[k]),5,0,7);ctx.fill()}}
ctx.fillStyle="#555";ctx.font="12px system-ui";ctx.fillText("Vicon",50,31);ctx.fillStyle="#7c3aed";ctx.fillText("estimate",96,31)}}
function drawTraces(canvas,pred,k){{const ctx=canvas.getContext("2d"),w=canvas.width,h=canvas.height,all=D.truth.concat(pred);
let vals=all.flat(),ymin=Math.min(...vals),ymax=Math.max(...vals),pad=.08*Math.max(ymax-ymin,.1);ymin-=pad;ymax+=pad;
const X=(r,i)=>42+i/(D.truth.length-1)*(w-57),Y=(axis)=>(r)=>h-35-(r[axis]-ymin)/(ymax-ymin)*(h-50);
frame(ctx,w,h);for(let a=0;a<3;a++){{line(ctx,D.truth,(r,i)=>X(r,i),Y(a),colors["xyz"[a]],[],D.truth.length-1);
line(ctx,pred,(r,i)=>X(r,i),Y(a),colors["xyz"[a]],[6,4],D.truth.length-1)}}
const marker=42+k/(D.truth.length-1)*(w-57);ctx.strokeStyle="#111";ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(marker,15);ctx.lineTo(marker,h-35);ctx.stroke();
ctx.font="12px system-ui";for(const [i,a] of ["x","y","z"].entries()){{ctx.fillStyle=colors[a];ctx.fillText(a,52+i*25,31)}}
ctx.fillStyle="#555";ctx.fillText("time (s)",w-70,h-11)}}
function draw(){{const k=+slider.value;clock.value=(k*D.dt).toFixed(2)+" s";
document.querySelectorAll(".model").forEach(card=>{{const p=D.models[card.dataset.model];drawXY(card.querySelector(".xy"),p,k);drawTraces(card.querySelector(".traces"),p,k)}})}}
play.onclick=()=>{{playing=!playing;play.textContent=playing?"Pause":"Play";if(playing){{timer=setInterval(()=>{{let n=+slider.value+2;if(n>+slider.max)n=0;slider.value=n;draw()}},20)}}else clearInterval(timer)}};
slider.oninput=draw;draw();
</script></body></html>"""
    )
    print(f"visualization written to {args.output}")


if __name__ == "__main__":
    main()
