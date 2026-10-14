"""Review figures exposing both capacity axes; no manuscript assets are replaced."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FixedFormatter, MultipleLocator
import pandas as pd

E_VALUES = (1,4,16,32)
COLORS = dict(zip(E_VALUES, plt.get_cmap("inferno")([.02,.32,.62,.82])))
MARKERS = dict(zip(E_VALUES, ("o","s","^","D")))
TRANSITIONS = ((12,50,12),(12,200,12),(50,200,50))


def draw(directory: Path, averaged=False, common12=False):
    data = pd.read_csv(directory/"response_summary.csv").query("method=='linear_log'")
    fig, axes = plt.subplots(1 if averaged else 2, 3,
        figsize=(7.1,2.65 if averaged else 4.3), squeeze=False, sharex=True, sharey=True)
    stages = ("log_mean",) if averaged else ("early","late")
    for row, stage in enumerate(stages):
        for col, (a,b,nlang) in enumerate(TRANSITIONS):
            ax = axes[row,col]
            analysis = "common12_all_K" if common12 else f"K{a}_to_K{b}"
            q = data.query("analysis==@analysis and reference_K==@a and target_K==@b and summary==@stage")
            assert len(q)==16
            for e in E_VALUES:
                g = q[q.E.eq(e)].sort_values("N_act")
                ax.plot(g.N_act/1e6, g.response, color=COLORS[e], marker=MARKERS[e],
                    ms=3.0, mew=.35, lw=1.1, clip_on=False)
            if row==0:
                ax.set_title(rf"$K={a}\rightarrow {b}$"+f"  ({12 if common12 else nlang} languages)", fontsize=8.5,pad=6)
            ax.axhline(0,color="#737373",lw=.7,ls=(0,(3,2)),zorder=0)
            ax.set_xscale("log")
            xs = sorted(q.N_act.unique()/1e6)
            ax.xaxis.set_major_locator(FixedLocator(xs))
            ax.xaxis.set_major_formatter(FixedFormatter(("93","190","349","590")))
            ax.xaxis.set_minor_locator(FixedLocator([]))
            ax.set_xlim(83,650)
            ax.yaxis.set_major_locator(MultipleLocator(.05))
            ax.grid(axis="y",color="#ededed",lw=.45)
            ax.set_axisbelow(True)
            ax.spines[["top","right"]].set_visible(False)
            ax.spines[["left","bottom"]].set_linewidth(.55)
            ax.tick_params(labelsize=7.5,length=2.5,width=.5,pad=2)
    if averaged:
        axes[0,0].set_ylim(-.057,.162)
        fig.subplots_adjust(left=.095,right=.987,bottom=.24,top=.72,wspace=.18)
        fig.text(.10,.842,"Mean over each language's common log-exposure window",fontsize=8.3)
        fig.supxlabel(r"Active parameters $N_{\mathrm{act}}$ (millions)",fontsize=9,y=.07)
    else:
        axes[0,0].set_ylim(-.111,.196)
        fig.subplots_adjust(left=.095,right=.987,bottom=.15,top=.78,wspace=.18,hspace=.44)
        fig.text(.10,.852,"Earlier exposure: each language's common-window start",fontsize=8.5,weight="bold")
        fig.text(.10,.457,"Later exposure: each language's common-window end",fontsize=8.5,weight="bold")
        fig.supxlabel(r"Active parameters $N_{\mathrm{act}}$ (millions)",fontsize=9,y=.065)
    fig.supylabel(("Mean coverage response (nats)" if averaged else r"Coverage response $\Delta_K L$ (nats)"),fontsize=9,x=.015)
    fig.legend(handles=[Line2D([],[],color=COLORS[e],marker=MARKERS[e],lw=1.1,ms=3,
        label=rf"$E={e}$"+(" (dense)" if e==1 else "")) for e in E_VALUES],
        ncol=4,frameon=False,loc="upper center",bbox_to_anchor=(.54,.995),fontsize=8.5,
        handlelength=2,columnspacing=2.4)
    name = "joint_capacity_integrated" if averaged else "joint_capacity_early_late"
    if common12:
        name += "_common12"
    fig.savefig(directory/f"{name}.pdf",metadata={"CreationDate":None,"ModDate":None})
    fig.savefig(directory/f"{name}.png",dpi=240)
    plt.close(fig)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True, help="Directory produced by analyze.py")
    args = parser.parse_args()
    with plt.rc_context({"font.family":"STIXGeneral","mathtext.fontset":"stix", "font.size":8,
        "pdf.fonttype":42,"ps.fonttype":42}):
        draw(args.directory)
        draw(args.directory, averaged=True)
        draw(args.directory, common12=True)
