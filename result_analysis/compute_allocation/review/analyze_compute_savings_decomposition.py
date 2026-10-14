"""Separate loss-gap/slope conversion from active-size selection at equal loss."""
import argparse
import json
import math
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from result_analysis.capacity_response.tables import sha256_file


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--savings',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    inputs=[(a.source,'native_mean_learning_curves.csv'),(a.savings,'equal_loss_compute_savings.csv')]
    for folder,name in inputs:
        assert sha256_file(folder/name)==json.loads((folder/'manifest.json').read_text())['outputs'][name]
    data=pd.read_csv(a.source/inputs[0][1]);savings=pd.read_csv(a.savings/inputs[1][1])
    data['F']=6.0*data.N_act.astype(float)*data.D_total.astype(float)/1e18
    curves={key:g.sort_values('F') for key,g in data.groupby(['K','ladder','E'])}
    def value(g,f):
        assert g.F.min()*(1-1e-12)<=f<=g.F.max()*(1+1e-12)
        return float(np.interp(np.log(f),np.log(g.F),g.mean_loss))
    def inverse(g,target):
        if target>g.mean_loss.iloc[0]:return None,'already_better_at_first_readout'
        if target<g.mean_loss.iloc[-1]:return None,'not_reached'
        f=float(np.exp(np.interp(target,g.mean_loss.to_numpy()[::-1],np.log(g.F.to_numpy()[::-1]))))
        assert abs(value(g,f)-target)<1e-11
        return f,'bracketed'
    def slopes(g,low,high):
        # Integrate independently over all intersected observed segments.
        xx=np.log(g.F.to_numpy());yy=g.mean_loss.to_numpy()
        lower,upper=math.log(low),math.log(high)
        widths=np.maximum(0.,np.minimum(xx[1:],upper)-np.maximum(xx[:-1],lower))
        slope=-(np.diff(yy)/np.diff(xx))
        assert abs(widths.sum()-(upper-lower))<1e-11
        average=float((widths*slope).sum()/widths.sum())
        idx=min(len(slope)-1,max(0,int(np.searchsorted(xx,upper,side='left')-1)))
        return average,float(slope[idx])
    rows=[]
    for r in savings.itertuples():
        f=float(r.dense_F_EF);fm=float(r.moe_F_EF);target=float(r.target_loss)
        assert r.status=='bracketed_equal_loss'
        selected=curves[r.K,r.moe_ladder,r.E]
        fixed=curves[r.K,r.dense_ladder,r.E]
        gap=target-value(selected,f)
        fixed_gap=target-value(fixed,f)
        avg,local=slopes(selected,fm,f)
        log_saved=math.log(f/fm)
        assert abs(gap-avg*log_saved)<1e-11
        ff,status=inverse(fixed,target)
        vals=dict(K=r.K,E=r.E,dense_F_EF=f,dense_ladder=r.dense_ladder,moe_ladder=r.moe_ladder,
            dense_loss=target,selected_moe_loss_at_dense_F=value(selected,f),selected_branch_loss_gap=gap,
            fixed_size_loss_gap=fixed_gap,size_selection_gap_change=gap-fixed_gap,
            effective_slope_nats_per_lnF=avg,local_slope_at_dense_F=local,
            actual_log_compute_saved=log_saved,actual_savings_percent=r.savings_percent,
            local_slope_approx_savings_percent=100*(1-math.exp(-gap/local)),
            selected_moe_F_EF=fm,fixed_size_inverse_status=status)
        if ff is not None:
            assert ff>=fm*(1-1e-10)
            fixed_slope,_=slopes(fixed,ff,f)
            assert abs(fixed_gap-fixed_slope*math.log(f/ff))<1e-11
            fixed_log=math.log(f/ff);size_log=math.log(ff/fm)
            assert abs(fixed_log+size_log-log_saved)<1e-11
            vals.update(fixed_size_moe_F_EF=ff,fixed_size_effective_slope=fixed_slope,
                fixed_size_savings_percent=100*(1-ff/f),
                size_selection_added_savings_pp=100*(ff-fm)/f,
                fixed_size_log_compute_saved=fixed_log,size_selection_log_compute_saved=size_log)
        rows.append(vals)
    result=pd.DataFrame(rows);result.to_csv(out/'savings_decomposition.csv',index=False)
    snap=result[result.dense_F_EF.isin([2.,4.])]
    snap.to_csv(out/'snapshots_2_and_4_EF.csv',index=False)
    comparisons=[]
    for e,g in result.groupby('E'):
        p=g.pivot(index='dense_F_EF',columns='K')
        for metric in ['selected_branch_loss_gap','effective_slope_nats_per_lnF','actual_savings_percent','fixed_size_savings_percent','size_selection_added_savings_pp']:
            delta=p[metric][200]-p[metric][12]
            comparisons.append(dict(E=int(e),metric=metric,K200_minus_K12_min=float(delta.min()),K200_minus_K12_max=float(delta.max()),mean_over_display_nodes=float(delta.mean())))
    pd.DataFrame(comparisons).to_csv(out/'K12_K200_comparison_ranges.csv',index=False)
    receipt=dict(input_hashes={str(folder/name):sha256_file(folder/name) for folder,name in inputs},
        builder_sha256=sha256_file(Path(__file__)),rows=len(result),inverse_status=result.fixed_size_inverse_status.value_counts().to_dict(),
        identities_checked=['segment-integrated slope times log compute saving equals selected loss gap',
                            'fixed-size and size-selection log savings add to total',
                            'fixed-size percent saving plus size-selection percentage points equals total'],
        local_approximation_max_error_pp=float((result.local_slope_approx_savings_percent-result.actual_savings_percent).abs().max()))
    (out/'verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(snap[(snap.E==32)&snap.dense_F_EF.eq(4)][['K','dense_ladder','moe_ladder','selected_branch_loss_gap','effective_slope_nats_per_lnF','actual_savings_percent','fixed_size_savings_percent','size_selection_added_savings_pp']].to_string(index=False))
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
