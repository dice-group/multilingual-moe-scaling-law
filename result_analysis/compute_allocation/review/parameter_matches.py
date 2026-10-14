"""Parameter-only pair inventory; no loss-based selection or extrapolation."""
from pathlib import Path
import argparse
from itertools import combinations
import json
import numpy as np
import pandas as pd
from result_analysis.capacity_response.tables import load_primary,sha256_file

ROOT=Path(__file__).resolve().parents[3]

def main(root: Path, output: Path):
 data=load_primary(root)
 output.mkdir(parents=True, exist_ok=False)
 recipes=data[['ladder','E','N_act','N_total']].drop_duplicates().sort_values('N_total')
 assert len(recipes)==16
 recipes['config']=recipes.ladder+'/E'+recipes.E.astype(str)
 cohort=sorted(data.loc[data.K.eq(12),'language'].unique());assert len(cohort)==12
 records=[];supports=[]
 for a,b in combinations(recipes.to_dict('records'),2):
  if a['N_act']==b['N_act'] or a['E']==b['E']:continue
  if a['N_act']>b['N_act']:a,b=b,a
  mismatch=100*(max(a['N_total'],b['N_total'])/min(a['N_total'],b['N_total'])-1)
  pair=a['config']+' vs '+b['config']
  record=dict(pair=pair,small_active_config=a['config'],large_active_config=b['config'],small_N_act=a['N_act'],large_N_act=b['N_act'],small_E=a['E'],large_E=b['E'],small_N_total=a['N_total'],large_N_total=b['N_total'],parameter_gap=abs(a['N_total']-b['N_total']),mismatch_pct=mismatch,tier='within 2%' if mismatch<=2 else '2–10%' if mismatch<=10 else 'over 10%')
  records.append(record)
  for k in [12,50,200]:
   intervals=[];selected=[]
   for recipe in [a,b]:
    sub=data.loc[data.ladder.eq(recipe['ladder'])&data.E.eq(recipe['E'])&data.K.eq(k)&data.language.isin(cohort)].copy()
    assert set(sub.language)==set(cohort)
    sub['F']=6*sub.N_act.astype(float)*sub.D_total.astype(float)
    bounds=sub.groupby('language').F.agg(['min','max'])
    intervals.append((float(bounds['min'].max()),float(bounds['max'].min())))
    selected.append(sub)
   lo=max(i[0] for i in intervals);hi=min(i[1] for i in intervals)
   supports.append(dict(pair=pair,K=k,common_languages=len(cohort),F_min=lo,F_max=hi,has_overlap=lo<=hi,small_readouts_in_overlap=selected[0].loc[selected[0].F.between(lo,hi)].readout_id.nunique(),large_readouts_in_overlap=selected[1].loc[selected[1].F.between(lo,hi)].readout_id.nunique()))
 pairs=pd.DataFrame(records).sort_values('mismatch_pct');support=pd.DataFrame(supports)
 joint=support.groupby('pair').agg(shared_F_min=('F_min','max'),shared_F_max=('F_max','min'))
 pairs=pairs.merge(joint,on='pair',validate='one_to_one')
 pairs['all_K_overlap']=pairs.shared_F_min<=pairs.shared_F_max
 pairs.to_csv(output/'all_cross_allocation_pairs.csv',index=False)
 matches=pairs.loc[pairs.mismatch_pct.le(10)].copy();matches.to_csv(output/'parameter_matches_within_10pct.csv',index=False)
 support.to_csv(output/'compute_support_by_pair_and_K.csv',index=False)
 recipes.to_csv(output/'architectures.csv',index=False)
 lines=['# Matched-parameter inventory','', 'Selection uses architecture parameter counts only; no losses are used. Different active sizes and expert counts are required. Mismatch = 100 × (larger stored count / smaller stored count − 1). The complete cross-allocation inventory is retained; ≤2% and ≤10% are descriptive tiers, not evidence of equivalence.','',f'16 architectures; {len(pairs)} cross-allocation pairs; {(pairs.mismatch_pct<=2).sum()} within 2%; {len(matches)} within 10%; no exact total-parameter matches.','', '| Pair (smaller active first) | Active sizes (M) | Stored sizes (B) | Mismatch | Shared compute (FLOPs) |','|---|---:|---:|---:|---:|']
 for r in matches.itertuples():
  lines.append(f'| {r.pair} | {r.small_N_act/1e6:.1f} / {r.large_N_act/1e6:.1f} | {r.small_N_total/1e9:.6f} / {r.large_N_total/1e9:.6f} | {r.mismatch_pct:.2f}% | {r.shared_F_min:.3e}–{r.shared_F_max:.3e} |')
 lines += ['', 'Compute intervals intersect both configurations and all three coverage settings K12/K50/K200, scoring the identical original 12 languages. Each language must have observations bracketing the interval. Comparable compute generally requires interpolation between cooled readouts, not identical checkpoint times. Per-configuration readout counts inside the overlap are in the support CSV. No losses or pairwise effects were calculated in this inventory.', '', 'The 9.732096B D/E32 configuration has no close stored-parameter counterpart at a different active size/expert count. Adding it as a matched pair would confound allocation with substantial stored-capacity differences. Wider-cap optimization can include it as a separate analysis.', '', 'Interpretation: the two ≤2% pairs provide the closest storage comparisons. The two additional 2–10% pairs are sensitivity comparisons, with residual capacity mismatch explicit. Three low-capacity pairs share architectures and are not independent replications. This inventory can support trends across tested pairs; it cannot identify a universal allocation law.']
 (output/'README.md').write_text('\n'.join(lines)+'\n')
 (output/'manifest.json').write_text(json.dumps(dict(generator_sha256=sha256_file(Path(__file__)),primary_release='stage1_abcd_jointmoe_x1_v1 (load_primary verifies sealed manifest)',cohort=cohort,pair_count=len(pairs),matches_within_2pct=int((pairs.mismatch_pct<=2).sum()),matches_within_10pct=len(matches)),indent=2)+'\n')
 print((output/'README.md').read_text())
 print('Nearest different-N/E counterpart for D/E32:')
 print(pairs.loc[pairs.small_active_config.eq('D/E32')|pairs.large_active_config.eq('D/E32')][['pair','mismatch_pct']].head(1).to_string(index=False))

if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--root', type=Path, default=ROOT, help='Repository containing the sealed primary release')
 parser.add_argument('--output', type=Path, required=True, help='New output directory; must not exist')
 args=parser.parse_args()
 main(args.root, args.output)
