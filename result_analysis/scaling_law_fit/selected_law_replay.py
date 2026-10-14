"""Replay R2 and mixture transfer before adopting the tied-coverage predictor."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from multilingual_megatron.experiments.scaling.stage1_fit.coverage_laws.replay import predict


def run(parent, selected, output):
    old = parent / "review/fresh_prediction_checks_v2"
    paths = [selected / "report/predictions.parquet", selected / "report/coefficients.json",
             old / "composition_inputs.parquet", old / "composition_fit_metrics.csv",
             parent / "recovery/v2/report/coefficients.json"]
    data = pd.read_parquet(paths[0]); rows = pd.read_parquet(paths[2])
    records = []
    for fold, g in data.query("arm == 'lbfgsb_box'").groupby('fold'):
        g = g[g.case.eq('L26' if fold == 'all_grid' else 'grid')]
        w = 1/g.groupby('trajectory_id').trajectory_id.transform('size').to_numpy(); w /= w.sum()
        y = g.adjusted_loss.to_numpy(); mse = np.dot(w, (y-g.prediction.to_numpy())**2)
        records.append(dict(fold=fold, rmse=np.sqrt(mse), r2=1-mse/np.dot(w,(y-np.dot(w,y))**2)))
    primary = ('largest_N_act','E32','K200_common_languages')
    metrics = []; predictions = []; rho = None
    for model, path in [('RN12-stored-common-K', paths[1]), ('RN14-stored-reuse', paths[4])]:
        fits = json.loads(path.read_text())['fits']
        for fold in primary:
            r, = [r['result']['selected'] for r in fits if r['model']==model and r['fold']==fold and r['arm']=='lbfgsb_box']
            p = rows.copy(); p['prediction'] = predict(model,r,p)+p.old_unigram_nll-p.unigram_nll
            p['model'],p['fold']=model,fold; predictions.append(p)
            for (seed,horizon), g in p.groupby(['seed','horizon']):
                wide=g.pivot(index=['E','language'],columns='arm',values=['prediction','adjusted_loss'])
                err=(wide.prediction.diverse-wide.prediction.compact)-(wide.adjusted_loss.diverse-wide.adjusted_loss.compact)
                metrics.append(dict(model=model,fold=fold,seed=seed,horizon=horizon,
                    absolute_rmse=float(np.sqrt(np.mean((g.prediction-g.adjusted_loss)**2))),contrast_rmse=float(np.sqrt(np.mean(err**2)))))
        if model=='RN12-stored-common-K':
            r, = [r['result']['selected'] for r in fits if r['model']==model and r['fold']=='all_grid' and r['arm']=='lbfgsb_box']
            rho=r['coordinates'][16]
    m=pd.DataFrame(metrics)
    prior=pd.read_csv(paths[3]); keys=['model','fold','seed','horizon']
    check=m[m.model.eq('RN14-stored-reuse')].merge(prior,on=keys,validate='one_to_one',suffixes=('','_old'))
    assert len(check)==54
    for field in ('absolute_rmse','contrast_rmse'):
        np.testing.assert_allclose(check[field],check[field+'_old'],atol=1e-10,rtol=0)
    output.mkdir(parents=True,exist_ok=False)
    pd.DataFrame(records).to_csv(output/'r2.csv',index=False)
    m.to_csv(output/'composition_metrics.csv',index=False)
    pd.concat(predictions).to_parquet(output/'composition_predictions.parquet',index=False)
    (output/'receipt.json').write_text(json.dumps(dict(inputs={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths+[Path(__file__)]},
        parent_composition_reproduced=True,all_grid_rho=rho,refitting=False),indent=2)+'\n')
    print(pd.DataFrame(records).to_string(index=False));print(m[m.horizon.eq(16e9)].groupby('model')[['absolute_rmse','contrast_rmse']].mean());print('rho',rho)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    for name in ('parent','selected','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.parent,a.selected,a.output)
