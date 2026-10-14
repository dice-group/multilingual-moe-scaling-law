"""Replay fixed predictors for the paper's existing estimators; never fit a law.

Paths are explicit CLI inputs. Outputs stay outside the manuscript and frozen
campaigns. PROTOCOL.md specifies the populations and comparisons in advance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq, minimize_scalar

NEW = ("RN14-stored-reuse", "RN13-psi-tied", "RN14-no-ridge")
OLD = ("M6-ND", "M5-ND", "M6")
FOLDS = ("largest_N_act", "E32", "K200_common_languages")
KEYS = ["trajectory_id", "readout_id", "language"]
COMPONENTS = ["D_t", "D_family", "D_script", "D_other"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


class Replay:
    def __init__(self, args):
        self.args = args
        self.out = args.output.resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        self.inputs = {}
        self.checks = {}
        self.ep = args.evidence.resolve()
        self.campaign = args.campaign.resolve()
        self.paper = args.paper.resolve()
        self.repo = Path(__file__).resolve().parents[2]
        sys.path[:0] = [str(self.campaign / "source"), str(self.repo / "src"), str(self.repo)]
        from stage1_fit.coverage_laws import replay as frozen
        from multilingual_megatron.experiments.scaling.language_composition.analysis import analysis_models, analysis_statistics
        from result_analysis.learning_curves import LearningCurve
        self.frozen, self.old_predict = frozen, analysis_models.predict
        self.comp_stats, self.LearningCurve = analysis_statistics, LearningCurve
        for p in (self.campaign / "source/stage1_fit").rglob("*.py"):
            self.bind(p)
        for module in (analysis_models, analysis_statistics):
            self.bind(Path(module.__file__))
        self.bind(self.repo / "result_analysis/learning_curves.py")
        self.bind(self.repo / "result_analysis/stage1_rn14_dependency_replay_20260921/PROTOCOL.md")
        self.bind(Path(__file__))
        current = {str(p.relative_to(self.paper)): sha(p) for p in self.paper.rglob("*")
                   if p.is_file() and "build" not in p.relative_to(self.paper).parts
                   and (p.suffix == ".tex" or (p.suffix == ".pdf" and "figures" in p.parts))}
        guard = self.out / "manuscript_before.json"
        if guard.exists():
            assert json.loads(guard.read_text()) == current, "manuscript changed during replay"
        else:
            write_json(guard, current)
        plan = self.read(self.campaign / "plan.json")
        self.selection = self.read(self.campaign / "selection.json")
        assert self.selection["chosen"]["form"] == "RN13-psi-tied"
        self.records = {}
        for fold in FOLDS:
            for model in NEW:
                r = self.read(self.campaign / f"results/certify--{fold}--{model}.json")
                assert r["certified"] and r["complete"] and r["plan_sha256"] == plan["sha256"]
                self.records[model, fold] = r["selected"]
        comp = self.ep / "composition_prediction/inputs"
        for record in self.read(comp / "models.json"):
            if record["model"] in OLD:
                self.records[record["model"], record["fold"]] = record["selected"]
        self.grid = self.read(args.scoring / "grid.parquet")
        self.saved = self.read(self.campaign / "report/predictions.parquet")
        self.context = self.read(self.ep / "primary_fits/policy_context.json")
        self.geometry = self.grid[["ladder", "E", "N_act", "N_total"]].drop_duplicates().sort_values(["N_total", "ladder", "E"])
        assert len(self.geometry) == 16
        self.geometry["config"] = self.geometry.ladder + "/E" + self.geometry.E.astype(str)
        for model in NEW:
            for fold in FOLDS:
                old = self.saved[(self.saved.model == model) & (self.saved.fold == fold) & (self.saved.case == "grid")]
                joined = old.merge(self.grid.drop(columns="adjusted_loss"), on=KEYS, validate="one_to_one")
                delta = float(np.max(abs(self.predict(model, fold, joined) - joined.prediction)))
                assert delta < 1e-10, (model, fold, delta)
                self.checks[f"saved_prediction_parity/{model}/{fold}"] = delta
        self.bind(self.campaign / "diagnostics/contrast_summary.csv")
        self.bind(self.campaign / "diagnostics/size_duration_summary.csv")
        self.refit = getattr(args, "refit_plan", None) is not None
        if self.refit:
            self.apply_refit(args.refit_plan, args.refit_report)

    def apply_refit(self, plan_path, report_path):
        """Use certified replacement coefficients with the SAME estimators."""
        from dataclasses import asdict
        from multilingual_megatron.experiments.scaling.stage1_fit import fit
        from multilingual_megatron.experiments.scaling.stage1_fit.paper_refit_workflow import inputs, runner, recovery
        if report_path is None:
            raise ValueError("refit replay requires an explicit report")
        plan_path, report_path = self.bind(plan_path), self.bind(report_path)
        plan, report = runner.load(plan_path), inputs.read_sealed(report_path)
        assert report["plan_sha256"] == plan["sha256"]
        for entry in report["outputs"]:
            self.bind(inputs.local_path(plan_path.parent, entry))
        recovery_path = inputs.local_path(plan_path.parent, report["recovery"]) if report.get("recovery") else None
        overrides = recovery.verified_overrides(plan_path, recovery_path)
        if recovery_path:
            self.bind(recovery_path)
        for key in self.records:
            model, fold = key
            task, = [t for t in plan["tasks"] if t["stage"] == "certify" and
                     (t["model"], t["fold"], t["arm"]) == (model, fold, "lbfgsb_box")]
            index = task["id"]
            receipt_path, record = overrides.get(index, (plan_path.parent / "results" / f"{index:05d}.json",
                                                          runner.receipt(plan_path, plan, index)))
            self.bind(receipt_path)
            assert record["certified"] and record["complete"]
            selected = dict(record["selected"])
            if model in OLD:
                selected["shared_parameters"] = asdict(fit.shared_from_outer("M6", selected["coordinates"][:14]))
                selected["zeta"] = selected["coordinates"][14] if model.endswith("-ND") else 0.
            self.records[key] = selected
        self.checks["refit_plan_sha256"] = plan["sha256"]
        self.checks["refit_report_sha256"] = report["sha256"]

    def bind(self, path):
        path = Path(path).resolve()
        self.inputs[str(path)] = sha(path)
        return path

    def read(self, path):
        path = self.bind(path)
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        if path.suffix in ("csv", ".csv") or path.name.endswith(".csv.gz"):
            return pd.read_csv(path)
        return json.loads(path.read_text())

    def predict(self, model, fold, rows):
        record = self.records[model, fold]
        if model in NEW:
            return self.frozen.predict(model, record, rows)
        return self.old_predict(rows, dict(model=model, selected=record))

    def save(self, name, frame):
        if name.endswith(".parquet"):
            frame.to_parquet(self.out / name, index=False)
        else:
            frame.to_csv(self.out / name, index=False)

    def finish(self, stage):
        for path, digest in self.inputs.items():
            assert sha(path) == digest, f"input changed: {path}"
        before = json.loads((self.out / "manuscript_before.json").read_text())
        assert all(sha(self.paper / p) == v for p, v in before.items())
        write_json(self.out / f"receipt_{stage}.json", dict(
            inputs=self.inputs, checks=self.checks, manuscript_unchanged=True,
            main_law="RN14-stored-reuse", historical_inner_selected=self.selection["chosen"]["form"],
            refits=0, neural_evaluations=0, training_runs=0,
            evidence="Developmental fixed-coefficient replay; budgets/languages/folds are not independent training runs."))

    def composition(self):
        inp = self.ep / "composition_prediction/inputs"
        spec = self.read(inp / "spec.json")
        losses = self.read(inp / "losses.parquet")
        exposure = self.read(inp / "exposures.parquet")
        rows = losses[losses.arm.isin(["compact", "diverse"]) & losses.language.isin(spec["shared_languages"])].copy()
        rows = rows.merge(exposure, on=["arm", "seed", "E", "horizon", "language"], validate="one_to_one")
        normal = pd.DataFrame(self.read(inp / "normalization.json"))
        rows = rows.merge(normal[["language", "old_unigram_nll"]], on="language", validate="many_to_one")
        rows = rows.merge(self.geometry[self.geometry.ladder.eq("B")][["E", "N_act", "N_total"]], on="E", validate="many_to_one")
        assert len(rows) == 756
        original = self.read(self.args.composition_original / "inputs/manifest.json")
        inventories = []
        for path, digest in original["source_bindings"].items():
            if Path(path).name != "megatron_data_manifest.json":
                continue
            assert sha(path) == digest
            manifest = self.read(path)
            arm = manifest["lineage"]["language_composition_arm"]["arm"]
            assert arm in ["compact", "diverse"]
            corpus_path = Path(manifest["lineage"]["corpus_manifest_path"])
            corpus = self.read(corpus_path)
            assert sha(corpus_path) == manifest["lineage"]["corpus_manifest_file_sha256"]
            hashes = self.read(corpus_path.parent / "hash_manifest.json")["files"]
            assert sha(corpus_path) == hashes["corpus_manifest.json"]
            for lang in spec["shared_languages"]:
                item = corpus["languages"][lang]["train"]
                binding = item["unigram_counts"]
                path = corpus_path.parent / binding["path"]
                count = self.read(path)
                assert sha(path) == binding["sha256"] == hashes[binding["path"]]
                assert count["language"] == lang and count["source_split"] == "train_only"
                assert count["tokenizer_model_sha256"] == corpus["inputs"]["tokenizer_model"]["sha256"]
                u = count["unique_train_tokens_including_special"]
                assert 0 < u <= count["exposure_train_tokens_including_special"] == item["selected_tokens"]
                assert u == count["non_special_token_count"] + count["excluded_special_token_count"]
                inventories.append(dict(arm=arm, language=lang, U_t=u,
                    physical_tokens=item["selected_tokens"], inventory_source=str(path)))
        inv = pd.DataFrame(inventories)
        assert len(inv) == 14 and not inv.duplicated(["arm", "language"]).any()
        rows = rows.merge(inv, on=["arm", "language"], validate="many_to_one")
        self.save("composition_inventories.csv", inv)
        self.save("composition_inputs.parquet", rows)
        predicted = []
        for model in (*NEW, *OLD):
            for fold in FOLDS:
                p = rows.copy()
                p["prediction"] = self.predict(model, fold, p) + p.old_unigram_nll - p.unigram_nll
                p["model"], p["fold"] = model, fold
                predicted.append(p)
        predicted = pd.concat(predicted, ignore_index=True)
        spec["models"] = list((*NEW, *OLD))
        metrics, seeds, contrasts, differences = self.comp_stats.prediction_metrics(predicted, spec)
        old = self.read(self.ep / "composition_prediction/derived/fit_metrics.csv")
        checks = metrics[metrics.model.isin(OLD)].merge(old, on=["model", "fold", "seed", "horizon"], suffixes=("", "_old"), validate="one_to_one")
        assert len(checks) == 162
        delta = max(float(abs(checks[k] - checks[k + "_old"]).max()) for k in ["contrast_rmse", "absolute_rmse", "interaction_rmse"])
        if not self.refit:
            assert delta < 1e-10, delta
        self.checks["composition_metric_change" if self.refit else "composition_historical_metric_parity"] = delta
        self.save("composition_historical_comparison.csv", checks)
        self.save("composition_predictions.parquet", predicted)
        self.save("composition_contrasts.parquet", contrasts)
        self.save("composition_fit_metrics.csv", metrics)
        self.save("composition_seed_metrics.csv", seeds)
        primary = seeds[seeds.horizon.eq(16_000_000_000)].groupby("model", as_index=False)[["contrast_rmse", "absolute_rmse", "interaction_rmse"]].mean()
        self.save("composition_primary.csv", primary)
        print("Composition at 16B:\n" + primary.to_string(index=False), flush=True)

    def policy(self):
        fold = "K200_common_languages"
        rows = self.read(self.ep / "primary_fits/policy_rows.parquet")
        u = self.grid[["K", "language", "U_t"]].drop_duplicates()
        rows = rows.merge(u, on=["K", "language"], validate="many_to_one")
        languages = sorted(self.context["cohorts"]["incumbent50"])
        bounds = rows.groupby("trajectory_id").agg(N=("N_act", "first"), low=("D_total", "min"), high=("D_total", "max"))
        budgets = np.geomspace((6 * bounds.N * bounds.low).max(), (6 * bounds.N * bounds.high).min(), 81)
        features, truth = [], []
        for row in self.geometry.itertuples():
            query = budgets / (6 * row.N_act)
            for lang in languages:
                g = rows[(rows.ladder == row.ladder) & (rows.E == row.E) & (rows.language == lang)].sort_values("D_total")
                assert query.min() >= g.D_total.min() * (1-1e-12) and query.max() <= g.D_total.max() * (1+1e-12)
                q = np.clip(query, g.D_total.min(), g.D_total.max())
                f = pd.DataFrame({c: np.interp(q, g.D_total, g[c]) for c in COMPONENTS})
                np.testing.assert_allclose(f[COMPONENTS].sum(axis=1), q, rtol=1e-12, atol=1e-6)
                f = f.assign(language=lang, K=200, E=row.E, N_act=row.N_act, N_total=row.N_total,
                             D_total=q, U_t=float(g.U_t.iloc[0]), config=row.config, FLOPs=budgets)
                features.append(f)
                loss = dict(linear_log=np.interp(np.log(q), np.log(g.D_total), g.adjusted_loss),
                    pchip_log=PchipInterpolator(np.log(g.D_total), g.adjusted_loss)(np.log(q)),
                    smooth=self.LearningCurve.fit(g.D_total.to_numpy(), g.adjusted_loss.to_numpy(), self.context["penalty"])(q))
                for method, vals in loss.items():
                    truth.append(pd.DataFrame(dict(config=row.config, language=lang, FLOPs=budgets, reference=method, observed=vals)))
        features = pd.concat(features, ignore_index=True)
        truth = pd.concat(truth, ignore_index=True)
        predictions = []
        for model in (*NEW, "M6-ND", "M5-ND"):
            p = features[["config", "language", "FLOPs", "N_total"]].copy()
            p["prediction"] = self.predict(model, fold, features)
            p["model"] = model
            predictions.append(p)
        predictions = pd.concat(predictions, ignore_index=True)
        self.save("policy_inputs.parquet", features)
        self.save("policy_predictions.parquet", predictions)
        self.save("policy_observed.parquet", truth)
        decisions = []
        for cohort, langs in self.context["cohorts"].items():
            obs = truth[truth.language.isin(langs)].groupby(["config", "FLOPs", "reference"], as_index=False)["observed"].mean()
            pred = predictions[predictions.language.isin(langs)].groupby(["model", "config", "FLOPs", "N_total"], as_index=False).prediction.mean()
            p = pred.merge(obs, on=["config", "FLOPs"], validate="many_to_many")
            for cap in (.4e9, .7e9, 2.2e9):
                for (model, ref, budget), g in p[p.N_total.le(cap)].groupby(["model", "reference", "FLOPs"]):
                    chosen = g.loc[g.prediction.idxmin()]
                    best = g.loc[g.observed.idxmin()]
                    decisions.append(dict(model=model, cohort=cohort, reference=ref, cap=cap, FLOPs=budget,
                        chosen=chosen.config, best=best.config, regret=chosen.observed-best.observed,
                        correct=chosen.config == best.config, candidate_count=len(g)))
        decisions = pd.DataFrame(decisions)
        metrics = decisions.groupby(["model", "cohort", "reference", "cap"], as_index=False).agg(
            mean_regret=("regret", "mean"), max_regret=("regret", "max"), correct_choices=("correct", "sum"),
            n_budgets=("regret", "size"), winner_accuracy=("correct", "mean"))
        old = self.read(self.ep / "original_analysis/policy_metrics.csv")
        checks = metrics[metrics.model.isin(["M6-ND", "M5-ND"])].merge(old, on=["model", "cohort", "reference", "cap"], suffixes=("", "_old"), validate="one_to_one")
        assert len(checks) == 36
        delta = max(float(abs(checks[k] - checks[k+"_old"]).max()) for k in ["mean_regret", "max_regret", "winner_accuracy"])
        if not self.refit:
            assert delta < 1e-9, delta
        self.checks["policy_metric_change" if self.refit else "policy_historical_metric_parity"] = delta
        self.save("policy_historical_comparison.csv", checks)
        self.save("policy_decisions.csv", decisions)
        self.save("policy_metrics.csv", metrics)
        print("Policy maximum regret across all caps/cohorts/interpolators:\n" + metrics.groupby("model").max_regret.max().to_string(), flush=True)

    def knots(self):
        fold = "K200_common_languages"
        old = self.read(self.ep / "combined_baseline/allocation_candidates.csv")
        inv = self.geometry[self.geometry.N_total.le(.4e9)]
        langs = self.context["cohorts"]["common12"]
        budgets = np.sort(old.FLOPs.unique())
        assert len(budgets) == 8
        grid = self.grid[self.grid.K.isin([12, 200]) & self.grid.language.isin(langs)].merge(
            inv[["ladder", "E", "config"]], on=["ladder", "E"], validate="many_to_one")
        old_truth = old[old.model.eq("M6-ND")].set_index(["config", "K", "FLOPs"]).observed
        candidates = []
        for model in (*NEW, "M6-ND", "M5-ND"):
            for (config, k, lang), g in grid.groupby(["config", "K", "language"]):
                g = g.sort_values("D_total").copy()
                saved_pred = self.predict(model, fold, g)
                query = budgets / (6*g.N_act.iloc[0])
                assert query.min() >= g.D_total.min()*(1-1e-12) and query.max() <= g.D_total.max()*(1+1e-12)
                query = np.clip(query, g.D_total.min(), g.D_total.max())
                f = pd.DataFrame({c: np.interp(query, g.D_total, g[c]) for c in COMPONENTS})
                for c in ["N_act", "N_total", "E", "K", "U_t", "language"]:
                    f[c] = g[c].iloc[0]
                for estimator, pred in [
                    ("saved_endpoint_interpolation", np.interp(np.log(f.D_t), np.log(g.D_t), saved_pred)),
                    ("direct_fixed_coefficients", self.predict(model, fold, f))]:
                    candidates.append(pd.DataFrame(dict(model=model, estimator=estimator, config=config,
                        K=k, language=lang, FLOPs=budgets, prediction=pred,
                        observed=np.interp(np.log(f.D_t), np.log(g.D_t), g.adjusted_loss))))
        candidates = pd.concat(candidates, ignore_index=True).groupby(["model", "estimator", "config", "K", "FLOPs"], as_index=False)[["prediction", "observed"]].mean()
        comp = candidates[candidates.model.eq("M6-ND") & candidates.estimator.eq("saved_endpoint_interpolation")].set_index(["config", "K", "FLOPs"])
        assert float(abs(comp.observed-old_truth).max()) < 1e-10
        oldpred = old[old.model.eq("M6-ND")].set_index(["config", "K", "FLOPs"]).predicted
        prediction_change = float(abs(comp.prediction.loc[oldpred.index]-oldpred).max())
        if not self.refit:
            assert prediction_change < 1e-10
        self.checks["knots_prediction_change"] = prediction_change
        decisions = []
        for (model, estimator, k, f), g in candidates.groupby(["model", "estimator", "K", "FLOPs"]):
            chosen, best = g.loc[g.prediction.idxmin()], g.loc[g.observed.idxmin()]
            decisions.append(dict(model=model, estimator=estimator, K=k, FLOPs=f,
                chosen=chosen.config, best=best.config, correct=chosen.config == best.config,
                regret=chosen.observed-best.observed))
        decisions = pd.DataFrame(decisions)
        metrics = decisions.groupby(["model", "estimator", "K"], as_index=False).agg(
            correct_choices=("correct", "sum"), n_budgets=("correct", "size"), max_regret=("regret", "max"), mean_regret=("regret", "mean"))
        self.save("knots_candidates.csv", candidates)
        self.save("knots_decisions.csv", decisions)
        self.save("knots_metrics.csv", metrics)
        self.checks["knots_observation_parity"] = True
        print("Eight-knot K200 decisions:\n" + metrics[metrics.K.eq(200)].to_string(index=False), flush=True)

    def continuous(self):
        """Search only N allocation; never optimize scaling coefficients."""
        fold = "E32"
        shares = self.read(self.args.fitted_source / "exposure_shares.csv")
        reduced = self.read(self.args.fitted_source / "reduced_mean_laws.csv").query("fold == 'E32'").set_index(["K", "E"])
        u = self.grid[["K", "language", "U_t"]].drop_duplicates()
        shares = shares.merge(u, on=["K", "language"], validate="one_to_one")
        anchors = self.geometry[self.geometry.E.eq(1)].sort_values("N_act").N_act.to_numpy()
        horizon = np.array([8e9, 16e9, 32e9, 32e9])
        predictors = {}
        repeat_boundaries = {k: (shares.loc[shares.K.eq(k), "D_t"] /
                                  shares.loc[shares.K.eq(k), "U_t"]).to_numpy()
                             for k in [1, 12, 200]}
        parity = 0.
        for model in (*NEW, "M6-ND"):
            for k in [1, 12, 200]:
                s = shares[shares.K.eq(k)].copy()
                assert len(s) == k
                cs = pd.DataFrame(self.records[model, fold]["language_coefficients"]).set_index("language").loc[s.language]
                for e in [1, 4, 16, 32]:
                    geom = self.geometry[self.geometry.E.eq(e)].sort_values("N_act")
                    gn, gm = geom.N_act.to_numpy(), geom.N_total.to_numpy()
                    if model == "M6-ND":
                        p = reduced.loc[k, e]
                        if self.refit:
                            from types import SimpleNamespace
                            from result_analysis.compute_allocation.law_allocation import coefficients
                            c = coefficients(self.records[model, fold], shares.set_index(["K", "language"]), k, e, s.language)
                            p = SimpleNamespace(base=float(np.mean(c[0]+s.unigram_nll.to_numpy())),
                                C=float(c[1].mean()), B=float(c[2].mean()), alpha=c[3], beta=c[4], zeta=c[5])
                        def loss(n, f, p=p):
                            n = np.atleast_1d(n)
                            d = f/(6*n*1e8)/1e9
                            return p.base+p.C*n**(-p.alpha)+p.B*n**p.zeta*d**(-p.beta)
                    else:
                        x = np.asarray(self.records[model, fold]["coordinates"])
                        h = np.log(e)/np.log(32.)
                        alpha = x[0]*(1-h)+x[1]*h+x[10]*max(0,np.log(k/12))/np.log(200/12)
                        beta = x[2]*(1-h)+x[3]*h
                        phi, psi = x[6]*(1-h)+x[7]*h, x[8]*(1-h)+x[9]*h
                        base = float(np.mean(cs.L_inf.to_numpy()+s.unigram_nll.to_numpy()))
                        cap = float(cs.A.mean())*e**x[4]*k**phi
                        b = cs.B.to_numpy()*e**x[5]*k**psi
                        share, inventory = s.D_t.to_numpy(), s.U_t.to_numpy()
                        def loss(n, f, base=base, cap=cap, b=b, alpha=alpha, beta=beta,
                                 x=x.copy(), share=share, inventory=inventory, model=model, gn=gn, gm=gm):
                            n = np.atleast_1d(n)
                            d = f/(6*n*1e8)
                            size = np.interp(n*1e8,gn,gm)/1e8 if model == "RN14-stored-reuse" else n
                            effective = self.frozen.saturate(d[:,None]*share, inventory[None,:], (x[15]*size**x[16])[:,None])/1e9
                            return base+cap*n**(-alpha)+n**x[14]*np.mean(b[None,:]*effective**(-beta),axis=1)
                    for n, d in [(1.35e8, 3e9), (5e8, 40e9)]:
                        rows = s.copy()
                        for c in COMPONENTS:
                            rows[c] *= d
                        rows = rows.assign(N_act=n, N_total=np.interp(n,gn,gm), E=e, D_total=d)
                        direct = float(np.mean(self.predict(model, fold, rows)+rows.unigram_nll))
                        error = abs(direct-float(loss(n/1e8,6*n*d)[0]))
                        assert error < 1e-10, (model,k,e,error)
                        parity = max(parity,error)
                    predictors[model,k,e] = loss
        self.checks["continuous_reduction_parity"] = parity
        def optimum(model,k,e,cap,f,dense=65):
            geom = self.geometry[self.geometry.E.eq(e)].sort_values("N_act")
            lower = anchors[0]/1e8
            upper = min(np.interp(cap,geom.N_total,geom.N_act),f/(6*.5e9))/1e8
            assert upper >= lower
            # Repetition onsets and stored-size interpolation knots divide the
            # objective into smooth pieces. Never let a local optimizer skip
            # one of these boundaries or assume the entire domain is unimodal.
            repeat = f*repeat_boundaries[k]/(6*1e8)
            boundaries = np.unique(np.r_[lower, upper,
                anchors[(anchors/1e8>lower)&(anchors/1e8<upper)]/1e8,
                repeat[(repeat>lower)&(repeat<upper)]])
            grid = np.unique(np.r_[np.geomspace(lower,upper,dense),boundaries])
            fun = predictors[model,k,e]
            values = fun(grid,f)
            options = [(float(values[i]),float(grid[i])) for i in np.unique(np.r_[0,len(grid)-1,np.argmin(values)])]
            minima = np.flatnonzero((values[1:-1]<=values[:-2])&(values[1:-1]<=values[2:]))+1
            intervals = []
            # A minimum can sit just inside a constraint even when the endpoint
            # is the best coarse sample. Explicitly refine boundary cells.
            if values[0] <= values[1]:
                intervals.append((grid[0],grid[1]))
            if values[-1] <= values[-2]:
                intervals.append((grid[-2],grid[-1]))
            for i in minima:
                options.append((float(values[i]),float(grid[i])))
                intervals.extend([(grid[i-1],grid[i]),(grid[i],grid[i+1])])
            for lo,hi in intervals:
                sol = minimize_scalar(lambda ln: float(fun(np.exp(ln),f)[0]),
                    bounds=(np.log(lo),np.log(hi)),method="bounded",options={"xatol":1e-10})
                assert sol.success
                options.append((float(sol.fun),float(np.exp(sol.x))))
            value,n = min(options)
            assert value <= float(values.min())+1e-10
            N = n*1e8
            D = f/(6*N)
            stored = float(np.interp(N,geom.N_act,geom.N_total))
            limit = float(np.exp(np.interp(np.log(N),np.log(anchors),np.log(horizon))))
            assert stored <= cap*(1+1e-10) and D >= .5e9*(1-1e-12)
            return dict(loss=value,N_act=N,N_total=stored,D_total=D,horizon_ratio=D/limit,extrapolated=bool(D>limit))
        records, switches, validation = [], [], []
        budgets = np.geomspace(1e18,1e23,401)
        for model in (*NEW,"M6-ND"):
            for k in [1,12,200]:
                for cap in [.4e9,1.2e9,5.1e9]:
                    es = [e for e in [1,4,16,32] if self.geometry[self.geometry.E.eq(e)].N_total.min()<=cap]
                    wins = []
                    for j,f in enumerate(budgets):
                        choices = []
                        for e in es:
                            r = optimum(model,k,e,cap,float(f))
                            records.append(dict(model=model,fold=fold,K=k,cap=cap,E=e,FLOPs=f,**r))
                            choices.append((r["loss"],e))
                            if j%100==0:
                                dense = optimum(model,k,e,cap,float(f),dense=257)
                                delta = abs(dense["loss"]-r["loss"])
                                assert delta < 1e-8, (model,k,cap,e,f,delta)
                                validation.append(dict(model=model,K=k,cap=cap,E=e,FLOPs=f,denser_grid_loss_gap=delta))
                        wins.append(min(choices)[1])
                    for i in np.flatnonzero(np.diff(wins)):
                        before,after = int(wins[i]),int(wins[i+1])
                        def gap(lf):
                            f=float(np.exp(lf))
                            return optimum(model,k,before,cap,f)["loss"]-optimum(model,k,after,cap,f)["loss"]
                        root=float(np.exp(brentq(gap,np.log(budgets[i]),np.log(budgets[i+1]),xtol=1e-11)))
                        a,b=optimum(model,k,before,cap,root),optimum(model,k,after,cap,root)
                        assert abs(a["loss"]-b["loss"])<1e-8
                        assert a["loss"]<=min(optimum(model,k,e,cap,root)["loss"] for e in es)+1e-8
                        switches.append(dict(model=model,K=k,cap=cap,before=before,after=after,FLOPs=root,
                            before_N_act=a["N_act"],after_N_act=b["N_act"],before_tokens=a["D_total"],after_tokens=b["D_total"],
                            before_extrapolated=a["extrapolated"],after_extrapolated=b["extrapolated"]))
                print(f"Continuous allocation completed: {model}, K={k}",flush=True)
        candidates=pd.DataFrame(records)
        winners=candidates.loc[candidates.groupby(["model","K","cap","FLOPs"]).loss.idxmin()]
        self.save("continuous_candidates.parquet",candidates)
        self.save("continuous_winners.csv",winners)
        self.save("continuous_switches.csv",pd.DataFrame(switches))
        self.save("continuous_numerical_checks.csv",pd.DataFrame(validation))
        self.checks["continuous_max_denser_grid_gap"] = max(x["denser_grid_loss_gap"] for x in validation)
        print("Continuous switches:\n"+pd.DataFrame(switches).to_string(index=False),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ["campaign","scoring","evidence","composition-original","fitted-source","paper","output"]:
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--stage",choices=["core","continuous","all"],default="all")
    p.add_argument("--refit-plan", type=Path)
    p.add_argument("--refit-report", type=Path)
    args=p.parse_args()
    work=Replay(args)
    if args.stage in ["core","all"]:
        work.composition()
        work.policy()
        work.knots()
        work.finish("core")
    if args.stage in ["continuous","all"]:
        work.continuous()
        work.finish("continuous")


if __name__=="__main__":
    main()
