"""Opt-in fixtures for the historical ATLAS recovery; never submit real jobs."""

from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

from multilingual_megatron.experiments.scaling.stage1_fit.law_search import atlas_followup as f


class AtlasFollowupTest(unittest.TestCase):
    def test_extension_is_separate_from_original_budget(self):
        self.assertEqual(f.RULES["minutes_per_model"],24)
        self.assertEqual(f.EXTENDED_RULES["minutes_per_model"],720)
        self.assertEqual(f.extended_budget(11),132.25)
        with self.assertRaises(ValueError):f.extended_budget(12)
        plan=dict(profile=dict(python_executable="/env/bin/python",account="fixture",partition="normal",memory="4G"),
            rules=f.EXTENDED_RULES,tasks=[{}]*11)
        script=f.script(Path("/tmp/extended/plan.json"),plan)
        self.assertIn("--time=720",script)
        self.assertIn("--array=0-10%11",script)
        self.assertIn("--time=15",f.script(Path("/tmp/extended/plan.json"),plan,True))
        subprocess.run(["bash","-n"],input=script,text=True,check=True)

    def test_expired_parent_ids_do_not_bypass_active_job_checks(self):
        with patch.object(f.subprocess,"check_output",return_value="123_[1-4]\n999_0\n"):
            self.assertEqual(f.active_parent_jobs(["123","456"]),{"123"})
            self.assertEqual(f.active_parent_jobs(["456"]),set())
        with patch.object(f.subprocess,"check_output",side_effect=subprocess.CalledProcessError(1,"squeue")):
            with self.assertRaises(subprocess.CalledProcessError):f.active_parent_jobs(["123"])

    def test_panel_uses_registered_forms_not_unrelated_experimental_extensions(self):
        specs=[s for s in f.m.panel() if s.form!="moe"]
        parent=dict(native_specs=[s.payload() for s in specs],unresolved=[])
        selected=f.registered_specs(parent)
        self.assertEqual(len(selected),49)
        self.assertFalse(any(s.form=="moe" for s in selected.values()))

    def test_budget_retains_prior_full_reservations(self):
        prior=dict(charged_core_hours=458.25,phase_charged_core_hours=dict(baselines=187.25))
        self.assertAlmostEqual(f.budget(29,prior),11.85)
        with self.assertRaises(ValueError):f.budget(30,prior)
        with self.assertRaises(ValueError):f.budget(29,prior|{"charged_core_hours":501.})
        with self.assertRaises(ValueError):f.budget(29,prior|{"phase_charged_core_hours":{"baselines":190.}})

    def test_certified_groups_are_never_refitted(self):
        group=dict(certified=True,selected={"coordinates":[1.]})
        with patch.object(f.n,"fit_group",side_effect=AssertionError("already certified")), \
             patch.object(f.recovery,"recover",side_effect=AssertionError("already certified")):
            self.assertIs(f.recover_group(None,{},group),group)

    def test_existing_failure_uses_training_recovery_not_new_multistart(self):
        previous=dict(certified=False,attempts=[{"index":0}],language="fixture")
        recovered=dict(certified=True,selected={"coordinates":[2.]})
        packet={"y":[1.]}
        with patch.object(f.n,"fit_group",side_effect=AssertionError("unnecessary refit")), \
             patch.object(f.recovery,"recover",return_value=recovered) as recover:
            value=f.recover_group(None,packet,previous)
        self.assertTrue(value["certified"])
        self.assertEqual(value["prior_fit"],previous)
        recover.assert_called_once_with(None,packet,packet,previous)

    def test_new_variants_need_native_fit_and_failed_audits_stay_failed(self):
        old=dict(certified=False,spec={})
        native=dict(certified=False,attempts=[{"index":0}])
        with patch.object(f.n,"fit_group",return_value=native) as initial, \
             patch.object(f.recovery,"recover",return_value=dict(certified=False)):
            value=f.recover_group(None,{},old)
        initial.assert_called_once()
        self.assertFalse(value["certified"])

    def test_cpu_only_array_and_report_scripts(self):
        plan=dict(profile=dict(python_executable="/env/bin/python",account="fixture",partition="normal",memory="4G"),
            tasks=[{}]*29)
        for report in (False,True):
            script=f.script(Path("/tmp/atlas followup/plan.json"),plan,report)
            subprocess.run(["bash","-n"],input=script,text=True,check=True)
            self.assertIn("--cpus-per-task=1",script)
            self.assertNotIn("--gres",script)
            self.assertNotIn("sbatch ",script)
            self.assertEqual("--array=0-28%29" in script,not report)


if __name__=="__main__":unittest.main()
