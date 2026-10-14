from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from multilingual_megatron.experiments.scaling.stage1_fit.m6_composition import (
    M6CompositionError,
    build_composition_components,
    component_content_payload,
    component_content_sha256,
    composition_exposure_components,
    composition_sets,
    composition_sets_payload,
    composition_sets_sha256,
    language_metadata_payload,
    language_metadata_sha256,
    load_language_metadata,
)


LANGUAGE_EXPOSURES = {
    "eng_Latn": 10.0,
    "deu_Latn": 20.0,
    "rus_Cyrl": 30.0,
    "swh_Latn": 40.0,
    "kaz_Cyrl": 50.0,
    "hau_Latn": 60.0,
}


def _fit_frame() -> pd.DataFrame:
    records = []
    for trajectory, readout, multiplier in (
        ("trajectory-a", "readout-a", 1.0),
        ("trajectory-b", "readout-b", 2.0),
    ):
        total = sum(LANGUAGE_EXPOSURES.values()) * multiplier
        for language, exposure in LANGUAGE_EXPOSURES.items():
            records.append(
                {
                    "trajectory_id": trajectory,
                    "readout_id": readout,
                    "language": language,
                    "K": len(LANGUAGE_EXPOSURES),
                    "D_total": total,
                    "D_t": exposure * multiplier,
                    "adjusted_loss": -1.0,
                }
            )
    return pd.DataFrame.from_records(records)


def _inventory_text(*, eng_script: str = "Latn") -> str:
    return "\n".join(
        (
            "subset,split,script,family",
            f"eng_Latn,train,{eng_script},Indo-European",
            "eng_Latn,test,Latn,Indo-European",
            "deu_Latn,train,Latn,Indo-European",
            "rus_Cyrl,train,Cyrl,Indo-European",
            "swh_Latn,train,Latn,Niger-Congo",
            "kaz_Cyrl,train,Cyrl,Turkic",
            "kaz_Cyrl_removed,train,Cyrl,Turkic",
            "",
        )
    )


class Stage1M6CompositionTest(unittest.TestCase):
    def _inventory(self, root: Path, *, eng_script: str = "Latn") -> Path:
        path = root / "fineweb2-language-distribution.csv"
        path.write_text(_inventory_text(eng_script=eng_script), encoding="utf-8")
        return path

    def test_metadata_and_memberships_are_loss_blind_and_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = self._inventory(Path(temporary))
            metadata = load_language_metadata(path, LANGUAGE_EXPOSURES)

        self.assertEqual(list(metadata), sorted(LANGUAGE_EXPOSURES))
        self.assertEqual(metadata["hau_Latn"], {"script": "Latn", "family": None})
        memberships = composition_sets(metadata)
        self.assertEqual(
            memberships["eng_Latn"],
            {
                "family": ("deu_Latn", "rus_Cyrl"),
                "script": ("hau_Latn", "swh_Latn"),
            },
        )
        self.assertEqual(
            memberships["hau_Latn"],
            {
                "family": (),
                "script": ("deu_Latn", "eng_Latn", "swh_Latn"),
            },
        )

        reversed_metadata = dict(reversed(tuple(metadata.items())))
        self.assertEqual(language_metadata_payload(reversed_metadata), metadata)
        self.assertEqual(
            language_metadata_sha256(reversed_metadata),
            language_metadata_sha256(metadata),
        )
        reordered_memberships = {
            target: {
                "family": tuple(reversed(value["family"])),
                "script": tuple(reversed(value["script"])),
            }
            for target, value in reversed(tuple(memberships.items()))
        }
        self.assertEqual(
            composition_sets_payload(reordered_memberships),
            composition_sets_payload(memberships),
        )
        self.assertEqual(
            composition_sets_sha256(reordered_memberships),
            composition_sets_sha256(memberships),
        )

    def test_components_are_disjoint_nonnegative_and_reconstruct_total(self) -> None:
        frame = _fit_frame().sample(frac=1.0, random_state=11)
        with tempfile.TemporaryDirectory() as temporary:
            path = self._inventory(Path(temporary))
            components, metadata, memberships = build_composition_components(
                frame, path
            )

        np.testing.assert_array_equal(components["D_target"], frame["D_t"])
        np.testing.assert_allclose(
            components.sum(axis=1), frame["D_total"], rtol=0.0, atol=0.0
        )
        self.assertTrue((components >= 0.0).to_numpy().all())
        eng = frame.index[
            (frame["readout_id"] == "readout-a")
            & (frame["language"] == "eng_Latn")
        ][0]
        self.assertEqual(components.loc[eng].to_dict(), {
            "D_target": 10.0,
            "D_family": 50.0,
            "D_script": 100.0,
            "D_other": 50.0,
        })
        hau = frame.index[
            (frame["readout_id"] == "readout-a")
            & (frame["language"] == "hau_Latn")
        ][0]
        self.assertEqual(components.loc[hau].to_dict(), {
            "D_target": 60.0,
            "D_family": 0.0,
            "D_script": 70.0,
            "D_other": 80.0,
        })
        self.assertIsNone(metadata["hau_Latn"]["family"])
        self.assertEqual(memberships["hau_Latn"]["family"], ())

        changed = frame.copy()
        changed["adjusted_loss"] = np.arange(len(changed), dtype=np.float64) * 1e6
        changed_components, changed_memberships = composition_exposure_components(
            changed, metadata
        )
        pd.testing.assert_frame_equal(changed_components, components)
        self.assertEqual(changed_memberships, memberships)

    def test_component_content_is_canonical_and_validated(self) -> None:
        frame = _fit_frame()
        with tempfile.TemporaryDirectory() as temporary:
            path = self._inventory(Path(temporary))
            components, _, _ = build_composition_components(frame, path)
        derived = frame.copy()
        for column in ("D_family", "D_script", "D_other"):
            derived[column] = components[column]

        shuffled = derived.sample(frac=1.0, random_state=5)
        self.assertEqual(
            component_content_payload(shuffled), component_content_payload(derived)
        )
        self.assertEqual(
            component_content_sha256(shuffled), component_content_sha256(derived)
        )

        negative = derived.copy()
        negative.loc[negative.index[0], "D_family"] = -1.0
        with self.assertRaisesRegex(M6CompositionError, "non-negative"):
            component_content_payload(negative)
        broken = derived.copy()
        broken.loc[broken.index[0], "D_other"] += 1.0
        with self.assertRaisesRegex(M6CompositionError, "reconstruct"):
            component_content_payload(broken)

    def test_invalid_inventory_and_mixture_fail_closed(self) -> None:
        frame = _fit_frame()
        with tempfile.TemporaryDirectory() as temporary:
            bad_inventory = self._inventory(Path(temporary), eng_script="Cyrl")
            with self.assertRaisesRegex(M6CompositionError, "script disagrees"):
                load_language_metadata(bad_inventory, frame["language"])

        duplicate = pd.concat((frame, frame.iloc[[0]]), ignore_index=True)
        metadata = {
            language: {
                "script": language.rpartition("_")[2],
                "family": None,
            }
            for language in LANGUAGE_EXPOSURES
        }
        with self.assertRaisesRegex(M6CompositionError, "duplicate"):
            composition_exposure_components(duplicate, metadata)
        incomplete = frame.drop(index=frame.index[0])
        with self.assertRaisesRegex(M6CompositionError, "declared K"):
            composition_exposure_components(incomplete, metadata)
        inconsistent = frame.copy()
        inconsistent.loc[inconsistent.index[0], "D_t"] += 1.0
        with self.assertRaisesRegex(M6CompositionError, "do not sum"):
            composition_exposure_components(inconsistent, metadata)


if __name__ == "__main__":
    unittest.main()
