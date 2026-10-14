"""Frozen K16-to-K12 migration contract constants."""

from multilingual_megatron.contracts.k12_adoption import (
    ADOPTION_SCHEMA_VERSION as SCHEMA_VERSION,
    K12_MIGRATION_TOOL_ID as TOOL_ID,
    UPSTREAM_ADOPTION_PATHS,
)

LANGUAGE_MAP_SCHEMA_VERSION = 2
OLD_TIERS = ("K1", "K16", "K50", "K200")
NEW_TIERS = ("K1", "K12", "K50", "K200")
SHARED_TIERS = ("K1", "K50", "K200")
REMOVED_FROM_K16 = {"ind_Latn", "nld_Latn", "swe_Latn", "tur_Latn"}
EXPECTED_INCREASED = {
    "cmn_Hani",
    "deu_Latn",
    "fra_Latn",
    "hin_Deva",
    "ita_Latn",
    "jpn_Jpan",
    "pol_Latn",
    "por_Latn",
    "rus_Cyrl",
    "spa_Latn",
    "vie_Latn",
}
EXPECTED_ACQUISITION = EXPECTED_INCREASED - {"rus_Cyrl", "vie_Latn"}
