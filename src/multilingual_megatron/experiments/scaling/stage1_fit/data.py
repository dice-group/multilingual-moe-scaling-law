from __future__ import annotations
import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

CONTRACT_VERSION = 'multilingual_megatron_stage1_scaling_fit_d1_d17/v1'
FIT_ROWS_SCHEMA_VERSION = 'multilingual_megatron_stage1_fit_rows/v1'
INPUT_RECEIPT_SCHEMA_VERSION = 'multilingual_megatron_stage1_fit_input_receipt/v1'
RELEASE_CONTRACT_SCHEMA_VERSION = 'multilingual_megatron_stage1_fit_release_contract/v1'
HOLDOUT_MANIFEST_SCHEMA_VERSION = 'multilingual_megatron_stage1_fit_holdouts/v1'
RESAMPLING_MANIFEST_SCHEMA_VERSION = 'multilingual_megatron_stage1_fit_resampling/v1'
RELEASE_ID = 'stage1_abcd_v1'
RELEASE_SCHEMA = 'multilingual_megatron_stage1_abcd_release/v1'
EXPECTED_RELEASE_MANIFEST_FILE_SHA256 = '3f87fe7d3f9412edc9134b6e3fd18584abc7cd25d645808e51839ed398c709b5'
EXPECTED_RELEASE_MANIFEST_INTEGRITY_SHA256 = '0c8d3977d34d9c50c3562fd42c784a4435d7e5e0c8cf5556b923694879aff0e9'
EXPECTED_TRAJECTORIES = 64
EXPECTED_READOUTS = 400
EXPECTED_FIT_ROWS = 26300
EXPECTED_EXPERT_LEVELS = (1, 4, 16, 32)
EXPECTED_LANGUAGE_COUNT_LEVELS = (1, 12, 50, 200)
EXPECTED_LADDERS = ('A', 'B', 'C', 'D')
REQUIRED_SOURCE_TABLES = ('trajectories', 'readouts', 'model_configs', 'validation_language', 'data_exposure')
PRIMARY_METRIC_ROLE = 'trained'
PRIMARY_METRIC_NAME = 'trained_equal_language_atlas_tao_unigram_adjusted_loss'
RANDOM_HOLDOUT_SEED = 20260803
UNCERTAINTY_SUBSAMPLE_SEED = 20260804
PAIRED_BOOTSTRAP_SEED = 20260805
RANDOM_HOLDOUT_TRAJECTORIES = 13
UNCERTAINTY_REPLICATES = 100
UNCERTAINTY_SUBSET_TRAJECTORIES = 51
PAIRED_BOOTSTRAP_REPLICATES = 10000
DEFAULT_RELEASE_RELATIVE_PATH = Path('result_analysis/stage1_abcd_v1')
DEFAULT_ARCHIVED_MANIFEST_RELATIVE_PATH = Path('outputs/pipeline_runs/stage1_ad_poststudy_20260730/control/archived_result_records/result_analysis/stage1_abcd_v1/release_manifest.json')
FIT_ROW_COLUMNS = ('trajectory_id', 'readout_id', 'language', 'N_act', 'N_act_nonembedding', 'E', 'K', 'D_total', 'D_t', 'adjusted_loss', 'raw_nll', 'validation_token_count', 'unigram_nll', 'ladder', 'model_seed')

class Stage1FitContractError(ValueError):
    """Raised when data or statistics violate the frozen D1--D16 contract."""

@dataclass(frozen=True)
class ReleaseContract:
    """Exact immutable release identity accepted by the fitter."""
    release_id: str
    release_schema: str
    archive_manifest_file_sha256: str
    release_manifest_integrity_sha256: str
    expected_trajectories: int
    expected_readouts: int
    expected_fit_rows: int
    required_source_tables: tuple[str, ...]
    contract_sha256: str
    schema_version: str = RELEASE_CONTRACT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload['required_source_tables'] = list(self.required_source_tables)
        return payload

@dataclass(frozen=True)
class FitRow:
    """One immutable per-language checkpoint observation (D1--D4)."""
    trajectory_id: str
    readout_id: str
    language: str
    N_act: float
    N_act_nonembedding: float
    E: int
    K: int
    D_total: float
    D_t: float
    adjusted_loss: float
    raw_nll: float
    validation_token_count: int
    unigram_nll: float
    ladder: str
    model_seed: int

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical table spelling of this record."""
        return {name: getattr(self, name) for name in FIT_ROW_COLUMNS}

@dataclass(frozen=True)
class SourceTableReceipt:
    """Verified identity of one canonical Parquet input table."""
    name: str
    path: str
    rows: int
    sha256: str
    manifest_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass(frozen=True)
class InputReceipt:
    """Machine-readable receipt for one written immutable fit snapshot."""
    release_root: str
    archive_manifest_path: str
    archive_manifest_file_sha256: str
    release_manifest_integrity_sha256: str
    trajectories: int
    readouts: int
    fit_rows: int
    fit_rows_path: str
    fit_rows_file_sha256: str
    fit_rows_content_sha256: str
    source_tables: tuple[SourceTableReceipt, ...]
    schema_version: str = INPUT_RECEIPT_SCHEMA_VERSION
    contract_version: str = CONTRACT_VERSION
    release_id: str = RELEASE_ID
    release_schema: str = RELEASE_SCHEMA
    fit_rows_schema_version: str = FIT_ROWS_SCHEMA_VERSION
    release_contract_path: str | None = None
    release_contract_file_sha256: str | None = None
    release_contract_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload['source_tables'] = [item.to_dict() for item in self.source_tables]
        for field in (
            'release_contract_path',
            'release_contract_file_sha256',
            'release_contract_sha256',
        ):
            if payload[field] is None:
                payload.pop(field)
        return payload

def canonical_json_bytes(payload: Any) -> bytes:
    """Serialize JSON deterministically and reject NaN or infinity."""
    try:
        encoded = json.dumps(payload, allow_nan=False, ensure_ascii=False, separators=(',', ':'), sort_keys=True)
    except (TypeError, ValueError) as error:
        raise Stage1FitContractError(f'payload is not canonical finite JSON: {error}') from error
    return encoded.encode('utf-8')

def canonical_json_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()

def sha256_file(path: Path) -> str:
    """Hash a file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def _require_sha256(value: Any, *, field: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(character not in '0123456789abcdef' for character in digest):
        raise Stage1FitContractError(f'{field} is not a lowercase SHA-256 digest')
    return digest

def _contract_unsigned(contract: ReleaseContract | Mapping[str, Any]) -> dict[str, Any]:
    payload = contract.to_dict() if isinstance(contract, ReleaseContract) else dict(contract)
    payload.pop('contract_sha256', None)
    return payload

def historical_release_contract() -> ReleaseContract:
    unsigned = {
        'schema_version': RELEASE_CONTRACT_SCHEMA_VERSION,
        'release_id': RELEASE_ID,
        'release_schema': RELEASE_SCHEMA,
        'archive_manifest_file_sha256': EXPECTED_RELEASE_MANIFEST_FILE_SHA256,
        'release_manifest_integrity_sha256': EXPECTED_RELEASE_MANIFEST_INTEGRITY_SHA256,
        'expected_trajectories': EXPECTED_TRAJECTORIES,
        'expected_readouts': EXPECTED_READOUTS,
        'expected_fit_rows': EXPECTED_FIT_ROWS,
        'required_source_tables': list(REQUIRED_SOURCE_TABLES),
    }
    return ReleaseContract(
        release_id=RELEASE_ID,
        release_schema=RELEASE_SCHEMA,
        archive_manifest_file_sha256=EXPECTED_RELEASE_MANIFEST_FILE_SHA256,
        release_manifest_integrity_sha256=EXPECTED_RELEASE_MANIFEST_INTEGRITY_SHA256,
        expected_trajectories=EXPECTED_TRAJECTORIES,
        expected_readouts=EXPECTED_READOUTS,
        expected_fit_rows=EXPECTED_FIT_ROWS,
        required_source_tables=REQUIRED_SOURCE_TABLES,
        contract_sha256=canonical_json_sha256(unsigned),
    )

def load_release_contract(path: Path | str) -> ReleaseContract:
    contract_path = Path(path).resolve()
    try:
        payload = json.loads(contract_path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as error:
        raise Stage1FitDataError(f'cannot read release contract: {error}') from error
    if not isinstance(payload, dict):
        raise Stage1FitDataError('release contract must be a JSON object')
    expected = {
        'schema_version', 'release_id', 'release_schema',
        'archive_manifest_file_sha256', 'release_manifest_integrity_sha256',
        'expected_trajectories', 'expected_readouts', 'expected_fit_rows',
        'required_source_tables', 'contract_sha256',
    }
    require_exact_keys(payload, expected, context='release contract')
    if payload.get('schema_version') != RELEASE_CONTRACT_SCHEMA_VERSION:
        raise Stage1FitDataError('release contract schema changed')
    declared = _require_sha256(payload.get('contract_sha256'), field='contract_sha256')
    if canonical_json_sha256(_contract_unsigned(payload)) != declared:
        raise Stage1FitDataError('release contract integrity mismatch')
    tables = payload.get('required_source_tables')
    if not isinstance(tables, list) or tuple(str(value) for value in tables) != REQUIRED_SOURCE_TABLES:
        raise Stage1FitDataError('release contract source-table set changed')
    counts = (
        int(payload.get('expected_trajectories', -1)),
        int(payload.get('expected_readouts', -1)),
        int(payload.get('expected_fit_rows', -1)),
    )
    if counts != (EXPECTED_TRAJECTORIES, EXPECTED_READOUTS, EXPECTED_FIT_ROWS):
        raise Stage1FitDataError(f'release contract canonical counts changed: {counts}')
    return ReleaseContract(
        release_id=str(payload['release_id']),
        release_schema=str(payload['release_schema']),
        archive_manifest_file_sha256=_require_sha256(
            payload['archive_manifest_file_sha256'], field='archive_manifest_file_sha256'
        ),
        release_manifest_integrity_sha256=_require_sha256(
            payload['release_manifest_integrity_sha256'], field='release_manifest_integrity_sha256'
        ),
        expected_trajectories=counts[0],
        expected_readouts=counts[1],
        expected_fit_rows=counts[2],
        required_source_tables=tuple(str(value) for value in tables),
        contract_sha256=declared,
    )

def _canonical_float(value: Any, field: str) -> str:
    number = float(value)
    if not math.isfinite(number):
        raise Stage1FitContractError(f'non-finite {field} in canonical fit rows')
    return number.hex()

def fit_rows_content_sha256(rows: tuple[FitRow, ...] | list[FitRow]) -> str:
    """Hash canonical fit-row content independently of Parquet encoding/order."""
    ordered = sorted(rows, key=lambda row: (row.trajectory_id, row.readout_id, row.language))
    keys = [(row.trajectory_id, row.readout_id, row.language) for row in ordered]
    if len(keys) != len(set(keys)):
        raise Stage1FitContractError('duplicate canonical fit-row keys')
    digest = hashlib.sha256()
    header = {'schema_version': FIT_ROWS_SCHEMA_VERSION, 'columns': list(FIT_ROW_COLUMNS)}
    header_bytes = canonical_json_bytes(header)
    digest.update(len(header_bytes).to_bytes(8, 'big'))
    digest.update(header_bytes)
    float_fields = {'N_act', 'N_act_nonembedding', 'D_total', 'D_t', 'adjusted_loss', 'raw_nll', 'unigram_nll'}
    integer_fields = {'E', 'K', 'validation_token_count', 'model_seed'}
    for row in ordered:
        record: list[Any] = []
        for field in FIT_ROW_COLUMNS:
            value = getattr(row, field)
            if field in float_fields:
                record.append(_canonical_float(value, field))
            elif field in integer_fields:
                record.append(int(value))
            else:
                record.append(str(value))
        encoded = canonical_json_bytes(record)
        digest.update(len(encoded).to_bytes(8, 'big'))
        digest.update(encoded)
    return digest.hexdigest()

def require_exact_keys(payload: Mapping[str, Any], expected: set[str], *, context: str) -> None:
    """Fail closed when a typed boundary gains or loses fields."""
    actual = set(payload)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise Stage1FitContractError(f'{context} keys changed; missing={missing}, extra={extra}')
REPOSITORY_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_RELEASE_ROOT = REPOSITORY_ROOT / DEFAULT_RELEASE_RELATIVE_PATH
DEFAULT_ARCHIVED_MANIFEST_PATH = REPOSITORY_ROOT / DEFAULT_ARCHIVED_MANIFEST_RELATIVE_PATH

class Stage1FitDataError(Stage1FitContractError):
    """Raised when the sealed input cannot satisfy the fit-data contract."""

@dataclass(frozen=True)
class PreparedFitData:
    """Immutable in-memory rows plus verified source provenance."""
    rows: tuple[FitRow, ...]
    source_tables: tuple[SourceTableReceipt, ...]
    release_root: str
    archive_manifest_path: str
    archive_manifest_file_sha256: str
    release_manifest_integrity_sha256: str
    content_sha256: str
    release_id: str
    release_schema: str
    release_contract_path: str | None = None
    release_contract_file_sha256: str | None = None
    release_contract_sha256: str | None = None

    @property
    def trajectory_count(self) -> int:
        return len({row.trajectory_id for row in self.rows})

    @property
    def readout_count(self) -> int:
        return len({row.readout_id for row in self.rows})

    def to_frame(self):
        """Materialize a new DataFrame; callers cannot mutate the stored rows."""
        import pandas as pd
        frame = pd.DataFrame.from_records([row.to_dict() for row in self.rows], columns=FIT_ROW_COLUMNS)
        string_columns = ('trajectory_id', 'readout_id', 'language', 'ladder')
        integer_columns = ('E', 'K', 'validation_token_count', 'model_seed')
        float_columns = ('N_act', 'N_act_nonembedding', 'D_total', 'D_t', 'adjusted_loss', 'raw_nll', 'unigram_nll')
        for name in string_columns:
            frame[name] = frame[name].astype('string')
        for name in integer_columns:
            frame[name] = frame[name].astype('int64')
        for name in float_columns:
            frame[name] = frame[name].astype('float64')
        return frame
_REQUIRED_COLUMNS: dict[str, tuple[str, ...]] = {'trajectories': ('trajectory_id', 'ladder', 'experts', 'language_count', 'model_config_id', 'model_seed', 'status', 'provenance_class'), 'readouts': ('readout_id', 'trajectory_id', 'ladder', 'experts', 'language_count', 'actual_tokens', 'trained_language_count', 'primary_metric_name', 'provenance_class'), 'model_configs': ('model_config_id', 'ladder', 'experts', 'declared_active_parameters', 'declared_active_nonembedding_parameters', 'provenance_class'), 'validation_language': ('readout_id', 'trajectory_id', 'ladder', 'experts', 'language_count', 'actual_tokens', 'language', 'metric_role', 'token_count', 'nll_sum', 'raw_nll', 'adjusted_nll_sum', 'atlas_tao_unigram_adjusted_loss', 'unigram_log_probability_sum', 'vocab_size', 'special_token_ids_json', 'validation_input_identity_sha256', 'target_token_ids_sha256', 'unigram_counts_sha256', 'provenance_class'), 'data_exposure': ('readout_id', 'trajectory_id', 'ladder', 'experts', 'language_count', 'actual_tokens', 'language', 'metric_role', 'realized_self_exposure_tokens', 'provenance_class')}

def _require_columns(frame: Any, table: str) -> None:
    missing = sorted(set(_REQUIRED_COLUMNS[table]) - set(frame.columns))
    if missing:
        raise Stage1FitDataError(f'{table} is missing required columns: {missing}')

def _require_unique(frame: Any, columns: Sequence[str], context: str) -> None:
    if frame.duplicated(list(columns)).any():
        example = frame.loc[frame.duplicated(list(columns), keep=False), list(columns)].head(3)
        raise Stage1FitDataError(f"duplicate {context} keys {list(columns)}: {example.to_dict('records')}")

def _require_equal(left: Any, right: Any, context: str) -> None:
    import numpy as np
    left_values = np.asarray(left)
    right_values = np.asarray(right)
    if left_values.shape != right_values.shape or not np.array_equal(left_values, right_values):
        mismatch = np.flatnonzero(left_values != right_values)
        first = int(mismatch[0]) if mismatch.size else 0
        raise Stage1FitDataError(f'{context} disagrees at row {first}: {left_values[first]!r} != {right_values[first]!r}')

def _require_allclose(left: Any, right: Any, context: str) -> None:
    import numpy as np
    left_values = np.asarray(left, dtype=np.float64)
    right_values = np.asarray(right, dtype=np.float64)
    if not np.allclose(left_values, right_values, rtol=1e-12, atol=1e-12):
        difference = np.abs(left_values - right_values)
        index = int(np.nanargmax(difference))
        raise Stage1FitDataError(f'{context} disagrees at row {index}: {left_values[index]!r} != {right_values[index]!r}')

def _validate_release_manifest(
    release_root: Path,
    archive_manifest_path: Path,
    contract: ReleaseContract,
) -> tuple[dict[str, Any], tuple[SourceTableReceipt, ...], str]:
    import pyarrow.parquet as pq
    release_root = release_root.resolve()
    archive_manifest_path = archive_manifest_path.resolve()
    if not release_root.is_dir():
        raise Stage1FitDataError(f'sealed release root is missing: {release_root}')
    if not archive_manifest_path.is_file():
        raise Stage1FitDataError(f'archived release manifest is missing: {archive_manifest_path}')
    manifest_file_sha256 = sha256_file(archive_manifest_path)
    if manifest_file_sha256 != contract.archive_manifest_file_sha256:
        raise Stage1FitDataError(f'archived release-manifest file hash changed: {manifest_file_sha256}')
    try:
        manifest = json.loads(archive_manifest_path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as error:
        raise Stage1FitDataError(f'cannot read release manifest: {error}') from error
    if not isinstance(manifest, dict):
        raise Stage1FitDataError('release manifest must be a JSON object')
    declared_integrity = manifest.get('manifest_integrity_sha256')
    if declared_integrity != contract.release_manifest_integrity_sha256:
        raise Stage1FitDataError(f'release manifest integrity identity changed: {declared_integrity!r}')
    integrity_payload = dict(manifest)
    integrity_payload.pop('manifest_integrity_sha256', None)
    actual_integrity = canonical_json_sha256(integrity_payload)
    if actual_integrity != declared_integrity:
        raise Stage1FitDataError(f'release manifest payload integrity mismatch: {actual_integrity}')
    if manifest.get('release_name') != contract.release_id:
        raise Stage1FitDataError(f"expected release {contract.release_id}, got {manifest.get('release_name')!r}")
    if manifest.get('schema') != contract.release_schema:
        raise Stage1FitDataError(f"expected release schema {contract.release_schema}, got {manifest.get('schema')!r}")
    if manifest.get('statistical_unit') != 'trajectory_id':
        raise Stage1FitDataError('sealed release statistical unit changed')
    inventory = manifest.get('table_inventory')
    if not isinstance(inventory, dict):
        raise Stage1FitDataError('release manifest table_inventory is not an object')
    receipts: list[SourceTableReceipt] = []
    for name in contract.required_source_tables:
        entry = inventory.get(name)
        if not isinstance(entry, dict):
            raise Stage1FitDataError(f'release manifest lacks table {name}')
        relative = Path(str(entry.get('path', '')))
        expected_relative = Path('tables') / f'{name}.parquet'
        if relative != expected_relative:
            raise Stage1FitDataError(f'manifest path for {name} changed: {relative}')
        path = (release_root / relative).resolve()
        if not path.is_relative_to(release_root) or not path.is_file():
            raise Stage1FitDataError(f'sealed source table is missing: {path}')
        actual_sha256 = sha256_file(path)
        expected_sha256 = str(entry.get('sha256', ''))
        if actual_sha256 != expected_sha256:
            raise Stage1FitDataError(f'sealed source hash mismatch for {name}: {actual_sha256}')
        rows = int(pq.ParquetFile(path).metadata.num_rows)
        if rows != int(entry.get('rows', -1)):
            raise Stage1FitDataError(f'sealed row count mismatch for {name}: {rows}')
        schema_names = set(pq.read_schema(path).names)
        missing = sorted(set(_REQUIRED_COLUMNS[name]) - schema_names)
        if missing:
            raise Stage1FitDataError(f'sealed {name} Parquet schema is missing columns: {missing}')
        receipts.append(SourceTableReceipt(name=name, path=str(path), rows=rows, sha256=actual_sha256, manifest_sha256=expected_sha256))
    return (manifest, tuple(receipts), manifest_file_sha256)

def build_release_contract(
    release_root: Path | str,
    archive_manifest_path: Path | str,
) -> ReleaseContract:
    """Freeze a self-sealed canonical release into an explicit fitter contract."""

    release_root = Path(release_root).resolve()
    archive_manifest_path = Path(archive_manifest_path).resolve()
    if not release_root.is_dir() or not archive_manifest_path.is_file():
        raise Stage1FitDataError('release root or manifest is missing')
    manifest_file_sha256 = sha256_file(archive_manifest_path)
    try:
        manifest = json.loads(archive_manifest_path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as error:
        raise Stage1FitDataError(f'cannot read release manifest: {error}') from error
    if not isinstance(manifest, dict):
        raise Stage1FitDataError('release manifest must be a JSON object')
    integrity = _require_sha256(
        manifest.get('manifest_integrity_sha256'),
        field='manifest_integrity_sha256',
    )
    unsigned_manifest = dict(manifest)
    unsigned_manifest.pop('manifest_integrity_sha256', None)
    if canonical_json_sha256(unsigned_manifest) != integrity:
        raise Stage1FitDataError('release manifest payload integrity mismatch')
    release_id = str(manifest.get('release_name', ''))
    release_schema = str(manifest.get('schema', ''))
    if not release_id or release_schema != RELEASE_SCHEMA:
        raise Stage1FitDataError('release name is empty or release schema changed')
    if manifest.get('statistical_unit') != 'trajectory_id':
        raise Stage1FitDataError('release statistical unit changed')
    inventory = manifest.get('table_inventory')
    if not isinstance(inventory, Mapping):
        raise Stage1FitDataError('release manifest table inventory is malformed')
    for name in REQUIRED_SOURCE_TABLES:
        entry = inventory.get(name)
        if not isinstance(entry, Mapping):
            raise Stage1FitDataError(f'release manifest lacks table {name}')
    if int(inventory['trajectories'].get('rows', -1)) != EXPECTED_TRAJECTORIES:
        raise Stage1FitDataError('release trajectory count changed')
    if int(inventory['readouts'].get('rows', -1)) != EXPECTED_READOUTS:
        raise Stage1FitDataError('release readout count changed')
    unsigned_contract = {
        'schema_version': RELEASE_CONTRACT_SCHEMA_VERSION,
        'release_id': release_id,
        'release_schema': release_schema,
        'archive_manifest_file_sha256': manifest_file_sha256,
        'release_manifest_integrity_sha256': integrity,
        'expected_trajectories': EXPECTED_TRAJECTORIES,
        'expected_readouts': EXPECTED_READOUTS,
        'expected_fit_rows': EXPECTED_FIT_ROWS,
        'required_source_tables': list(REQUIRED_SOURCE_TABLES),
    }
    contract = ReleaseContract(
        release_id=release_id,
        release_schema=release_schema,
        archive_manifest_file_sha256=manifest_file_sha256,
        release_manifest_integrity_sha256=integrity,
        expected_trajectories=EXPECTED_TRAJECTORIES,
        expected_readouts=EXPECTED_READOUTS,
        expected_fit_rows=EXPECTED_FIT_ROWS,
        required_source_tables=REQUIRED_SOURCE_TABLES,
        contract_sha256=canonical_json_sha256(unsigned_contract),
    )
    _validate_release_manifest(release_root, archive_manifest_path, contract)
    return contract

def assemble_fit_frame(*, trajectories: Any, readouts: Any, model_configs: Any, validation_language: Any, data_exposure: Any, strict_counts: bool=True):
    """Join already-loaded canonical tables with fail-closed validations.

    ``strict_counts=False`` exists only for small synthetic unit fixtures.  It
    never weakens join, provenance, response, or positivity validation.
    """
    import numpy as np
    import pandas as pd
    frames = {'trajectories': trajectories.copy(), 'readouts': readouts.copy(), 'model_configs': model_configs.copy(), 'validation_language': validation_language.copy(), 'data_exposure': data_exposure.copy()}
    for name, frame in frames.items():
        _require_columns(frame, name)
    trajectories = frames['trajectories']
    readouts = frames['readouts']
    model_configs = frames['model_configs']
    validation_language = frames['validation_language']
    data_exposure = frames['data_exposure']
    _require_unique(trajectories, ('trajectory_id',), 'trajectory')
    _require_unique(model_configs, ('model_config_id',), 'model configuration')
    _require_unique(model_configs, ('ladder', 'experts'), 'model design')
    _require_unique(readouts, ('readout_id',), 'readout')
    _require_unique(trajectories, ('ladder', 'experts', 'language_count'), 'design cell')
    if not trajectories['status'].eq('observed_complete').all():
        raise Stage1FitDataError('non-complete trajectory entered the fit')
    if not trajectories['provenance_class'].eq('configured').all():
        raise Stage1FitDataError('trajectory provenance must be configured')
    if not model_configs['provenance_class'].eq('configured').all():
        raise Stage1FitDataError('model provenance must be configured')
    if not readouts['provenance_class'].eq('observed').all():
        raise Stage1FitDataError('readout provenance must be observed')
    if not readouts['primary_metric_name'].eq(PRIMARY_METRIC_NAME).all():
        raise Stage1FitDataError('readout primary metric definition changed')
    model_reference = model_configs[['model_config_id', 'ladder', 'experts', 'declared_active_parameters', 'declared_active_nonembedding_parameters']].rename(columns={'ladder': 'model_ladder', 'experts': 'model_experts'})
    trajectory_reference = trajectories.merge(model_reference, on='model_config_id', how='left', validate='many_to_one', indicator=True)
    if not trajectory_reference['_merge'].eq('both').all():
        raise Stage1FitDataError('trajectory is missing its model configuration')
    trajectory_reference = trajectory_reference.drop(columns='_merge')
    _require_equal(trajectory_reference['ladder'], trajectory_reference['model_ladder'], 'trajectory/model ladder')
    _require_equal(trajectory_reference['experts'], trajectory_reference['model_experts'], 'trajectory/model expert count')
    trajectory_lookup = trajectory_reference[['trajectory_id', 'ladder', 'experts', 'language_count', 'model_seed', 'declared_active_parameters', 'declared_active_nonembedding_parameters']].rename(columns={'ladder': 'trajectory_ladder', 'experts': 'trajectory_experts', 'language_count': 'trajectory_language_count'})
    readout_reference = readouts.merge(trajectory_lookup, on='trajectory_id', how='left', validate='many_to_one', indicator=True)
    if not readout_reference['_merge'].eq('both').all():
        raise Stage1FitDataError('readout is missing its trajectory')
    readout_reference = readout_reference.drop(columns='_merge')
    _require_equal(readout_reference['ladder'], readout_reference['trajectory_ladder'], 'readout/trajectory ladder')
    _require_equal(readout_reference['experts'], readout_reference['trajectory_experts'], 'readout/trajectory expert count')
    _require_equal(readout_reference['language_count'], readout_reference['trajectory_language_count'], 'readout/trajectory language count')
    if set(readout_reference['trajectory_id']) != set(trajectory_reference['trajectory_id']):
        raise Stage1FitDataError('readouts do not cover every trajectory')
    validation = validation_language.loc[validation_language['metric_role'].eq(PRIMARY_METRIC_ROLE)].copy()
    exposure = data_exposure.loc[data_exposure['metric_role'].eq(PRIMARY_METRIC_ROLE)].copy()
    if validation.empty or exposure.empty:
        raise Stage1FitDataError('trained validation/exposure rows are empty')
    _require_unique(validation, ('trajectory_id', 'readout_id', 'language'), 'trained validation')
    _require_unique(exposure, ('trajectory_id', 'readout_id', 'language'), 'trained exposure')
    if not validation['provenance_class'].eq('observed').all():
        raise Stage1FitDataError('validation provenance must be observed')
    if not exposure['provenance_class'].eq('observed').all():
        raise Stage1FitDataError('exposure provenance must be observed')
    counts = validation.groupby('readout_id', observed=True).size()
    expected_counts = readouts.set_index('readout_id')['trained_language_count']
    if set(counts.index) != set(expected_counts.index):
        raise Stage1FitDataError('trained validation does not cover every readout')
    if not counts.reindex(expected_counts.index).eq(expected_counts).all():
        raise Stage1FitDataError('trained-language counts disagree with readout metadata')
    joined = validation.merge(exposure, on=('readout_id', 'language'), how='outer', validate='one_to_one', suffixes=('_validation', '_exposure'), indicator=True)
    if not joined['_merge'].eq('both').all():
        missing = joined['_merge'].value_counts().to_dict()
        raise Stage1FitDataError(f'trained validation/exposure coverage mismatch: {missing}')
    joined = joined.drop(columns='_merge')
    for field in ('trajectory_id', 'ladder', 'experts', 'language_count', 'actual_tokens'):
        _require_equal(joined[f'{field}_validation'], joined[f'{field}_exposure'], f'validation/exposure {field}')
    readout_lookup = readout_reference[['readout_id', 'trajectory_id', 'ladder', 'experts', 'language_count', 'actual_tokens', 'model_seed', 'declared_active_parameters', 'declared_active_nonembedding_parameters']].rename(columns={'trajectory_id': 'trajectory_id_readout', 'ladder': 'ladder_readout', 'experts': 'experts_readout', 'language_count': 'language_count_readout', 'actual_tokens': 'actual_tokens_readout'})
    joined = joined.merge(readout_lookup, on='readout_id', how='left', validate='many_to_one', indicator=True)
    if not joined['_merge'].eq('both').all():
        raise Stage1FitDataError('trained row is missing its readout')
    joined = joined.drop(columns='_merge')
    for field in ('trajectory_id', 'ladder', 'experts', 'language_count', 'actual_tokens'):
        _require_equal(joined[f'{field}_validation'], joined[f'{field}_readout'], f'validation/readout {field}')
    stable_definition_fields = ('validation_input_identity_sha256', 'target_token_ids_sha256', 'unigram_counts_sha256', 'token_count', 'unigram_log_probability_sum', 'vocab_size', 'special_token_ids_json')
    definition_counts = validation.groupby('language', observed=True)[list(stable_definition_fields)].nunique(dropna=False)
    changed = [field for field in stable_definition_fields if (definition_counts[field] != 1).any()]
    if changed:
        raise Stage1FitDataError(f'validation/unigram definition changes within language: {changed}')
    token_count = joined['token_count'].to_numpy(dtype=np.float64)
    if np.any(token_count <= 0.0):
        raise Stage1FitDataError('validation token counts must be positive')
    raw_from_sum = joined['nll_sum'].to_numpy(dtype=np.float64) / token_count
    adjusted_from_sum = joined['adjusted_nll_sum'].to_numpy(dtype=np.float64) / token_count
    unigram_nll = -joined['unigram_log_probability_sum'].to_numpy(dtype=np.float64) / token_count
    raw_nll = joined['raw_nll'].to_numpy(dtype=np.float64)
    adjusted_loss = joined['atlas_tao_unigram_adjusted_loss'].to_numpy(dtype=np.float64)
    _require_allclose(raw_nll, raw_from_sum, 'raw NLL sufficient statistics')
    _require_allclose(adjusted_loss, adjusted_from_sum, 'adjusted-loss sufficient statistics')
    _require_allclose(adjusted_loss, raw_nll - unigram_nll, 'D1 adjusted-loss definition')
    output = pd.DataFrame({'trajectory_id': joined['trajectory_id_readout'].astype('string'), 'readout_id': joined['readout_id'].astype('string'), 'language': joined['language'].astype('string'), 'N_act': joined['declared_active_parameters'].astype('float64'), 'N_act_nonembedding': joined['declared_active_nonembedding_parameters'].astype('float64'), 'E': joined['experts_readout'].astype('int64'), 'K': joined['language_count_readout'].astype('int64'), 'D_total': joined['actual_tokens_readout'].astype('float64'), 'D_t': joined['realized_self_exposure_tokens'].astype('float64'), 'adjusted_loss': adjusted_loss.astype(np.float64), 'raw_nll': raw_nll.astype(np.float64), 'validation_token_count': joined['token_count'].astype('int64'), 'unigram_nll': unigram_nll.astype(np.float64), 'ladder': joined['ladder_readout'].astype('string'), 'model_seed': joined['model_seed'].astype('int64')}, columns=FIT_ROW_COLUMNS)
    numeric = output[['N_act', 'N_act_nonembedding', 'E', 'K', 'D_total', 'D_t', 'adjusted_loss', 'raw_nll', 'unigram_nll']].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise Stage1FitDataError('fit rows contain missing or non-finite values')
    if (output[['N_act', 'N_act_nonembedding', 'D_total', 'D_t']] <= 0).any().any():
        raise Stage1FitDataError('N_act, non-embedding N_act, D_total, and D_t must be positive')
    _require_unique(output, ('trajectory_id', 'readout_id', 'language'), 'canonical fit row')
    if strict_counts:
        counts_observed = (len(trajectories), len(readouts), len(output))
        counts_expected = (EXPECTED_TRAJECTORIES, EXPECTED_READOUTS, EXPECTED_FIT_ROWS)
        if counts_observed != counts_expected:
            raise Stage1FitDataError(f'sealed v1 count mismatch; expected trajectories/readouts/rows {counts_expected}, got {counts_observed}')
        if tuple(sorted((int(value) for value in output['E'].unique()))) != EXPECTED_EXPERT_LEVELS:
            raise Stage1FitDataError('sealed expert levels changed')
        if tuple(sorted((int(value) for value in output['K'].unique()))) != EXPECTED_LANGUAGE_COUNT_LEVELS:
            raise Stage1FitDataError('sealed language-count levels changed')
        if tuple(sorted((str(value) for value in output['ladder'].unique()))) != EXPECTED_LADDERS:
            raise Stage1FitDataError('sealed ladder levels changed')
    output = output.sort_values(['trajectory_id', 'readout_id', 'language'], kind='mergesort').reset_index(drop=True)
    return output

def _fit_rows_from_frame(frame: Any) -> tuple[FitRow, ...]:
    return tuple((FitRow(trajectory_id=str(item.trajectory_id), readout_id=str(item.readout_id), language=str(item.language), N_act=float(item.N_act), N_act_nonembedding=float(item.N_act_nonembedding), E=int(item.E), K=int(item.K), D_total=float(item.D_total), D_t=float(item.D_t), adjusted_loss=float(item.adjusted_loss), raw_nll=float(item.raw_nll), validation_token_count=int(item.validation_token_count), unigram_nll=float(item.unigram_nll), ladder=str(item.ladder), model_seed=int(item.model_seed)) for item in frame.itertuples(index=False)))

def load_canonical_fit_data(
    release_root: Path | str=DEFAULT_RELEASE_ROOT,
    *,
    archive_manifest_path: Path | str=DEFAULT_ARCHIVED_MANIFEST_PATH,
    release_contract_path: Path | str | None=None,
    strict_counts: bool=True,
) -> PreparedFitData:
    """Read and validate the sealed release without mutating it."""
    import pandas as pd
    release_root = Path(release_root).resolve()
    archive_manifest_path = Path(archive_manifest_path).resolve()
    contract_path = (
        None if release_contract_path is None else Path(release_contract_path).resolve()
    )
    contract = (
        historical_release_contract()
        if contract_path is None
        else load_release_contract(contract_path)
    )
    contract_file_sha256 = None if contract_path is None else sha256_file(contract_path)
    manifest, table_receipts, manifest_file_sha256 = _validate_release_manifest(
        release_root, archive_manifest_path, contract
    )
    loaded: dict[str, Any] = {}
    for receipt in table_receipts:
        loaded[receipt.name] = pd.read_parquet(receipt.path, columns=list(_REQUIRED_COLUMNS[receipt.name]))
    frame = assemble_fit_frame(trajectories=loaded['trajectories'], readouts=loaded['readouts'], model_configs=loaded['model_configs'], validation_language=loaded['validation_language'], data_exposure=loaded['data_exposure'], strict_counts=strict_counts)
    rows = _fit_rows_from_frame(frame)
    content_sha256 = fit_rows_content_sha256(list(rows))
    for receipt in table_receipts:
        if sha256_file(Path(receipt.path)) != receipt.sha256:
            raise Stage1FitDataError(f'sealed source changed while being read: {receipt.name}')
    if sha256_file(archive_manifest_path) != manifest_file_sha256:
        raise Stage1FitDataError('archived release manifest changed while being read')
    if contract_path is not None and sha256_file(contract_path) != contract_file_sha256:
        raise Stage1FitDataError('release contract changed while being read')
    return PreparedFitData(
        rows=rows,
        source_tables=table_receipts,
        release_root=str(release_root),
        archive_manifest_path=str(archive_manifest_path),
        archive_manifest_file_sha256=manifest_file_sha256,
        release_manifest_integrity_sha256=str(manifest['manifest_integrity_sha256']),
        content_sha256=content_sha256,
        release_id=contract.release_id,
        release_schema=contract.release_schema,
        release_contract_path=None if contract_path is None else str(contract_path),
        release_contract_file_sha256=contract_file_sha256,
        release_contract_sha256=None if contract_path is None else contract.contract_sha256,
    )

def _atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f'.{path.name}.', suffix='.tmp')
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, 'wb') as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

def write_release_contract(
    contract: ReleaseContract,
    output_path: Path | str,
) -> Path:
    """Write a new immutable release contract without overwriting evidence."""

    path = Path(output_path).resolve()
    if path.exists():
        raise Stage1FitDataError(f'refusing to overwrite release contract: {path}')
    _atomic_write_bytes(path, canonical_json_bytes(contract.to_dict()) + b'\n')
    if load_release_contract(path) != contract:
        raise Stage1FitDataError('written release contract failed round-trip validation')
    return path

def write_fit_data_snapshot(prepared: PreparedFitData, output_root: Path | str, *, overwrite: bool=False) -> InputReceipt:
    """Atomically write ``input/fit_rows.parquet`` and its receipt."""
    output_root = Path(output_root).resolve()
    finalization = output_root / 'receipts' / 'finalization.json'
    if finalization.exists():
        raise Stage1FitDataError(f'refusing finalized output root: {output_root}')
    input_root = output_root / 'input'
    fit_rows_path = input_root / 'fit_rows.parquet'
    receipt_path = input_root / 'input_receipt.json'
    existing = [path for path in (fit_rows_path, receipt_path) if path.exists()]
    if existing and (not overwrite):
        raise Stage1FitDataError('refusing to overwrite prepared input artifacts: ' + ', '.join((str(path) for path in existing)))
    input_root.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=input_root, prefix='.fit_rows.', suffix='.parquet.tmp')
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        prepared.to_frame().to_parquet(temporary, index=False, engine='pyarrow', compression='zstd')
        os.replace(temporary, fit_rows_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    fit_rows_file_sha256 = sha256_file(fit_rows_path)
    receipt = InputReceipt(
        release_root=prepared.release_root,
        archive_manifest_path=prepared.archive_manifest_path,
        archive_manifest_file_sha256=prepared.archive_manifest_file_sha256,
        release_manifest_integrity_sha256=prepared.release_manifest_integrity_sha256,
        trajectories=prepared.trajectory_count,
        readouts=prepared.readout_count,
        fit_rows=len(prepared.rows),
        fit_rows_path=str(fit_rows_path),
        fit_rows_file_sha256=fit_rows_file_sha256,
        fit_rows_content_sha256=prepared.content_sha256,
        source_tables=prepared.source_tables,
        release_id=prepared.release_id,
        release_schema=prepared.release_schema,
        release_contract_path=prepared.release_contract_path,
        release_contract_file_sha256=prepared.release_contract_file_sha256,
        release_contract_sha256=prepared.release_contract_sha256,
    )
    _atomic_write_bytes(receipt_path, canonical_json_bytes(receipt.to_dict()) + b'\n')
    return receipt

def load_fit_rows_snapshot(receipt: Mapping[str, Any]) -> Any:
    """Revalidate and load the exact Parquet snapshot bound by an input receipt."""
    import pandas as pd
    required = {'schema_version', 'fit_rows_path', 'fit_rows_file_sha256',
        'fit_rows_content_sha256', 'trajectories', 'readouts', 'fit_rows'}
    if not required <= set(receipt) or receipt.get('schema_version') != INPUT_RECEIPT_SCHEMA_VERSION:
        raise Stage1FitDataError('input receipt is incomplete or has the wrong schema')
    path = Path(str(receipt['fit_rows_path']))
    if sha256_file(path) != receipt['fit_rows_file_sha256']:
        raise Stage1FitDataError('fit-row snapshot file hash changed')
    frame = pd.read_parquet(path, columns=list(FIT_ROW_COLUMNS))
    rows = _fit_rows_from_frame(frame)
    if fit_rows_content_sha256(list(rows)) != receipt['fit_rows_content_sha256']:
        raise Stage1FitDataError('fit-row snapshot canonical content hash changed')
    observed = (len({row.trajectory_id for row in rows}), len({row.readout_id for row in rows}), len(rows))
    expected = (int(receipt['trajectories']), int(receipt['readouts']), int(receipt['fit_rows']))
    if observed != expected:
        raise Stage1FitDataError(f'fit-row snapshot counts changed: expected {expected}, got {observed}')
    if sha256_file(path) != receipt['fit_rows_file_sha256']:
        raise Stage1FitDataError('fit-row snapshot changed while being read')
    return frame

def prepare_fit_data(
    release_root: Path | str=DEFAULT_RELEASE_ROOT,
    *,
    archive_manifest_path: Path | str=DEFAULT_ARCHIVED_MANIFEST_PATH,
    release_contract_path: Path | str | None=None,
    output_root: Path | str | None=None,
    dry_run: bool=False,
    overwrite: bool=False,
) -> tuple[PreparedFitData, InputReceipt | None]:
    """Convenience entrypoint shared by local and Slurm planning CLIs."""
    prepared = load_canonical_fit_data(
        release_root,
        archive_manifest_path=archive_manifest_path,
        release_contract_path=release_contract_path,
        strict_counts=True,
    )
    if output_root is None or dry_run:
        return (prepared, None)
    return (prepared, write_fit_data_snapshot(prepared, output_root, overwrite=overwrite))
