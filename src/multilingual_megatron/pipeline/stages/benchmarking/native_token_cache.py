"""Sealed, memory-mapped CPU token caches for native likelihood evaluation."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any

from multilingual_megatron.contracts.hashing import canonical_json_sha256, sha256_file


def tokenizer_cache_identity(tokenizer):
    """Bind caches to the complete native tokenizer, including HF special tokens."""
    if tokenizer.get('sha256'):
        return tokenizer['sha256']
    if tokenizer.get('files'):
        return canonical_json_sha256(tokenizer['files'])
    raise ValueError('token caches require a verified tokenizer identity')


def document_windows(token_ids, *, context_token_id: int, max_length: int, stride: int):
    """Yield causal token windows covering every document target exactly once.

    The first window has a synthetic prefix. Later windows retain as much
    preceding context as fits; only new targets contribute to likelihood.
    Token IDs are never decoded/re-encoded at window boundaries.
    """
    if not token_ids or not 1 <= stride <= max_length:
        raise ValueError('document windows require nonempty tokens and a valid stride')
    target_start = 0
    while target_start < len(token_ids):
        target_end = min(len(token_ids), max_length if target_start == 0 else target_start + stride)
        context_start = max(0, target_end - max_length - 1)
        context = [context_token_id] if target_start == 0 else list(token_ids[context_start:target_start])
        targets = list(token_ids[target_start:target_end])
        yield context, targets, dict(context_start=context_start, target_start=target_start,
                                    target_end=target_end, document_tokens=len(token_ids))
        target_start = target_end


def token_window_sha256(tokens):
    import numpy as np
    return hashlib.sha256(np.asarray(tokens, dtype='<i4').tobytes()).hexdigest()


class TokenCacheWriter:
    """Store each causal window once; labels share the shifted token array."""
    def __init__(self, directory: Path):
        import numpy as np
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        self.handle = (directory / 'tokens.i32').open('wb')
        self.offsets = [0]
        self.metadata = []
        self.np = np

    def append(self, sequence):
        values = self.np.asarray((*sequence.input_ids, sequence.labels[-1]), dtype='<i4')
        self.handle.write(values.tobytes())
        self.offsets.append(self.offsets[-1] + len(values))
        self.metadata.append((sequence.continuation_start, sequence.continuation_tokens, sequence.truncated_tokens))

    def finish(self, *, request_sha256: str, tokenizer: dict, sequence_length: int):
        self.handle.close()
        self.np.save(self.directory / 'offsets.npy', self.np.asarray(self.offsets, dtype='<i8'), allow_pickle=False)
        self.np.save(self.directory / 'metadata.npy', self.np.asarray(self.metadata, dtype='<i4'), allow_pickle=False)
        return _write_manifest(self.directory, len(self.metadata), request_sha256, tokenizer, sequence_length)


def _write_manifest(directory, count, request_sha256, tokenizer, sequence_length):
    payload = {'schema': 'native_likelihood_token_cache/v1', 'requests': count,
               'request_sha256': request_sha256, 'tokenizer_sha256': tokenizer_cache_identity(tokenizer),
               'context_token_id': tokenizer['context_token_id'], 'sequence_length': sequence_length,
               'files': {name: {'path': str((directory / name).resolve()),
                                'sha256': sha256_file(directory / name)}
                         for name in ('tokens.i32', 'offsets.npy', 'metadata.npy')}}
    payload['manifest_sha256'] = canonical_json_sha256(payload)
    path = directory / 'manifest.json'
    path.write_text(json.dumps(payload, indent=2) + '\n')
    return {'path': str(path.resolve()), 'sha256': sha256_file(path)}


def combine_token_caches(references, directory, *, request_sha256, tokenizer, sequence_length):
    import shutil
    import numpy as np
    payloads = []
    for ref in references:
        path = Path(ref['path'])
        if sha256_file(path) != ref['sha256']:
            raise ValueError('source token cache manifest drift')
        payload = json.loads(path.read_text())
        expected_hash = payload.pop('manifest_sha256')
        if canonical_json_sha256(payload) != expected_hash or payload.get('schema') != 'native_likelihood_token_cache/v1':
            raise ValueError('invalid source token cache manifest')
        if (payload['tokenizer_sha256'] != tokenizer_cache_identity(tokenizer) or payload['sequence_length'] != sequence_length
                or payload['context_token_id'] != tokenizer['context_token_id']):
            raise ValueError('incompatible source token caches')
        for file in payload['files'].values():
            if sha256_file(Path(file['path'])) != file['sha256']:
                raise ValueError('source token array drift')
        payloads.append(payload)
    directory.mkdir(parents=True, exist_ok=True)
    count = sum(p['requests'] for p in payloads)
    offsets = np.lib.format.open_memmap(directory / 'offsets.npy', mode='w+', dtype='<i8', shape=(count + 1,))
    metadata = np.lib.format.open_memmap(directory / 'metadata.npy', mode='w+', dtype='<i4', shape=(count, 3))
    cursor, token_offset = 0, 0
    offsets[0] = 0
    with (directory / 'tokens.i32').open('wb') as target:
        for payload in payloads:
            n = payload['requests']
            files = payload['files']
            local_offsets = np.load(files['offsets.npy']['path'], allow_pickle=False)
            offsets[cursor + 1:cursor + n + 1] = local_offsets[1:] + token_offset
            metadata[cursor:cursor + n] = np.load(files['metadata.npy']['path'], allow_pickle=False)
            with open(files['tokens.i32']['path'], 'rb') as source:
                shutil.copyfileobj(source, target, length=1024 * 1024)
            cursor += n
            token_offset += int(local_offsets[-1])
    offsets.flush()
    metadata.flush()
    return _write_manifest(directory, count, request_sha256, tokenizer, sequence_length)


def load_cached_sequences(reference: dict, spec: Any, request_rows: list):
    import numpy as np
    from .native_megatron_scoring import ScoredSequence
    path = Path(reference['path'])
    if sha256_file(path) != reference['sha256']:
        raise ValueError('token cache manifest drift')
    payload = json.loads(path.read_text())
    expected_hash = payload.pop('manifest_sha256')
    if canonical_json_sha256(payload) != expected_hash or payload.get('schema') != 'native_likelihood_token_cache/v1':
        raise ValueError('invalid token cache manifest')
    for field, expected in {
        'request_sha256': spec.request_manifest_sha256, 'requests': len(request_rows),
        'tokenizer_sha256': tokenizer_cache_identity({'sha256': spec.tokenizer_model_sha256,
            'files': getattr(spec, 'tokenizer_files_sha256', {})}), 'sequence_length': spec.sequence_length,
        'context_token_id': spec.context_token_id,
    }.items():
        if payload.get(field) != expected:
            raise ValueError(f'token cache contract mismatch: {field}')
    files = payload['files']
    for ref in files.values():
        if sha256_file(Path(ref['path'])) != ref['sha256']:
            raise ValueError('token cache array drift')
    tokens = np.memmap(files['tokens.i32']['path'], mode='r', dtype='<i4')
    offsets = np.load(files['offsets.npy']['path'], mmap_mode='r', allow_pickle=False)
    metadata = np.load(files['metadata.npy']['path'], mmap_mode='r', allow_pickle=False)
    if offsets.dtype != np.dtype('<i8') or metadata.dtype != np.dtype('<i4'):
        raise ValueError('invalid token cache dtypes')
    if offsets.shape != (len(request_rows) + 1,) or metadata.shape != (len(request_rows), 3):
        raise ValueError('invalid token cache shapes')
    widths = np.diff(offsets) - 1
    if offsets[0] != 0 or offsets[-1] != len(tokens) or np.any(widths < 1) or np.any(widths > spec.sequence_length):
        raise ValueError('invalid cached sequence boundaries')
    if np.any(metadata < 0) or np.any(metadata[:, 1] < 1) or np.any(metadata[:, 0] + metadata[:, 1] != widths):
        raise ValueError('invalid cached continuation boundaries')
    if np.any(tokens < 0) or np.any(tokens >= spec.tokenizer_vocab_size):
        raise ValueError('cached token outside vocabulary')
    for i, row in enumerate(request_rows):
        if row.get('request_type') != 'token_window_loglikelihood':
            continue
        window = row['token_window']
        start, end, context_start = (window[k] for k in ('target_start', 'target_end', 'context_start'))
        if not 0 <= context_start <= start < end <= window['document_tokens']:
            raise ValueError('invalid document target interval')
        expected_context = 1 if start == 0 else start - context_start
        if (expected_context < 1 or int(metadata[i, 1]) != end - start
                or int(metadata[i, 0]) != expected_context - 1 or metadata[i, 2] != 0):
            raise ValueError('cached document target interval changed')
        values = tokens[offsets[i]:offsets[i+1]]
        if token_window_sha256(values) != window['window_token_ids_sha256']:
            raise ValueError('cached document window token identity changed')
    return [ScoredSequence(request_index=i, request_id=row['request_id'],
                           input_ids=tokens[offsets[i]:offsets[i + 1] - 1],
                           labels=tokens[offsets[i] + 1:offsets[i + 1]],
                           continuation_start=int(metadata[i, 0]), continuation_tokens=int(metadata[i, 1]),
                           truncated_tokens=int(metadata[i, 2])) for i, row in enumerate(request_rows)]
