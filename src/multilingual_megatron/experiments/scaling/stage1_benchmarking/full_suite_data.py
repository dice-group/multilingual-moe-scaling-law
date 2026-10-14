"""Complete labeled evaluation splits for the expanded Stage 1 benchmark suite.

Source inventories are pinned by revision and file hash. Rendering retains the
study's zero-shot, full-option continuation protocol; no row sampling occurs.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

from multilingual_megatron.contracts.hashing import canonical_json_sha256, sha256_file
from .tasks import _global_mmlu_target, _localized, _preprocess_hellaswag_text


def read_documents(item):
    path = Path(item['source'])
    if sha256_file(path) != item['source_sha256']:
        raise ValueError(f'source drift: {path}')
    if path.suffix == '.json':
        rows = json.loads(path.read_text())
    elif path.suffix == '.xml':
        rows = [dict(premise=x.findtext('p'), choice1=x.findtext('a1'),
                     choice2=x.findtext('a2'), question=x.attrib['asks-for'],
                     label=int(x.attrib['most-plausible-alternative']) - 1)
                for x in ET.parse(path).getroot().findall('item')]
    elif path.suffix == '.parquet':
        import pyarrow.parquet as pq
        rows = pq.read_table(path).to_pylist()
    else:
        import pyarrow as pa
        with pa.memory_map(str(path)) as stream:
            rows = pa.ipc.open_stream(stream).read_all().to_pylist()
    if len(rows) != item['full_documents']:
        raise ValueError(f'incomplete source split: {path}')
    return rows


def render_document(benchmark, language, doc, xcopa):
    if benchmark == 'belebele_cloze':
        context = f"P: {doc['flores_passage']}\nQ: {doc['question'].strip()}\nAnswer:"
        choices = [doc[f'mc_answer{i}'] for i in range(1, 5)]
        target = int(doc['correct_answer_num']) - 1
    elif benchmark == 'okapi_hellaswag':
        local = lambda value: _localized(value, language)
        clean = _preprocess_hellaswag_text
        context = clean(local(doc['activity_label']) + ': ' + local(doc['ctx_a']) + ' ' + local(doc['ctx_b']).capitalize())
        choices = [clean(x) for x in local(doc['endings'])]
        target = int(doc['label'])
    elif benchmark == 'xstorycloze':
        context = ' '.join(doc[f'input_sentence_{i}'] for i in range(1, 5))
        choices = [doc['sentence_quiz1'], doc['sentence_quiz2']]
        target = int(doc['answer_right_ending']) - 1
    elif benchmark == 'xcopa':
        context = (getattr(xcopa, f'doc_to_text_{language}')(doc) if language != 'en' else
                   doc['premise'].strip()[:-1] + (' because' if doc['question'] == 'cause' else ' therefore'))
        choices = xcopa.doc_to_choice(doc)
        target = int(doc['label'])
    elif benchmark == 'uhura_arc_easy':
        context = f"Question: {doc['question'].strip()}\nAnswer:"
        choices = doc['choices']['text']
        target = list(doc['choices']['label']).index(doc['answerKey'])
    elif benchmark == 'global_mmlu_full':
        context = f"Question: {doc['question'].strip()}\nAnswer:"
        choices = [doc[f'option_{x}'] for x in 'abcd']
        target = _global_mmlu_target(doc)
    else:
        raise ValueError(f'unknown benchmark: {benchmark}')
    if not choices or any(not isinstance(x, str) or not x.strip() for x in choices):
        raise ValueError('empty or invalid option')
    if not 0 <= target < len(choices):
        raise ValueError('gold answer outside option range')
    return context, choices, target


def load_xcopa(harness):
    path = Path(harness) / 'lm_eval/tasks/xcopa/utils.py'
    spec = importlib.util.spec_from_file_location('stage1_full_xcopa', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def freeze_language(args):
    """Freeze one complete language and audit every encoded request on CPU."""
    item, output, harness, tokenizer_cfg, sequence_length = args
    from collections import Counter
    import sentencepiece as spm
    from .requests import _request_id
    from multilingual_megatron.pipeline.stages.benchmarking.native_megatron_scoring import encode_lm_eval_pair, prepare_scored_sequence
    from multilingual_megatron.pipeline.stages.benchmarking.native_token_cache import TokenCacheWriter

    implementation = spm.SentencePieceProcessor(model_file=tokenizer_cfg['model'])
    tokenizer = SimpleNamespace(tokenize=implementation.encode)
    xcopa = load_xcopa(harness)
    benchmark, language = item['benchmark'], item['language']
    task = f'stage1_full_{benchmark}_{language}'
    cache = TokenCacheWriter(Path(output) / 'token_cache' / task)
    rows, documents, histogram, long_targets = [], [], Counter(), []
    allowed = {row['doc_id']: row for row in item.get('approved_exclusions', [])}
    exclusions = []
    truncated = 0
    max_target = 0
    for doc_id, doc in enumerate(read_documents(item)):
        try:
            context, choices, target = render_document(benchmark, language, doc, xcopa)
        except ValueError as exc:
            approved = allowed.get(doc_id)
            if approved is None or approved['document_sha256'] != canonical_json_sha256(doc) or str(exc) != approved['reason']:
                raise ValueError(f'{benchmark}/{language}/{doc_id}: {exc}') from exc
            exclusions.append(approved)
            continue
        if doc_id in allowed:
            raise ValueError('approved exclusion no longer reproduces its documented source error')
        documents.append({'doc_id': doc_id, 'target': target, 'choice_lengths': [len(x) for x in choices],
                          **({'subject': doc['subject']} if 'subject' in doc else {})})
        for index, choice in enumerate(choices):
            continuation = ' ' + choice
            context_ids, target_ids = encode_lm_eval_pair(tokenizer, context, continuation,
                                                        context_token_id=tokenizer_cfg['context_token_id'])
            width = len(context_ids) + len(target_ids) - 1
            histogram[min(sequence_length, width)] += 1
            truncated += width > sequence_length
            max_target = max(max_target, len(target_ids))
            if len(target_ids) > sequence_length:
                long_targets.append({'doc_id': doc_id, 'choice_index': index, 'target_tokens': len(target_ids)})
            rows.append({'request_id': _request_id(task, doc_id, index, (context, continuation)),
                         'task': task, 'benchmark': benchmark, 'language': language,
                         'doc_id': doc_id, 'choice_index': index, 'request_type': 'loglikelihood',
                         'context': context, 'continuation': continuation})
            cache.append(prepare_scored_sequence(request_index=len(rows) - 1, request_id=rows[-1]['request_id'],
                         context_ids=context_ids, continuation_ids=target_ids, max_length=sequence_length))
    if {x['doc_id'] for x in exclusions} != set(allowed):
        raise ValueError('approved exclusion did not match a source row')
    metadata = {'task': task, 'benchmark': benchmark, 'language': language, 'output_type': 'multiple_choice',
                'documents': documents, 'source': item, 'exclusions': exclusions, 'metrics': ['acc', 'acc_norm']}
    payload = {'schema': 'stage1_lm_eval_frozen_requests/v1', 'protocol': 'zero_shot_full_option_cloze',
               'sampling': None, 'tasks': [metadata], 'requests': rows}
    payload['manifest_sha256'] = canonical_json_sha256(payload)
    path = Path(output) / 'languages' / (task + '.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, separators=(',', ':')) + '\n')
    cache_ref = cache.finish(request_sha256=sha256_file(path), tokenizer=tokenizer_cfg, sequence_length=sequence_length)
    receipt = {'benchmark': benchmark, 'language': language, 'documents': len(documents), 'requests': len(rows),
               'path': str(path), 'sha256': sha256_file(path), 'width_histogram': {str(k): v for k, v in sorted(histogram.items())},
               'context_left_truncated_requests': truncated, 'max_target_tokens': max_target,
               'unsupported_long_targets': long_targets, 'exclusions': exclusions, 'token_cache': cache_ref}
    audit = path.with_suffix('.audit.json')
    audit.write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


def aggregate_metrics(manifest, scores):
    """Use harness-compatible first-argmax and raw option character lengths."""
    import math
    expected = {row['request_id']: row for row in manifest['requests']}
    grouped = {}
    seen = set()
    for row in scores:
        key = row['request_id']
        if key not in expected or key in seen or not math.isfinite(row['loglikelihood']):
            raise ValueError('invalid, duplicate, or unexpected score')
        seen.add(key)
        request = expected[key]
        grouped.setdefault((request['task'], request['doc_id']), {})[request['choice_index']] = row['loglikelihood']
    if seen != expected.keys():
        raise ValueError('incomplete scores')
    result = []
    for task in manifest['tasks']:
        raw, normalized = 0, 0
        for doc in task['documents']:
            values = grouped[(task['task'], doc['doc_id'])]
            lengths = doc['choice_lengths']
            if set(values) != set(range(len(lengths))) or any(n <= 0 for n in lengths):
                raise ValueError('incomplete option group or invalid normalization')
            raw += max(range(len(lengths)), key=lambda i: values[i]) == doc['target']
            normalized += max(range(len(lengths)), key=lambda i: values[i] / lengths[i]) == doc['target']
        count = len(task['documents'])
        result.append({'task': task['task'], 'benchmark': task['benchmark'], 'language': task['language'],
                       'samples': count, 'acc': raw / count, 'acc_norm': normalized / count})
    return result


def main():
    import argparse
    from concurrent.futures import ProcessPoolExecutor, as_completed
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--harness', type=Path, required=True)
    parser.add_argument('--score-spec', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--approved-exclusions', type=Path)
    args = parser.parse_args()
    spec = json.loads(args.score_spec.read_text())
    items = json.loads(args.inventory.read_text())
    if args.approved_exclusions:
        approved = json.loads(args.approved_exclusions.read_text())
        for item in items:
            item['approved_exclusions'] = [dict(doc_id=row['doc_id'], reason=row['error'],
                document_sha256=canonical_json_sha256(row['doc']), source_sha256=item['source_sha256'])
                for row in approved if (row['benchmark'], row['language']) == (item['benchmark'], item['language'])]
        if sum(len(x['approved_exclusions']) for x in items) != len(approved):
            raise ValueError('exclusions reference absent source languages')
    if len({(x['benchmark'], x['language']) for x in items}) != len(items):
        raise ValueError('duplicate inventory entries')
    if sha256_file(Path(spec['tokenizer']['model'])) != spec['tokenizer']['sha256']:
        raise ValueError('tokenizer drift')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    receipts = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(freeze_language, (item, str(output), str(args.harness.resolve()),
                              spec['tokenizer'], spec['evaluation']['sequence_length'])) for item in items]
        for future in as_completed(futures):
            receipt = future.result()
            receipts.append(receipt)
            print(receipt['benchmark'], receipt['language'], receipt['documents'],
                  'long_targets', len(receipt['unsupported_long_targets']), flush=True)
    receipts.sort(key=lambda x: (x['benchmark'], x['language']))
    families = []
    for benchmark in sorted({x['benchmark'] for x in receipts}):
        from multilingual_megatron.pipeline.stages.benchmarking.native_token_cache import combine_token_caches
        tasks, requests = [], []
        for entry in receipts:
            if entry['benchmark'] != benchmark:
                continue
            part = json.loads(Path(entry['path']).read_text())
            tasks.extend(part['tasks'])
            requests.extend(part['requests'])
        payload = {'schema': 'stage1_lm_eval_frozen_requests/v1', 'sampling': None,
                   'protocol': 'zero_shot_full_option_cloze', 'tasks': tasks, 'requests': requests}
        payload['manifest_sha256'] = canonical_json_sha256(payload)
        path = output / (benchmark + '.requests.json')
        path.write_text(json.dumps(payload, ensure_ascii=False, separators=(',', ':')) + '\n')
        cache_ref = combine_token_caches([x['token_cache'] for x in receipts if x['benchmark'] == benchmark],
            output / 'token_cache' / benchmark, request_sha256=sha256_file(path),
            tokenizer=spec['tokenizer'], sequence_length=spec['evaluation']['sequence_length'])
        families.append({'benchmark': benchmark, 'path': str(path), 'sha256': sha256_file(path), 'token_cache': cache_ref,
                         'languages': len(tasks), 'documents': sum(len(x['documents']) for x in tasks),
                         'requests': len(requests)})
        print('family complete', benchmark, len(requests), flush=True)
        del payload, requests, tasks, part
    receipt = {'schema': 'stage1_full_suite_data/v1', 'sampling': None, 'families': families,
               'languages': receipts, 'inventory': {'path': str(args.inventory.resolve()), 'sha256': sha256_file(args.inventory)},
               'tokenizer': spec['tokenizer'], 'sequence_length': spec['evaluation']['sequence_length'],
               'xcopa_renderer_sha256': sha256_file(args.harness / 'lm_eval/tasks/xcopa/utils.py'),
               'unsupported_long_targets': sum(len(x['unsupported_long_targets']) for x in receipts)}
    receipt['excluded_documents'] = sum(len(x['exclusions']) for x in receipts)
    if args.approved_exclusions:
        receipt['approved_exclusions'] = {'path': str(args.approved_exclusions.resolve()),
                                        'sha256': sha256_file(args.approved_exclusions)}
    receipt['manifest_sha256'] = canonical_json_sha256(receipt)
    (output / 'data_manifest.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print('complete', str(output / 'data_manifest.json'), flush=True)


if __name__ == '__main__':
    main()
