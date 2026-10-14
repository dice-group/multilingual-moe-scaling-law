"""Forecast complete Stage 1 evaluation from measured H100 architecture cells."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from statistics import median

from multilingual_megatron.contracts.hashing import sha256_file


def estimate(models, data, pilot_root):
    """Retain length-aware and request-count extrapolations, plus explicit overhead."""
    import numpy as np
    histograms = defaultdict(lambda: defaultdict(int))
    for language in data['languages']:
        for width, count in language['width_histogram'].items():
            histograms[language['benchmark']][int(width)] += count
    calibrations = {}
    for model in models:
        key = (model['ladder'], model['E'])
        if key in calibrations:
            continue
        phase = f'{key[0].lower()}_e{key[1]}_expanded_fixed'
        path = Path(pilot_root) / 'scores' / phase / 'scores.json'
        score = json.loads(path.read_text())
        if score['status'] != 'complete' or not score['parity_gate']['passed'] or not score['replica_gate']['passed']:
            raise ValueError(f'invalid calibration: {path}')
        timing = score['timing']
        batches = timing['forward_batches']
        batch_size = score['batching']['micro_batch_size']
        # Quantiles of sequence length smooth timer jitter while retaining the
        # measured fixed-batch runtime, including the actual routing kernels.
        grouped = defaultdict(list)
        for row in batches[1:]:
            if row['requests'] == batch_size:
                grouped[(row['width'] + 31) // 32].append(row)
        points = sorted((median(x['width'] for x in rows), median(x['seconds'] for x in rows) / batch_size)
                        for rows in grouped.values())
        count = score['counts']['requests']
        python_per_request = max(0, timing['scoring_seconds'] - sum(x['seconds'] for x in batches)) / count
        launcher = json.loads((path.parent / 'launcher_result.json').read_text())
        # Explicit allowance for full-size JSON/cache validation and output I/O;
        # startup itself is anchored to the measured complete wrapper runtime.
        startup = max(120, launcher['runtime_seconds'] - timing['scoring_seconds'])
        calibrations[key] = dict(points=points, batch_size=batch_size,
            measured_scoring_seconds=timing['scoring_seconds'], measured_requests=count,
            python_seconds_per_request=python_per_request, startup_seconds=startup,
            source={'path': str(path), 'sha256': sha256_file(path)},
            peak_vram_bytes=timing['peak_vram_bytes'])
    estimates = []
    for model in models:
        cal = calibrations[(model['ladder'], model['E'])]
        xs, ys = zip(*cal['points'])
        families = {}
        total_requests = sum(sum(h.values()) for h in histograms.values())
        for benchmark, histogram in histograms.items():
            length_seconds = 0.0
            for width, count in histogram.items():
                rate = float(np.interp(width, xs, ys))
                if width > xs[-1]:
                    rate *= (width / xs[-1]) ** 2
                length_seconds += count * (rate + cal['python_seconds_per_request'])
            request_seconds = sum(histogram.values()) * cal['measured_scoring_seconds'] / cal['measured_requests']
            overhead = cal['startup_seconds'] + 300 * sum(histogram.values()) / total_requests
            low = min(length_seconds, request_seconds) + overhead
            high = max(length_seconds, request_seconds) + overhead
            families[benchmark] = {'length_seconds': length_seconds, 'request_seconds': request_seconds,
                                   'fixed_overhead_seconds': overhead, 'h100_low_hours': low / 3600,
                                   'h100_high_hours': high / 3600, 'reserve_hours': high * 1.5 / 3600}
        low = sum(x['h100_low_hours'] for x in families.values()) * 3600
        high = sum(x['h100_high_hours'] for x in families.values()) * 3600
        estimates.append({**model, 'h100_low_hours': low / 3600, 'h100_high_hours': high / 3600,
                          'reserve_hours': high * 1.5 / 3600, 'benchmarks_h100': families,
                          'fixed_overhead_seconds': sum(x['fixed_overhead_seconds'] for x in families.values()),
                          'calibration': cal['source'], 'h100_micro_batch_size': cal['batch_size']})
    cohorts = {}
    for name, selected in [('all_82', estimates), ('canonical_64', [x for x in estimates if x['seed'] == 1234]),
                           ('paper_D_16', [x for x in estimates if x['ladder'] == 'D' and x['seed'] == 1234])]:
        cohorts[name] = {'checkpoints': len(selected), **{key: sum(x[key] for x in selected)
                        for key in ('h100_low_hours', 'h100_high_hours', 'reserve_hours')}}
    by_benchmark = {name: {kind: sum(x['benchmarks_h100'][name][kind] for x in estimates)
                           for kind in ('h100_low_hours', 'h100_high_hours', 'reserve_hours')} for name in histograms}
    return {'status': 'measured_H100_calibration_with_full_workload_extrapolation',
            'hardware': 'NVIDIA H100; measured CUDA memory 99951706112 bytes',
            'calibration_cells': len(calibrations), 'cohorts': cohorts, 'benchmarks_hours': by_benchmark,
            'models': estimates, 'calibrations': {f'{k[0]}_E{k[1]}': v for k, v in calibrations.items()},
            'layout': '492 independent jobs: final checkpoint x benchmark; every language of one benchmark stays in that job',
            'method': 'Measured batch-width interpolation versus measured request-count scaling. Each benchmark job gets measured startup (120s minimum); 300s full-suite I/O allowance is divided by request count. Reserve 50 percent above upper estimate.',
            'limits': 'One K200 seed per architecture; K/seed routing differences and shared storage contention remain extrapolations. Bounds are engineering scenarios, not statistical confidence intervals. Pilot allocation cost is additional.'}
