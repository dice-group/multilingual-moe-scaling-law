"""Run equation fits from verified private source bytes, not a changing checkout."""
import hashlib
from pathlib import Path
import tarfile

from .. import atlas_full_pilot as io


def freeze(root, paths, repository, *, content_sources=None):
    runtime = root/"runtime"
    bindings = []
    for original in sorted(set(paths)):
        relative = original.relative_to(repository)
        content = (content_sources or {}).get(original, original).read_bytes()
        target = runtime/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as handle:
            handle.write(content)
        bound = io.bind(target)
        if bound["file_sha256"] != hashlib.sha256(content).hexdigest():
            raise ValueError("source snapshot write did not preserve exact bytes")
        bindings.append(bound | dict(original_path=str(original)))
    archive = root/"source.tar.gz"
    with tarfile.open(archive, "x:gz") as tar:
        for bound in bindings:
            tar.add(bound["path"], arcname=str(Path(bound["path"]).relative_to(runtime)), recursive=False)
    return runtime, bindings, io.bind(archive)


def require_frozen_module(plan, module_path):
    root = Path(plan["runtime_root"])
    expected = root/"src/multilingual_megatron/experiments/scaling/stage1_fit/atlas_comparison_workflow/campaign.py"
    if Path(module_path).resolve() != expected:
        raise ValueError("use the campaign's run.sh to execute its isolated frozen source")
