"""Score the main prediction head while retaining all checkpoint parameters."""

from contextlib import contextmanager


@contextmanager
def main_prediction_only(models):
    """Temporarily skip training-only MTP computation after checkpoint loading.

    MCore's ordinary evaluation forward still executes MTP unless both the
    module flag and the loss-processing count are disabled. Leave the MTP
    modules/weights intact, and restore the flags even when evaluation fails.
    """
    changes = []
    seen = set()
    try:
        for model in models:
            for module in model.modules():
                if getattr(module, "mtp_process", False):
                    config = module.config
                    if id(config) not in seen and not getattr(config, "mtp_num_layers", None):
                        raise ValueError("MTP module lacks its layer-count configuration")
                    changes.append((module, "mtp_process", module.mtp_process))
                    module.mtp_process = False
                    if id(config) not in seen:
                        changes.append((config, "mtp_num_layers", config.mtp_num_layers))
                        seen.add(id(config))
                        config.mtp_num_layers = None
        yield
    finally:
        for owner, field, value in reversed(changes):
            setattr(owner, field, value)
