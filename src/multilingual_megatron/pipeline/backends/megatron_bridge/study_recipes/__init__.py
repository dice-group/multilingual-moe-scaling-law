"""Architecture-family modules for study-local Bridge recipes.

The import shims must be installed before any family imports Megatron or
Bridge. Keeping that bootstrap at the package boundary also makes direct
family-module imports behave like the compatibility facade.
"""

from ..megatron_training_import_shim import (
    install_bridge_mamba_model_import_shim,
    install_bridge_mimo_model_import_shim,
    install_bridge_packed_sequence_specs_import_shim,
    install_bridge_pretraining_dataset_import_shims,
    install_megatron_training_config_import_shim,
    install_mcore_cuda_graphs_import_shim,
)


install_megatron_training_config_import_shim()
install_bridge_packed_sequence_specs_import_shim()
install_bridge_pretraining_dataset_import_shims()
install_bridge_mamba_model_import_shim()
install_bridge_mimo_model_import_shim()
install_mcore_cuda_graphs_import_shim()
