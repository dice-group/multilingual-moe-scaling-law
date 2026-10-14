from result_analysis.cli import module_main
from .reproduce import run
from .analysis import from_parquet

if __name__ == "__main__":
    module_main(run, "Recompute observed and fitted capacity allocation on CPU.", from_parquet)
