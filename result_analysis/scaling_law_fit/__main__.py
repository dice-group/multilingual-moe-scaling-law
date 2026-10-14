from result_analysis.cli import module_main
from .reproduce import run
from .analysis import from_parquet

if __name__ == "__main__":
    module_main(run, "Replay certified scaling laws and score held-out predictions on CPU.", from_parquet)
