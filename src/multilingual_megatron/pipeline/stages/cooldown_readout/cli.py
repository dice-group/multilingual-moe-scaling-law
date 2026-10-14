from .._cli import run_single_stage


def main() -> None:
    raise SystemExit(run_single_stage("cooldown_readout"))


if __name__ == "__main__":
    main()
