"""Leakage-safe recommendation baselines for the HSE RecSys challenge."""


def main() -> None:
    """Run the end-to-end baseline with its default arguments."""
    from recsys.pipeline import main as pipeline_main

    pipeline_main()
