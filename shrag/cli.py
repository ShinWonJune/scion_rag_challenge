"""Console-script entry for `shrag`.

Re-exports `shrag.pipeline.run:main` so that `pip install -e .` registers a
single `shrag` binary that delegates to the 5-step pipeline orchestrator.
"""

from shrag.pipeline.run import main

if __name__ == "__main__":
    main()
