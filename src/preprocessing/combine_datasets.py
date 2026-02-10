"""Stage-3 compatible dataset combiner entrypoint.

Delegates to merge_all_datasets.py used by the current pipeline.
"""

from src.data_acquisition.merge_all_datasets import main


if __name__ == "__main__":
    main()
