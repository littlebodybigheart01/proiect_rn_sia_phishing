"""Stage-3 compatible data cleaning entrypoint.

For this project, cleaning is executed inside preprocess_and_split.py.
This wrapper keeps template compatibility and a clear CLI.
"""

from src.preprocessing.preprocess_and_split import preprocess_and_split


if __name__ == "__main__":
    preprocess_and_split()
