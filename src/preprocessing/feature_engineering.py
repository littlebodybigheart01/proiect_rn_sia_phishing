"""Stage-3 compatible feature engineering entrypoint.

Text normalization and filtering are performed in preprocess_and_split.py.
"""

from src.preprocessing.preprocess_and_split import preprocess_and_split


if __name__ == "__main__":
    preprocess_and_split()
