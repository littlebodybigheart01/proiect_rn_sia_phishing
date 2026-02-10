"""Compatibility wrapper for report visualizations.

This module exists to keep repository structure aligned with the RN template.
It delegates to report_assets.py, which generates all Stage 6 plots.
"""

from src.neural_network.report_assets import main


if __name__ == "__main__":
    main()
