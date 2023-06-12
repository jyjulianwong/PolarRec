"""
Constants and methods for retrieving environment variables.
"""
import os


class Config:
    """
    Constants and methods for retrieving environment variables.
    """
    # The absolute root directory of the application on its local machine.
    APP_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    # The secret API key for IEEE Xplore.
    IEEE_XPLORE_API_KEY = os.environ.get("PLR_IEEE_XPLORE_API_KEY")
    # The secret API key for Semantic Scholar (S2).
    S2_API_KEY = os.environ.get("PLR_S2_API_KEY")
