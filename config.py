"""
Constants and methods for retrieving environment variables.
"""
import os


class Config:
    """
    Constants and methods for retrieving environment variables.
    """
    APP_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    IEEE_XPLORE_API_KEY = os.environ.get("PLR_IEEE_XPLORE_API_KEY")
    S2_API_KEY = os.environ.get("PLR_S2_API_KEY")
