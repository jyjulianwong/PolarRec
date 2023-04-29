"""
Constants and methods for retrieving environment variables.
"""
import os


class Config:
    """
    Constants and methods for retrieving environment variables.
    """
    IEEE_XPLORE_API_KEY = os.environ.get("PLR_IEEE_XPLORE_API_KEY")
