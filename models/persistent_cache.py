"""
Definition of the PersistentCache class.
"""
import json
import os
from models.custom_logger import log
from models.hyperparams import Hyperparams as hp


class PersistentCache:
    """
    Methods related to saving and loading data, typically API response data,
    to and from local JSON cache files.
    """

    @classmethod
    def cache_enabled(cls):
        """
        :return: Whether cache is enabled in the current environment or not.
        :rtype: bool
        """
        return not hp.DISABLE_CACHED_DATA

    @classmethod
    def load_cache_file(cls, filepath):
        """
        To be used only if cache is enabled in the current environment.

        :param filepath: The filepath of the cache file.
        :type filepath: str
        :return: The JSON data.
        :rtype: dict
        """
        if not os.path.isfile(filepath):
            log(
                f"{filepath} is not a valid filepath",
                "DevCache",
                "error"
            )
            return {}

        with open(filepath, "r", encoding="utf-8") as file_object:
            json_data = json.load(file_object)
            return json_data

    @classmethod
    def save_cache_file(cls, filepath, data):
        """
        To be used only if cache is enabled in the current environment.

        :param filepath: The filepath of the cache file.
        :type filepath: str
        :param data: The JSON data.
        :type data: dict
        """
        json_data = json.dumps(data, indent=4)
        with open(filepath, "w+", encoding="utf-8") as file_object:
            file_object.write(json_data)
