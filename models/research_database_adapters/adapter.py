"""
Research database adapter interface.
"""


class QueryBuilder:
    """
    Builds and sends queries for a research database.
    """

    def __init__(self):
        self._MAX_RES_ITEMS_COUNT = 10

    def set_author(self, author):
        pass

    def set_min_date(self, min_date):
        pass

    def set_max_date(self, max_date):
        pass

    def add_keyword(self, keyword):
        pass

    def get_resources(self):
        """
        Sends a GET request to the database's API and returns the response.

        :return: Response
        """
        pass
