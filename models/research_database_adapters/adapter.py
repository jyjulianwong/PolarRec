"""
Research database adapter interface.
"""


class QueryBuilder:
    """
    Builds and sends queries for a research database.
    """

    def __init__(self):
        pass

    def set_author(self, author):
        pass

    def set_start_date(self, start_date):
        pass

    def set_end_date(self, end_date):
        pass

    def add_keyword(self, keyword):
        pass

    def get_response(self):
        """
        Sends a GET request to the database's API and returns the response.

        :return: Response
        """
        pass
