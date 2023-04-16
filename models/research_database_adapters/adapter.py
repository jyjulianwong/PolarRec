"""
Research database adapter interface.
"""


class QueryBuilder:
    """
    Builds and sends queries for a research database.
    """

    def __init__(self):
        pass

    def set_authors(self, authors):
        pass

    def set_min_date(self, min_date):
        pass

    def set_max_date(self, max_date):
        pass

    def add_keyword(self, keyword):
        pass

    def add_keywords(self, keywords):
        for keyword in keywords:
            self.add_keyword(keyword)

    def get_resources(
        self,
        max_resources_returned,
        must_have_all_fields=True
    ):
        """
        Sends a GET request to the database's API and returns the response.

        :return: Response
        """
        pass
