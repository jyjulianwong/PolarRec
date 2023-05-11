"""
Resource database adapter interface.
"""


class QueryBuilder:
    """
    Builds and sends queries for a resource database.
    """

    def __init__(self):
        pass

    def _get_translated_url_str(self, url_str):
        """
        Replaces special characters that would be unwanted in a URL string, such
        as particular German characters that commonly appear in authors' names.

        :param url_str: The original URL string
        :return: The URL string with special characters replaced
        """
        german_special_char_dict = {
            ord("ä"): "ae",
            ord("ü"): "ue",
            ord("ö"): "oe",
            ord("ß"): "ss"
        }
        return url_str.translate(german_special_char_dict)

    def _get_author_last_name(self, author_name):
        """
        :param author_name: The author's name.
        :type author_name: str
        :return: The author's last name.
        :rtype: str
        """
        author_name_parts = author_name.split(" ")
        return author_name_parts[-1]

    def _get_joined_author_name(self, author_name, join_char, flipped=False):
        """
        Joins the first and last names of authors' names appropriately for use
        in database queries and in URLs.

        :param author_name: The author's name.
        :type author_name: str
        :param join_char: The character used to join first and last names.
        :type join_char: str
        :param flipped: Flip the last names and first names around.
        :type flipped: bool
        :return: The URL-friendly string representing the author's name.
        :rtype: str
        """
        author_name_parts = author_name.split(" ")
        author_last_name = author_name_parts[-1]
        author_first_name = author_name_parts[0]
        if flipped:
            formatted_name = f"{author_last_name}{join_char}{author_first_name}"
        else:
            formatted_name = f"{author_first_name}{join_char}{author_last_name}"
        if "." in author_first_name:
            # Do not include first names that have been shortened.
            # This is usually not supported by resource database adapters.
            formatted_name = author_last_name
        formatted_name = formatted_name.lower()
        return formatted_name

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
