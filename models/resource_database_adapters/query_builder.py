"""
Resource database adapter interface.
"""
import unicodedata
from models.custom_logger import log, log_extended_line
from models.resource import Resource


class QueryBuilder:
    """
    Builds and sends search queries to a resource database.
    """

    @classmethod
    def get_id(cls):
        """
        :return: The custom ID of the QueryBuilder class for filtering purposes.
        :rtype: str
        """
        pass

    def __init__(self):
        pass

    def _get_translated_url_str(self, url_str):
        """
        A helper function that replaces special characters that would be
        unwanted in a URL string, such as particular German characters that
        commonly appear in authors' names.

        :param url_str: The original URL string
        :type url_str: str
        :return: The URL string with special characters replaced
        :rtype: str
        """
        german_special_char_dict = {
            ord("ä"): "ae",
            ord("ü"): "ue",
            ord("ö"): "oe",
            ord("ß"): "ss"
        }
        # Replace all German-specific characters with correct translations.
        result = url_str.translate(german_special_char_dict)
        # Replace accented characters with non-accented equivalents.
        result = unicodedata.normalize("NFKD", result)
        # Ignore all other characters that cannot be translated, e.g. Kanji.
        result = result.encode("ascii", "ignore").decode("ascii")
        return result

    def _get_author_last_name(self, author_name):
        """
        A helper function that extracts an author's last name from their full
        names for different name formats.

        :param author_name: The author's name.
        :type author_name: str
        :return: The author's last name.
        :rtype: str
        """
        author_name_parts = author_name.split(" ")
        return author_name_parts[-1]

    def _get_joined_author_name(self, author_name, join_char, flipped=False):
        """
        A helper function that joins the first and last names of authors' names
        appropriately for use in database queries and in URLs.

        :param author_name: The author's name.
        :type author_name: str
        :param join_char: The character used to join first and last names.
        :type join_char: str
        :param flipped: Whether to flip the last names and first names around.
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
            # This is usually not supported by database search interfaces.
            formatted_name = author_last_name

        # Convert the string into a lowercase, non-punctuated string.
        formatted_name = Resource.get_comparable_str(formatted_name)
        formatted_name = formatted_name.replace(" ", "+")

        return formatted_name

    def _summarise_results_data(self, resources):
        """
        A helper function that prints a summary of the results returned by the
        resource database search query.
        Analyses the percentage of data fields returned by the query.

        :param resources: The resources returned by the query.
        :type resources: list[Resource]
        """
        authors_count = 0
        title_count = 0
        year_count = 0
        month_count = 0
        abstract_count = 0
        doi_count = 0
        url_count = 0

        for resource in resources:
            if resource.authors is not None:
                authors_count += 100
            if resource.title is not None:
                title_count += 100
            if resource.year is not None:
                year_count += 100
            if resource.month is not None:
                month_count += 100
            if resource.abstract is not None:
                abstract_count += 100
            if resource.doi is not None:
                doi_count += 100
            if resource.url is not None:
                url_count += 100

        log("Summary of fields returned by query:", self.__class__.__name__)
        # Set minimum denominator to avoid division-by-zero.
        resource_count = max(len(resources), 1)
        log_extended_line(f"% with authors: {authors_count / resource_count}")
        log_extended_line(f"% with title: {title_count / resource_count}")
        log_extended_line(f"% with year: {year_count / resource_count}")
        log_extended_line(f"% with month: {month_count / resource_count}")
        log_extended_line(f"% with abstract: {abstract_count / resource_count}")
        log_extended_line(f"% with doi: {doi_count / resource_count}")
        log_extended_line(f"% with url: {url_count / resource_count}")

    def set_authors(self, authors):
        pass

    def set_title(self, title):
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
        must_have_all_fields=True,
        summarise_results_data=False
    ):
        """
        Sends an HTTP request to the database's API and returns the response.

        :param max_resources_returned: The maximum number of results to return.
        :type max_resources_returned: int
        :param must_have_all_fields: Whether to filter results that do not meet all query requirements.
        :type must_have_all_fields: bool
        :param summarise_results_data: whether to print a summary of results.
        :type summarise_results_data: bool
        :return: The list of resources returned by the query.
        :rtype: list[Resource]
        """
        pass
