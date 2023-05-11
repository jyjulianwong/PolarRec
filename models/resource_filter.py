"""
Definition of the ResourceFilter class.
"""
from models.resource import Resource


class ResourceFilter:
    """
    A resource filter object, designed to be initialised from JSON data.
    Represents a filter that can be applied to the recommendation process.
    Converts JSON data into Python object instance in a safe manner.
    Finds non-existant fields in JSON data and assigns them as None in instance.
    """

    def __init__(self, args):
        """
        :param args: A JSON-like dictionary that contains the following fields:
            ``authors: list[str]`` (the authors of the resource),
            ``conference_name: str`` (the name of the associated conference).
        :raises ValueError: When value of arguments are invalid.
        """
        self.authors = None
        if "authors" in args:
            self.authors = args["authors"]
        self.conference_name = None
        if "conference_name" in args:
            self.conference_name = args["conference_name"]

    def _get_author_last_name(self, author_name):
        """
        :param author_name: The author's name.
        :type author_name: str
        :return: The author's last name.
        :rtype: str
        """
        author_name_parts = author_name.split(" ")
        return author_name_parts[-1]

    def _author_lists_match(self, author_list):
        """
        :param author_list: A list of authors from a candidate resource.
        :type author_list: list[str]
        :return: Whether the author list matches the requirements of the filter.
        :rtype: bool
        """
        # Compare the candidate authors to the required ones.
        cand_last_names = [
            self._get_author_last_name(a) for a in author_list
        ]
        for author in self.authors:
            required_last_name = self._get_author_last_name(author)
            # Only compare the last names of two authors.
            # First names are sometimes abbreviated, causing false rejects.
            if required_last_name not in cand_last_names:
                return False
        return True

    def get_filtered(self, resources):
        """
        :param resources: The list of candidate resources.
        :type resources: list[Resource]
        :return: The list of resources that matches the filters.
        :rtype: list[Resource]
        """
        filtered_ress = []

        for resource in resources:
            # Keep track of whether a filter requirement could not be met.
            remove = False

            if self.authors is not None:
                # A filter for matching authors is required.
                if resource.authors is None:
                    # The candidate resource has no recorded authors.
                    remove = True
                else:
                    # A list of candidate authors exists.
                    if not self._author_lists_match(resource.authors):
                        remove = True

            if self.conference_name is not None:
                # A filter for matching conference name is required.
                if resource.conference_name != self.conference_name:
                    remove = True

            if not remove:
                filtered_ress.append(resource)

        return filtered_ress
