"""
Definition of the ResourceFilter class.
"""
import math
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
        A helper function that extracts an author's last name from their full
        names for different name formats.

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
            if required_last_name in cand_last_names:
                return True
        return False

    def _conference_names_match(self, conference_name):
        """
        :param conference_name: The conf. name from the candidate resource.
        :type conference_name: str
        :return: Whether the conf. name matches the requirements of the filter.
        :rtype: bool
        """
        # Remove punctuation or capitalisation that may affect this comparison.
        targ_conf_name = Resource.get_comparable_str(self.conference_name)
        cand_conf_name = Resource.get_comparable_str(conference_name)

        # Check if there are any overlapping words between target and candidate.
        targ_conf_words = set(targ_conf_name.split())
        cand_conf_words = cand_conf_name.split()
        common_words = list(targ_conf_words.intersection(cand_conf_words))

        # Consider a "match" if overlapping words are sufficient in quantity.
        return len(common_words) >= math.floor(0.5 * len(targ_conf_words))

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
                elif not self._author_lists_match(resource.authors):
                    remove = True

            if self.conference_name is not None:
                # A filter for matching conference name is required.
                if resource.conference_name is None:
                    # The candidate resource has no recorded conference name.
                    remove = True
                elif not self._conference_names_match(resource.conference_name):
                    remove = True

            if not remove:
                # The candidate passes all the filter requirements.
                filtered_ress.append(resource)

        return filtered_ress


if __name__ == "__main__":
    # Instantiate example objects for testing.
    author_list1 = ["A", "B", "C"]
    author_list2 = ["C", "D", "E"]

    conf_name1 = "2023 IEEE Nuclear and Space Radiation Effects Conference"
    conf_name2 = "IEEE Nuclear and Space Radiation Effects '23"

    res_filter = ResourceFilter({
        "authors": author_list1,
        "conference_name": conf_name1
    })

    print("\nResourceFilter: Matching two author lists")
    match = res_filter._author_lists_match(author_list2)
    print(f"ResourceFilter: Author lists match: {match}")

    print("\nResourceFilter: Matching two conference names")
    match = res_filter._conference_names_match(conf_name2)
    print(f"ResourceFilter: Conference names match: {match}")
