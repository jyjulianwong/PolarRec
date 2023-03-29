"""
Objects and methods for the /recommend service of the API.
"""


class TargetResource:
    """
    A target academic resource object.
    """

    def __init__(self, title, authors, date, abstract, introduction):
        """
        :param title: The title of the resource.
        :type title: str
        :param authors: The authors of the resource.
        :type authors: list[str]
        :param date: The date of resource publication.
        :type date: str
        :param abstract: The abstract from the resource.
        :type abstract: str
        :param introduction: The introduction section from the resource.
        :type introduction: str
        """
        self.title = title
        self.authors = authors
        self.date = date  # TODO: Convert to datetime type.
        self.abstract = abstract
        self.introduction = introduction


def recommend(resources):
    """
    Calls research database APIs and recommends a list of academic resources
    based on target resources.

    :param resources: The list of target resources.
    :type resources: list[TargetResource]
    :return: A list of recommended academic resources.
    """
    pass
