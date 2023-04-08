"""
Objects and methods for the /recommend service of the API.
"""


class Resource:
    """
    An academic resource object.
    """

    def __init__(
        self,
        title=None,
        authors=None,
        date=None,
        abstract=None,
        introduction=None,
        doi=None,
        url=None,
    ):
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
        :param doi: The DOI number of the resource.
        :type doi: str
        "param url: The URL to the resource online.
        :type url: str
        """
        self.title = title
        self.authors = authors
        self.date = date  # TODO: Convert to datetime type.
        self.abstract = abstract
        self.introduction = introduction
        self.doi = doi
        self.url = url


def recommend(resources):
    """
    Calls research database APIs and recommends a list of academic resources
    based on target resources.

    :param resources: The list of target resources.
    :type resources: list[Resource]
    :return: A list of recommended academic resources.
    """
    for resource in resources:
        pass
    pass
