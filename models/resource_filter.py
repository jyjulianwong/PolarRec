"""
Definition of the ResourceFilter class.
"""


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
