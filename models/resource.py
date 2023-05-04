"""
Definition of the Resource (an academic resource object) class.
"""
import datetime


class Resource:
    """
    An academic resource object, designed to be initialised from JSON data.
    Unifies metadata to a singular representation for every type of resource.
    Converts JSON data into Python object instance in a safe manner.
    Finds non-existant fields in JSON data and assigns them as None in instance.
    """

    def __init__(self, args):
        """
        :param args: A JSON-like dictionary that contains the following fields:
            ``authors: list[str]`` (the authors of the resource),
            ``conference_name: str`` (the name of the associated conference),
            ``conference_location: str`` (the location of the associated conference),
            ``title: str`` (the title of the resource),
            ``year: int | str`` (the year of resource publication),
            ``month: int | str`` (the month of resource publication),
            ``abstract: str`` (the abstract from the resource),
            ``doi: str`` (the DOI number of the resource),
            ``url: str`` (the URL to the resource online),
            ``references: list[Resource]`` (the references used by resource).
        :raises ValueError: When value of arguments are invalid.
        """
        # Type validation and conversion for year parameter.
        if "year" in args:
            if isinstance(args["year"], str):
                try:
                    args["year"] = int(args["year"])
                except ValueError:
                    raise ValueError("Value of year must be a number")
            elif not isinstance(args["year"], int):
                raise ValueError("Value of year must be a number")

            if args["year"] < 0 or args["year"] > datetime.date.today().year:
                raise ValueError("Value of year is outside of valid range")

        # Type validation and conversion for month parameter.
        if "month" in args:
            if isinstance(args["month"], str):
                try:
                    args["month"] = int(args["month"])
                except ValueError:
                    raise ValueError("Value of month must be a number")
            elif not isinstance(args["month"], int):
                raise ValueError("Value of month must be a number")

            if args["month"] < 1 or args["month"] > 12:
                raise ValueError("Value of month is outside of valid range")

        self.authors = None
        if "authors" in args:
            self.authors = args["authors"]
        self.conference_name = None
        if "conference_name" in args:
            self.conference_name = args["conference_name"]
        self.conference_location = None
        if "conference_location" in args:
            self.conference_location = args["conference_location"]
        self.title = None
        if "title" in args:
            self.title = args["title"]
        self.year = None
        if "year" in args:
            self.year = args["year"]
        self.month = None
        if "month" in args:
            self.month = args["month"]
        self.abstract = None
        if "abstract" in args:
            self.abstract = args["abstract"]
        self.doi = None
        if "doi" in args:
            self.doi = args["doi"]
        self.url = None
        if "url" in args:
            self.url = args["url"]
        self.references = None
        if "references" in args:
            self.references = args["references"]

    def __eq__(self, other):
        # The "identity" of an academic resource is defined by its full title.
        # Low likelihood of two well-known academic resources having same title.
        return self.title == other.title

    def __lt__(self, other):
        return self.title < other.title

    def __hash__(self):
        return hash(self.title)

    def to_dict(self):
        """
        Returns the data in the Resource instance in the form of a dict.

        :return: The data in the Resource instance in the form of a dict.
        :rtype: dict
        """
        as_dict = {}
        if self.authors is not None:
            as_dict["authors"] = self.authors
        if self.conference_name is not None:
            as_dict["conference_name"] = self.conference_name
        if self.conference_location is not None:
            as_dict["conference_location"] = self.conference_location
        if self.title is not None:
            as_dict["title"] = self.title
        if self.year is not None:
            as_dict["year"] = self.year
        if self.month is not None:
            as_dict["month"] = self.month
        if self.abstract is not None:
            as_dict["abstract"] = self.abstract
        if self.doi is not None:
            as_dict["doi"] = self.doi
        if self.url is not None:
            as_dict["url"] = self.url
        if self.references is not None:
            as_dict["references"] = []
            for reference in self.references:
                as_dict["references"].append(reference.to_dict())
        return as_dict


if __name__ == "__main__":
    resource_dict1 = {
        "authors": ["Y. Lecun", "L. Bottou", "Y. Bengio", "P. Haffner"],
        "title": "Gradient-based learning applied to document recognition",
        "year": 1998,
        "abstract": "Multilayer neural networks trained with...",
        "doi": "10.1109/5.726791",
        "url": "https://ieeexplore.ieee.org/document/726791"
    }
    resource = Resource(resource_dict1)
    resource_dict2 = resource.to_dict()
    print(f"dict1 == dict2: {resource_dict1 == resource_dict2}")
