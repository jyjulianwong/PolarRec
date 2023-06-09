"""
Definition of the Resource (an academic resource object) class.
"""
import datetime
import string
import unicodedata


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
            ``references: list[Resource]`` (the references used by resource),
            ``citation_count: int`` (the number of times the resource has been cited).
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

        # Type validation and conversion for citation_count parameter.
        if "citation_count" in args:
            if isinstance(args["citation_count"], str):
                try:
                    args["citation_count"] = int(args["citation_count"])
                except ValueError:
                    raise ValueError("Value of citation_count must be a number")
            elif not isinstance(args["citation_count"], int):
                raise ValueError("Value of citation_count must be a number")

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
        self.citation_count = None
        if "citation_count" in args:
            self.citation_count = args["citation_count"]

    def __eq__(self, other):
        # The "identity" of an academic resource is defined by its full title.
        # Low likelihood of two well-known academic resources having same title.
        return self.get_comparable_title() == other.get_comparable_title()

    def __lt__(self, other):
        return self.get_comparable_title() < other.get_comparable_title()

    def __hash__(self):
        return hash(self.get_comparable_title())

    @classmethod
    def get_comparable_str(cls, s, whitespace=True):
        """
        Depending on the source, the same resource can have slightly different
        title strings depending on the formatting, such as punctuation and
        capitalisation.
        In order to check whether two resources are the same by their title,
        their titles need to be processed in a way that removes all these
        variables.

        :param s: The input string.
        :type s: str
        :return: The input string in lowercase and with no punctuation.
        :rtype: str
        """
        if not isinstance(s, str):
            return ""

        # Turn to lowercase.
        result = s.lower()
        # Remove all punctuation.
        result = result.translate(
            str.maketrans(string.punctuation, " " * len(string.punctuation))
        )
        # Remove repeated whitespace.
        join_str = " " if whitespace else ""
        result = join_str.join(result.split())
        # Replace accented characters with non-accented equivalents.
        # Ignore all other characters that cannot be translated, e.g. Kanji.
        result = unicodedata.normalize("NFKD", result)
        result = result.encode("ascii", "ignore")
        result = result.decode("ascii")
        return result

    def get_comparable_title(self):
        return self.__class__.get_comparable_str(self.title, whitespace=False)

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
        if self.citation_count is not None:
            as_dict["citation_count"] = self.citation_count
        return as_dict

    def to_rankable_resource(self):
        """
        Initialises a RankedResource object from a Resource object.

        :return: The resultant RankedResource from the parent.
        :rtype: RankableResource
        """
        args = self.to_dict()
        # FIXME: References are currently ignored due to dict type conversion.
        if "references" in args:
            args.pop("references")
        return RankableResource(args)


class RankableResource(Resource):
    """
    An academic resource object with ranking information from the recommendation
    algorithm.
    """

    def __init__(self, args):
        super().__init__(args)
        self.author_based_ranking = -1
        self.citation_based_ranking = -1
        self.citation_count_ranking = -1
        self.keyword_based_ranking = -1

    def to_dict(self):
        as_dict = super().to_dict()
        as_dict["author_based_ranking"] = self.author_based_ranking
        as_dict["citation_based_ranking"] = self.citation_based_ranking
        as_dict["citation_count_ranking"] = self.citation_count_ranking
        as_dict["keyword_based_ranking"] = self.keyword_based_ranking
        return as_dict


if __name__ == "__main__":
    print("\nResource: Verify the correctness of the to_dict method")

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
    print(f"Resource: dict1 == dict2: {resource_dict1 == resource_dict2}")

    print("\nResource: Verify the correctness of the instantiation method")

    res_obj1 = Resource(resource_dict1)
    res_obj2 = Resource(resource_dict1)
    print(f"Resource: res_obj1 == res_obj1: {res_obj1 == res_obj1}")
    print(f"Resource: res_obj1 == res_obj2: {res_obj1 == res_obj2}")

    print("\nResource: Verify the correctness of the hashcode method")

    some_dict = {}
    some_dict[res_obj1] = 1
    some_dict[res_obj2] = 2
    # res_obj1 and res_obj2 should be the same object.
    # The dict should store only one key, with the value being 2.
    print(f"Resource: some_dict: {some_dict}")
