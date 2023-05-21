"""
Definition of the SampleResourceGetter class.
"""
from models.resource import Resource
from models.resource_database_adapters.ieee_xplore_adapter import \
    IEEEXploreQueryBuilder


class SampleResourceGetter:
    """
    A getter object used to collect sample resources to use for performance and
    quality evaluation of the recommendation algorithm.
    """
    SAMPLE_RESOURCE_TITLES = [
        # Machine learning
        "SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation",
        "DNN-Based Indoor Localization Under Limited Dataset Using GANs and Semi-Supervised Learning",
        "Credit Card Fraud Detection Using State-of-the-Art Machine Learning and Deep Learning Algorithms",
        # Natural language processing
        "Librispeech: An ASR corpus based on public domain audio books",
        # Metaverse
        "A Metaverse: Taxonomy, Components, Applications, and Open Challenges",
        # Telecommunication
        "Integrated Sensing and Communications: Toward Dual-Functional Wireless Networks for 6G and Beyond",
        # Renewable energy
        "On the History and Future of 100% Renewable Energy Systems Research",
    ]

    @classmethod
    def get_resources(cls):
        """
        :return: The list of sample resources as Resource objects.
        :rtype: list[Resource]
        """
        resources = []
        for title in cls.SAMPLE_RESOURCE_TITLES:
            query_builder = IEEEXploreQueryBuilder()
            query_builder.set_title(title)
            resources.append(query_builder.get_resources(1)[0])
        return resources


if __name__ == "__main__":
    print("\nsample_resource_getter: Validate the collected resources")
    resources = SampleResourceGetter.get_resources()
    for i, resource in enumerate(resources):
        print(f"[{i}]:\t{resource.title}")
        print(f"\t\t{resource.authors}")
        print(f"\t\t{resource.conference_name}")
        print(f"\t\t{resource.conference_location}")
        print(f"\t\t{resource.year}")
        print(f"\t\t{resource.month}")
        print(f"\t\t{resource.abstract}")
        print(f"\t\t{resource.doi}")
        print(f"\t\t{resource.url}")
        print(f"\t\t{resource.references}")
        print(f"\t\t{resource.predef_keywords}")
