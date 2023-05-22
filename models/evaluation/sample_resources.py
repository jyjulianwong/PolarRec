"""
Objects and methods used to obtain sample resources used for evaluation.
"""
import json
import os
import sys
from models.custom_logger import log
from models.citation_database_adapters.s2ag_adapter import S2agAdapter
from models.resource import Resource
from models.resource_database_adapters.arxiv_adapter import ArxivQueryBuilder
from models.resource_database_adapters.ieee_xplore_adapter import \
    IEEEXploreQueryBuilder

ARXIV_SAMPLE_FILEPATH = "arxiv-sample-resource-data.json"
ARXIV_SAMPLE_TITLES = [
    # Human-computer interaction
    "The Effectiveness of Applying Different Strategies on Recognition and Recall Textual Password",
    "The social media use of adult New Zealanders: Evidence from an online survey",
    "Multimodal Earable Sensing for Human Energy Expenditure Estimation",
    # Machine learning
    "Bayesian Over-the-Air FedAvg via Channel Driven Stochastic Gradient Langevin Dynamics",
    "Calibration Error Estimation Using Fuzzy Binning",
    "Federated Learning Operations Made Simple with Flame",
    # Astrophysics (Earth and Planetary Astrophysics)
    "Physical properties of the slow-rotating near-Earth asteroid (2059) Baboquivari from one apparition",
    "(433) Eros and (25143) Itokawa surface properties from reflectance spectra",
    "Substructures in Compact Disks of the Taurus Star-forming Region",
]

IEEE_XPLORE_SAMPLE_FILEPATH = "ieee-xplore-sample-resource-data.json"
IEEE_XPLORE_SAMPLE_TITLES = [
    # Human-computer interaction
    "Human-Computer Interaction for BCI Games: Usability and User Experience",
    "Various levels of human stress & their impact on human computer interaction",
    "Understanding users! Perception of privacy in human-robot interaction",
    # Machine learning
    "A new heuristic of the decision tree induction",
    "A fuzzy classification method based on support vector machine",
    "Survey on lie group machine learning",
    # Astrophysics
    "A walk on the warped side: Astrophysics with gravitational waves",
    "Computational Astrophysics",
    "Programming an astrophysics application in an object-oriented parallel language",
]

RES_SET_FILEPATH_DICT = {
    ARXIV_SAMPLE_FILEPATH: ARXIV_SAMPLE_TITLES,
    IEEE_XPLORE_SAMPLE_FILEPATH: IEEE_XPLORE_SAMPLE_TITLES
}

RES_SET_ADAPTER_DICT = {
    ARXIV_SAMPLE_FILEPATH: ArxivQueryBuilder(),
    IEEE_XPLORE_SAMPLE_FILEPATH: IEEEXploreQueryBuilder()
}


def save_resources_as_json():
    """
    Collects the sample resources from the databases once and saves the
    resources into a JSON file locally.
    """
    if os.environ.get("FLASK_ENV", "development") != "development":
        # This should only be run in a development environment.
        sys.exit(-1)

    for filepath, titles in RES_SET_FILEPATH_DICT.items():
        resources: list[Resource] = []
        for title in titles:
            query_builder = RES_SET_ADAPTER_DICT[filepath]
            query_builder.set_title(title)
            resource = query_builder.get_resources(1)[0]
            resources.append(resource)

        reference_dict = S2agAdapter.get_references(resources)
        for resource, references in reference_dict.items():
            if len(references) > 0:
                resource.references = references

        data_list = [resource.to_dict() for resource in resources]
        json_list = json.dumps(data_list, indent=4)
        with open(filepath, "w+", encoding="utf-8") as file_object:
            file_object.write(json_list)


def load_resources_from_json():
    """
    Loads the sample resources from a local JSON file, if present.

    :return: The list of saved resources.
    :rtype: dict[str, list[Resource]]
    """
    if os.environ.get("FLASK_ENV", "development") != "development":
        # This should only be run in a development environment.
        sys.exit(-1)

    res_set_dict: dict[str, list[Resource]] = {}
    for filepath in RES_SET_FILEPATH_DICT:
        ress: list[Resource] = []

        if not os.path.isfile(filepath):
            log(
                f"{filepath} is not a valid filepath",
                "sample_resources",
                "error"
            )
            res_set_dict[filepath] = []
            continue

        with open(filepath, "r", encoding="utf-8") as file_object:
            json_list = json.load(file_object)
            for json_data in json_list:
                ress.append(Resource(json_data))
            res_set_dict[filepath] = ress

    return res_set_dict


if __name__ == "__main__":
    # Uncomment the next line to recollect all the sample resources.
    # save_resources_as_json()
    res_set_dict = load_resources_from_json()
    for filepath, resources in res_set_dict.items():
        print(f"Relative file path: {filepath}")
        print(f"\tNumber of sample resources: {len(resources)}")
