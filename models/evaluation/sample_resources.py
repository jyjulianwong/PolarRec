"""
Objects and methods used to obtain sample resources used for evaluation.
"""
import json
import os
import sys
from models.custom_logger import log
from models.resource import Resource
from models.resource_database_adapters.arxiv_query_builder import \
    ArxivQueryBuilder
from models.resource_database_adapters.ieee_xplore_query_builder import \
    IEEEXploreQueryBuilder

ARXIV_SAMPLE_FILEPATH = "arxiv-sample-resource-data.json"
ARXIV_SAMPLE_TITLES = [
    # Human-computer interaction
    "Human or Machine: Reflections on Turing-Inspired Testing for the Everyday",
    # "Towards a Deep(er) Understanding of Interaction through Modeling, Simulation, and Optimization",
    # "Timeline Design Space for Immersive Exploration of Time-Varying Spatial 3D Data",
    # Machine learning
    "Interpretation of Time-Series Deep Models: A Survey",
    "Dynamic Causal Explanation Based Diffusion-Variational Graph Neural Network for Spatio-temporal Forecasting",
    # "Calibration Error Estimation Using Fuzzy Binning",
    # Astrophysics (Earth and Planetary Astrophysics)
    "Star-Planet Interaction at radio wavelengths in YZ Ceti: Inferring planetary magnetic field",
    "Substructures in Compact Disks of the Taurus Star-forming Region",
    # "A Bayesian Analysis of Technological Intelligence in Land and Oceans",
]

IEEE_XPLORE_SAMPLE_FILEPATH = "ieee-xplore-sample-resource-data.json"
IEEE_XPLORE_SAMPLE_TITLES = [
    # Human-computer interaction
    "Spatial approximation of volumetric images for simplified transmission and display",
    # "Recent development in stereoscopic display technology",
    # "Electoronic hologram generation using high quality color and depth information of natural scene",
    # Machine learning
    "Wavelet Basis Function Neural Networks for Sequential Learning",
    "Bayesian Bidirectional Backpropagation Learning",
    # "A Bagging Long Short-term Memory Network for Financial Transmission Rights Forecasting",
    # Astrophysics
    "Comparative Study on the Performance of Two Different Planetary Geared Permanent Magnet Planetary Gear Motors",
    "Prediction of DC magnetic fields for magnetic cleanliness on spacecraft",
    # "Planetary radio occultation technique and inversion method for YH-1 Mars mission",
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
            query_builder.set_title(Resource.get_comparable_str(title))
            result_resources = query_builder.get_resources(5)
            for resource in result_resources:
                result_comp_title = Resource.get_comparable_str(resource.title)
                target_comp_title = Resource.get_comparable_str(title)
                if result_comp_title == target_comp_title:
                    # A sample resource has been found.
                    resources.append(resource)

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
