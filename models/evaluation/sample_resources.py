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

# The data filepath for arXiv database samples.
# Data for sample resources are stored here.
ARXIV_SAMPLE_FILEPATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "arxiv-sample-resource-data.json"
)
# The titles of the sample resources used from the arXiv database.
ARXIV_SAMPLE_TITLES = [
    # Human-computer interaction
    "Human or Machine: Reflections on Turing-Inspired Testing for the Everyday",
    "Towards a Deep(er) Understanding of Interaction through Modeling, Simulation, and Optimization",
    "Timeline Design Space for Immersive Exploration of Time-Varying Spatial 3D Data",
    # Machine learning
    "Interpretation of Time-Series Deep Models: A Survey",
    "Dynamic Causal Explanation Based Diffusion-Variational Graph Neural Network for Spatio-temporal Forecasting",
    "Calibration Error Estimation Using Fuzzy Binning",
    # Astrophysics (Earth and Planetary Astrophysics)
    "Star-Planet Interaction at radio wavelengths in YZ Ceti: Inferring planetary magnetic field",
    "Substructures in Compact Disks of the Taurus Star-forming Region",
    "A Bayesian Analysis of Technological Intelligence in Land and Oceans",
]

# The data filepath for IEEE Xplore database samples.
# Data for sample resources are stored here.
IEEE_XPLORE_SAMPLE_FILEPATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "ieee-xplore-sample-resource-data.json"
)
# The titles of the sample resources used from the IEEE Xplore database.
IEEE_XPLORE_SAMPLE_TITLES = [
    # Human-computer interaction
    "Spatial approximation of volumetric images for simplified transmission and display",
    "Recent development in stereoscopic display technology",
    "Electoronic hologram generation using high quality color and depth information of natural scene",
    # Machine learning
    "Wavelet Basis Function Neural Networks for Sequential Learning",
    "Bayesian Bidirectional Backpropagation Learning",
    "A Bagging Long Short-term Memory Network for Financial Transmission Rights Forecasting",
    # Astrophysics
    "Comparative Study on the Performance of Two Different Planetary Geared Permanent Magnet Planetary Gear Motors",
    "Prediction of DC magnetic fields for magnetic cleanliness on spacecraft",
    "Planetary radio occultation technique and inversion method for YH-1 Mars mission",
]

# The mapping between data filepaths and titles of the resources they store.
FILEPATH_TO_TITLES_DICT = {
    ARXIV_SAMPLE_FILEPATH: ARXIV_SAMPLE_TITLES,
    IEEE_XPLORE_SAMPLE_FILEPATH: IEEE_XPLORE_SAMPLE_TITLES
}

# The mapping between data filepaths and the adapters used to extract the data.
FILEPATH_TO_ADAPTER_DICT = {
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

    for filepath, titles in FILEPATH_TO_TITLES_DICT.items():
        resources: list[Resource] = []
        for title in titles:
            # Get the appropriate adapter object that should be used.
            query_builder = FILEPATH_TO_ADAPTER_DICT[filepath]
            # Use the title of the sample resource as the search query.
            query_builder.set_title(Resource.get_comparable_str(title))
            # Execute the search query.
            result_resources = query_builder.get_resources(5)
            for resource in result_resources:
                # Transform the results' titles into a comparable string.
                result_comp_title = Resource.get_comparable_str(resource.title)
                target_comp_title = Resource.get_comparable_str(title)
                if result_comp_title == target_comp_title:
                    # The correct sample resource has been found.
                    resources.append(resource)

        # Transform the Resource object into its JSON-like data representation.
        data_list = [resource.to_dict() for resource in resources]
        # Transform the JSON data into a singular JSON string.
        json_list = json.dumps(data_list, indent=4)
        # Write the JSON data to a persistent local file.
        with open(filepath, "w+", encoding="utf-8") as file_object:
            file_object.write(json_list)


def load_resources_from_json():
    """
    Loads the sample resources from a local JSON file, if present.

    :return: A mapping between data filepaths and the resources they store.
    :rtype: dict[str, list[Resource]]
    """
    if os.environ.get("FLASK_ENV", "development") != "development":
        # This should only be run in a development environment.
        sys.exit(-1)

    # A mapping between data filepaths and the resources they store.
    filepath_to_resource_dict: dict[str, list[Resource]] = {}
    for filepath in FILEPATH_TO_TITLES_DICT:
        # The list of resources this data file stores.
        ress: list[Resource] = []

        # Check if the persistent data file exists on the local machine.
        if not os.path.isfile(filepath):
            log(
                f"{filepath} is not a valid filepath",
                "sample_resources",
                "error"
            )
            filepath_to_resource_dict[filepath] = []
            continue

        # Read JSON data from the persistent data file.
        with open(filepath, "r", encoding="utf-8") as file_object:
            json_list = json.load(file_object)
            for json_data in json_list:
                # Instantiate Resource objects from the JSON data.
                ress.append(Resource(json_data))
            filepath_to_resource_dict[filepath] = ress

    return filepath_to_resource_dict


if __name__ == "__main__":
    # Uncomment the next line to recollect all the sample resources.
    # save_resources_as_json()
    filepath_to_resource_dict = load_resources_from_json()
    for filepath, resources in filepath_to_resource_dict.items():
        print(f"Absolute file path: {filepath}")
        print(f"\tNumber of sample resources: {len(resources)}")
