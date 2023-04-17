"""
The entry point to the API.
"""
from flask import Flask, jsonify, request
from flask_cors import CORS
from models.keywords import keywords
from models.recommend import get_related_resources
from models.resource import Resource

application = Flask(__name__)
CORS(application)
with application.app_context():
    keywords_model = keywords.get_model()
    print("keywords_model successfully loaded.")


@application.route("/", methods=["GET"])
def root():
    """
    :return: The root page of the API.
    :rtype: Response
    """
    return ""


@application.route("/recommend", methods=["POST"])
def recommend():
    """
    Takes POST requests of target academic resources and related metadata, and
    returns recommendations.
    Requires ``"content-type": "application/json"`` to be set in the request
    header.
    See comment below for an example of a request body.

    :return: A list of recommended academic resources, in the form of a JSON.
    :rtype: Response
    """
    # POST request body example:
    # {
    #     "targets": [
    #         {
    #             "authors": ["Y. Lecun", "L. Bottou", "Y. Bengio", "P. Haffner"],
    #             "title": "Gradient-based learning applied to document recognition",
    #             "year": 1998,
    #             "month": 1,
    #             "abstract": "Multilayer neural networks trained with...",
    #             "doi": "10.1109/5.726791",
    #             "url": "https://ieeexplore.ieee.org/document/726791"
    #         }
    #     ]
    # }
    global keywords_model

    req_data = request.json

    target_resources = []
    for target_json in req_data["targets"]:
        target_resources.append(Resource(target_json))

    related_resources = get_related_resources(target_resources, keywords_model)

    res_data = {"related": []}
    for related_resource in related_resources:
        res_data["related"].append(related_resource.to_dict())

    return jsonify(res_data)


if __name__ == "__main__":
    # TODO: Remove this if redundant for deployment.
    with application.app_context():
        keywords_model = keywords.get_model()
        print("keywords_model successfully loaded from __main__.")
    application.run()