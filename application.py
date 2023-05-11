"""
The entry point to the API.
"""
import os
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from models.custom_logger import log
from models.recommend import get_related_resources
from models.resource import Resource
from models.resource_filter import ResourceFilter
from models.resource_rankers.keyword_ranker import KeywordRanker

application = Flask(__name__)
CORS(application)
with application.app_context():
    keyword_model = KeywordRanker.get_model()
    log("keyword_model successfully loaded", "application")


@application.route("/", methods=["GET"])
def root():
    """
    :return: The root page of the API.
    :rtype: Response
    """
    return ""


@application.route("/favicon.ico", methods=["GET"])
def favicon():
    """
    :return: The application favicon.
    :rtype: Response
    """
    return send_from_directory(
        os.path.join(application.root_path, "static"),
        "favicon.ico",
        mimetype="image/vnd.microsoft.icon"
    )


@application.route("/recommend", methods=["POST"])
def recommend():
    """
    Takes POST requests of target academic resources and related metadata, and
    returns recommendations.
    Requires ``"content-type": "application/json"`` to be set in the request
    header.
    See README.md for an example of a request body.

    :return: A list of recommended academic resources, in the form of a JSON.
    :rtype: Response
    """
    global keyword_model

    req_data = request.json

    target_resources = []
    if "targets" in req_data:
        for target_json in req_data["targets"]:
            target_resources.append(Resource(target_json))

    existing_related_resources = []
    if "existing_related" in req_data:
        for existing_related_json in req_data["existing_related"]:
            existing_related_resources.append(Resource(existing_related_json))

    resource_filter = ResourceFilter({})
    if "filter" in req_data:
        resource_filter = ResourceFilter(req_data["filter"])

    related_resources = get_related_resources(
        target_resources,
        existing_related_resources,
        resource_filter,
        keyword_model
    )

    res_data = {"related": []}
    for related_resource in related_resources:
        res_data["related"].append(related_resource.to_dict())

    return jsonify(res_data)


if __name__ == "__main__":
    application.run()
