"""
The entry point to the API.
"""
import os
import time
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
    log("KeywordRanker model successfully loaded", "application")


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

    # The start time of processing the request.
    t1 = time.time()

    # Convert the request payload as JSON-like data.
    req_data = request.json

    # Process all the target resources into Resource objects.
    target_resources = []
    if "targets" in req_data:
        for target_json in req_data["targets"]:
            target_resources.append(Resource(target_json))

    # Process all the existing related resources into Resource objects.
    existing_related_resources = []
    if "existing_related" in req_data:
        for existing_related_json in req_data["existing_related"]:
            existing_related_resources.append(Resource(existing_related_json))

    # Process all the user-specified filters into a ResourceFilter object.
    resource_filter = ResourceFilter({})
    if "filter" in req_data:
        resource_filter = ResourceFilter(req_data["filter"])

    # Retrieve the list of related resources via the recommendation algorithm.
    related_resources = get_related_resources(
        target_resources,
        existing_related_resources,
        resource_filter,
        keyword_model
    )

    # Convert each related resource into JSON-like data.
    res_data = {"related": []}
    for related_resource in related_resources:
        res_data["related"].append(related_resource.to_dict())

    # The end time of processing the request.
    t2 = time.time()

    # Add the processing time to the data returned by the API.
    res_data["proc_time"] = t2 - t1

    return jsonify(res_data)


if __name__ == "__main__":
    application.run()
