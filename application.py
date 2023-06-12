"""
The entry point to the web API application.
"""
import os
import time
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from models.custom_logger import log
from models.recommend import get_recommended_resources
from models.resource import Resource
from models.resource_filter import ResourceFilter
from models.resource_rankers.keyword_ranker import KeywordRanker

# Start-up routine of the application.
application = Flask(__name__)
CORS(application)
with application.app_context():
    # Pre-load all necessary keyword-related models during start-up.
    keyword_model = KeywordRanker.get_model()
    log("KeywordRanker model successfully loaded", "application")


@application.route("/", methods=["GET"])
def root():
    """
    :return: The (empty) root page of the API.
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
    returns recommendations as JSON data.
    Requires ``"content-type": "application/json"`` to be set in the request
    header.
    See README.md for examples of an API request body and response body.

    :return: Lists of recommended academic resources, in the form of JSON data.
    :rtype: Response
    """
    # Assume the pre-loaded KeywordRanker model is available within app context.
    global keyword_model

    # Record the start time of processing the request.
    t1 = time.time()

    # Convert the request payload as JSON-like data.
    req_data = request.json

    # Process all the target resources into Resource objects.
    target_resources = []
    if "target_resources" in req_data:
        for target_json in req_data["target_resources"]:
            target_resources.append(Resource(target_json))

    # Process all the existing resources into Resource objects.
    existing_resources = []
    if "existing_resources" in req_data:
        for existing_json in req_data["existing_resources"]:
            existing_resources.append(Resource(existing_json))

    # Process all the user-specified filters into a ResourceFilter object.
    resource_filter = ResourceFilter({})
    if "filter" in req_data:
        resource_filter = ResourceFilter(req_data["filter"])

    # Process all the user-specified resource databases to search through.
    resource_database_ids = []
    if "resource_databases" in req_data:
        resource_database_ids = req_data["resource_databases"]

    # Retrieve the lists of ranked resources via the recommendation algorithm.
    reco_existing_ress, reco_database_ress = get_recommended_resources(
        target_resources=target_resources,
        existing_resources=existing_resources,
        resource_filter=resource_filter,
        resource_database_ids=resource_database_ids,
        keyword_model=keyword_model
    )

    # Convert each ranked resource into JSON-like data.
    res_data = {
        "ranked_existing_resources": [],
        "ranked_database_resources": []
    }
    for ranked_resource in reco_existing_ress:
        res_data["ranked_existing_resources"].append(ranked_resource.to_dict())
    for ranked_resource in reco_database_ress:
        res_data["ranked_database_resources"].append(ranked_resource.to_dict())

    # Record the end time of processing the request.
    t2 = time.time()

    # Add the processing time to the data returned by the API.
    res_data["processing_time"] = t2 - t1

    # Return the data in JSON format.
    return jsonify(res_data)


if __name__ == "__main__":
    application.run()
