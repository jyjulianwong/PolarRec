"""
The entry point to the API.
"""
from flask import Flask, jsonify, request
from flask_cors import CORS
from models.recommend import get_related_resources
from models.resource import Resource

app = Flask(__name__)
CORS(app)


@app.route('/', methods=['GET'])
def root():
    """
    :return: The root page of the API.
    :rtype: Response
    """
    return ''


@app.route('/recommend', methods=['POST'])
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
    #             "authors": ["Vijay Badrinarayanan", "Alex Kendall", "Roberto Cipolla"],
    #             "title": "SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation",
    #             "year": 2017,
    #             "month": 1,
    #             "abstract": "We present a novel and practical...",
    #             "doi": "10.1109/TPAMI.2016.2644615",
    #             "url": "https://ieeexplore.ieee.org/document/7803544"
    #         }
    #     ]
    # }
    req_data = request.json

    target_resources = []
    for target_json in req_data["targets"]:
        target_resources.append(Resource(target_json))

    related_resources = get_related_resources(target_resources)

    res_data = {"related": []}
    for related_resource in related_resources:
        res_data["related"].append(related_resource.to_dict())

    return jsonify(res_data)


if __name__ == '__main__':
    app.run()
