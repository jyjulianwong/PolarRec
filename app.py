"""
The entry point to the API.
"""
from flask import Flask, jsonify, request
from flask_cors import CORS
from models import recommend

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

    POST request body parameters:
        targets: [
            {
                title: "A",
                authors: ["A", "B", "C"],
                date: "A",
                abstract: "A",
                introduction: "A"
            },
            ...
        ]

    :return: A list of recommended academic resources, in the form of a JSON.
    :rtype: Response
    """
    req_data = request.json

    resources = []
    for target_json in req_data["targets"]:
        target_resource = recommend.TargetResource(
            title=target_json["title"],
            authors=target_json["authors"],
            date=target_json["date"],
            abstract=target_json["abstract"],
            introduction=target_json["introduction"],
        )
        resources.append(target_resource)
    recommend.recommend(resources)  # TODO: Store output.

    res_data = {}
    return jsonify(res_data)


if __name__ == '__main__':
    app.run()
