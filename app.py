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

    target_resources = []
    for target_json in req_data["targets"]:
        target_resource = recommend.Resource(
            title=target_json["title"],
            authors=target_json["authors"],  # TODO: Data type conversion.
            date=target_json["date"],
            abstract=target_json["abstract"],
            introduction=target_json["introduction"]
        )
        target_resources.append(target_resource)

    related_resources = recommend.recommend(target_resources)
    res_data = {"related": []}
    for related_resource in related_resources:
        res_data["related"].append({
            "title": related_resource.title,
            "authors": related_resource.authors,  # TODO: Data type conversion.
            "date": related_resource.date,
            "abstract": related_resource.abstract,
            "introduction": related_resource.introduction
        })

    return jsonify(res_data)


if __name__ == '__main__':
    app.run()
