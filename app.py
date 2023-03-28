from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.route('/', methods=['GET'])
def root():
    return ''


@app.route('/recommend', methods=['POST'])
def recommend():
    req_data = request.json
    res_data = {}
    return jsonify(res_data)


if __name__ == '__main__':
    app.run()
