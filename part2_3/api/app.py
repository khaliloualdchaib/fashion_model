from flask import Flask, request, jsonify
from recommender import recommend_clothing

app = Flask(__name__)

@app.route('/recommend', methods=['GET'])
def get_recommendations():
    body_type = request.args.get('bodytype')
    if not body_type:
        return jsonify({"error": "Please provide a valid body type (hourglass, apple, rectangle)"}), 400

    recommendations = recommend_clothing(body_type)
    if 'error' in recommendations:
        return jsonify({"error": "Please provide a valid body type (hourglass, apple, rectangle)"}), 400
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8003, debug=True)

