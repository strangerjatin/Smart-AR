from flask import Flask, jsonify
import random

app = Flask(__name__)

@app.route('/detect')
def detect():
    # Replace with your real YOLO/OpenCV code
    directions = ["Obstacle Ahead", "Move Left", "Move Right", "Clear"]
    result = random.choice(directions)

    return jsonify({"message": result})

app.run(host="0.0.0.0", port=5000)
