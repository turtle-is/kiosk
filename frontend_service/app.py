from flask import Flask, render_template, request, jsonify
import requests
import base64

app = Flask(__name__)
BACKEND_URL = "http://backend_service:5002"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/purchase.html')
def purchase():
    return render_template('purchase.html')

@app.route('/detect', methods=['POST'])
def detect():
    file = request.files["frame"]
    image_data = file.read()
    image_base64 = base64.b64encode(image_data).decode('utf-8')
    
    response = requests.post(f"{BACKEND_URL}/detect", json={"image": image_base64})

    if response.status_code == 200:
        return jsonify(response.json())
    else:
        return jsonify({"error": "Ошибка детекции"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
