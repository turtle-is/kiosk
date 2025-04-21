from flask import Flask, Response, jsonify, request
from flask_cors import CORS
import cv2
import base64
import requests
import numpy as np
import time
import io
import logging
import qrcode
from object_detection import detect_objects, CLASS_NAMES
import gc

app = Flask(__name__)
CORS(app)

DATABASE_URL = "http://localhost:5003"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

cart = {}
prev_counts = {}

def update_cart(detections: list[dict]) -> None:
    global cart, prev_counts

    new_counts = {}
    for det in detections:
        label = det.get("class")
        if label not in CLASS_NAMES:
            continue
        category_id = CLASS_NAMES.index(label)
        new_counts[category_id] = new_counts.get(category_id, 0) + 1

    for cat, new_count in new_counts.items():
        old_count = prev_counts.get(cat, 0)
        diff = new_count - old_count
        if diff > 0:
            logging.info(f"[+] '{CLASS_NAMES[cat]}' (ID {cat}): {diff} шт.")
        elif diff < 0:
            logging.info(f"[-] '{CLASS_NAMES[cat]}' (ID {cat}): {-diff} шт.")

    for cat in prev_counts:
        if cat not in new_counts:
            logging.info(f"[x] '{CLASS_NAMES[cat]}' (ID {cat}) полностью исчезли.")

    try:
        response = requests.get(f"{DATABASE_URL}/get_products", timeout=2)
        response.raise_for_status()
        products = response.json()
    except Exception as e:
        logging.error(f"Ошибка получения продуктов: {e}")
        products = []

    new_cart = {}
    for cat, count in new_counts.items():
        product = next((p for p in products if int(p.get("category", -1)) == cat), None)
        if product:
            unit_price = float(product.get("price", 0))
            new_cart[cat] = {
                "name": product.get("name"),
                "unit_price": unit_price,
                "quantity": count,
                "total_price": round(unit_price * count, 2)
            }

    cart = new_cart
    prev_counts = new_counts.copy()

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        start_time = time.time()
        processed_frame, detected_items = detect_objects(frame)
        fps = 1.0 / (time.time() - start_time + 1e-5)

        cv2.putText(processed_frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        update_cart(detected_items)

        _, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()

        del frame, processed_frame, buffer, detected_items
        gc.collect()

        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detect', methods=['POST'])
def detect():
    data = request.json
    if not data or "image" not in data:
        return jsonify({"status": "error", "message": "No image provided"}), 400

    try:
        image_data = base64.b64decode(data["image"])
        img_array = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image")
    except Exception as e:
        logging.warning(f"Ошибка декодирования изображения: {e}")
        return jsonify({"status": "error", "message": "Invalid image data"}), 400

    processed_img, detected_items = detect_objects(img)
    update_cart(detected_items)

    _, buffer = cv2.imencode('.jpg', processed_img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    del image_data, img_array, img, processed_img, detected_items, buffer
    gc.collect()

    return jsonify({
        "status": "success",
        "image": img_base64,
        "detections": detected_items
    }), 200

@app.route('/cart', methods=['GET'])
def get_cart():
    return jsonify(cart)

@app.route('/pay', methods=['GET'])
def pay():
    total_sum = round(sum(item['total_price'] for item in cart.values()), 2)
    payment_url = f"https://payment.example.com/pay?amount={total_sum}"
    
    qr = qrcode.make(payment_url)
    buffer = io.BytesIO()
    qr.save(buffer, format="PNG")
    qr_code_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    return jsonify({"qr_code": qr_code_base64})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=False)
