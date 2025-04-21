import os
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
import logging
import time


# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Путь к модели (можно задать через переменную окружения)
MODEL_PATH = os.getenv("MODEL_PATH", "models/best.onnx")
CLASS_NAMES = ["Lays", "Water", "Twix", "Bounty", "Snickers", "Cola"]

CONFIDENCE_THRESHOLD = 0.7
IOU_THRESHOLD = 0.5
INPUT_SIZE = 640

# ONNX Runtime с оптимизациями
options = ort.SessionOptions()
options.inter_op_num_threads = 4
try:
    session = ort.InferenceSession(MODEL_PATH, sess_options=options, providers=['CoreMLExecutionProvider'])
    logging.info(f"Модель успешно загружена: {MODEL_PATH}")
except Exception as e:
    logging.error(f"Ошибка загрузки модели: {e}")
    raise

def preprocess_image(image):
    """Подготовка изображения: масштабирование, нормализация, форматирование."""
    image = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # RGB если нужно
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
    logging.debug("Изображение успешно подготовлено для инференса.")
    return np.expand_dims(image, axis=0)


def postprocess(output, orig_shape):
    """Оптимизированная постобработка результатов инференса."""
    output = output[0]  # [1, num, 85] -> [num, 85]
    h_orig, w_orig = orig_shape[:2]

    # Разделение данных
    boxes = output[:, :4]
    scores = output[:, 4:]
    class_ids = np.argmax(scores, axis=1)
    confidences = np.max(scores, axis=1)

    # Фильтрация по порогу
    keep_idxs = np.where(confidences > CONFIDENCE_THRESHOLD)[0]
    boxes = boxes[keep_idxs]
    confidences = confidences[keep_idxs]
    class_ids = class_ids[keep_idxs]

    # Преобразование координат
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = ((cx - w / 2) * w_orig).astype(np.int32)
    y1 = ((cy - h / 2) * h_orig).astype(np.int32)
    x2 = ((cx + w / 2) * w_orig).astype(np.int32)
    y2 = ((cy + h / 2) * h_orig).astype(np.int32)

    # Подготовка боксов для NMS: [x, y, w, h]
    nms_boxes = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
    confidences_list = confidences.tolist()

    # Non-Maximum Suppression
    indices = cv2.dnn.NMSBoxes(nms_boxes, confidences_list, CONFIDENCE_THRESHOLD, IOU_THRESHOLD)

    # Формирование финальных детекций
    detections = []
    for i in indices:
        i = i[0] if isinstance(i, (list, np.ndarray)) else i
        detection = {
            "class": CLASS_NAMES[class_ids[i]],
            "confidence": round(confidences[i], 2),
            "box": [x1[i], y1[i], x2[i], y2[i]]
        }
        detections.append(detection)
        logging.debug(f"Детекция: {detection['class']} с уверенностью {detection['confidence']}")

    logging.info(f"Постобработка завершена, количество детекций: {len(detections)}")
    return detections

def draw_boxes(image, detections):
    """Отрисовка прямоугольников и названий классов на изображении."""
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        label = det["class"]
        confidence = det["confidence"]
        color = (0, 255, 0)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, f"{label} {confidence:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return image

def detect_objects(image):
    """Основная функция: возвращает изображение и список детекций."""
    try:
        start_total = time.time()
        input_tensor = preprocess_image(image)
        pre_time = time.time() - start_total

        start_infer = time.time()
        outputs = session.run(None, {session.get_inputs()[0].name: input_tensor})
        infer_time = time.time() - start_infer

        start_post = time.time()
        detections = postprocess(outputs, image.shape)
        image_with_boxes = draw_boxes(image.copy(), detections)
        post_time = time.time() - start_post

        logging.info(f"Time [pre: {pre_time:.3f}s, infer: {infer_time:.3f}s, post: {post_time:.3f}s] "
                     f"Detections: {len(detections)}")

        if len(detections) == 0:
            logging.warning("Не было найдено объектов на изображении.")

        return image_with_boxes, detections

    except Exception as e:
        logging.error(f"Ошибка при детекции: {e}")
        return image, []
