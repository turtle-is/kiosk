import onnx
import onnxsim

# Загружаем ONNX-модель
model_path = "/Users/kikita/Desktop/Jeremy/university/диплом/kiosk/backend_service/models/yolov5s.onnx"
output_path = "yolov5s_simplified.onnx"

model = onnx.load(model_path)
model_simplified, check = onnxsim.simplify(model)

# Сохраняем оптимизированную модель
onnx.save(model_simplified, output_path)
print("✅ Оптимизированная модель сохранена:", output_path)
