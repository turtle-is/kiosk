import threading
import queue
import time
from object_detection import detect_objects

class AsyncDetector:
    def __init__(self):
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.running = True
        self.worker = threading.Thread(target=self._process, daemon=True)
        self.worker.start()

    def _process(self):
        while self.running:
            try:
                frame_id, frame = self.input_queue.get(timeout=1)
                processed, detections = detect_objects(frame)
                self.output_queue.put((frame_id, processed, detections))
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[AsyncDetector] Error: {e}")

    def submit(self, frame_id, frame):
        """Добавляет кадр в очередь для обработки."""
        self.input_queue.put((frame_id, frame))

    def get_result(self):
        """Пытается получить готовый результат."""
        try:
            return self.output_queue.get_nowait()
        except queue.Empty:
            return None

    def stop(self):
        self.running = False
        self.worker.join()
