import json
import base64
from PIL import Image
import io
import os

from ultralytics import YOLO


def init_context(context):
    context.logger.info("Init context...  0%")

    model_path = "yolov8n_detection_960.pt"
    if os.path.exists(model_path):
        context.logger.info(f"Loading custom model from {model_path}")
    else:
        raise f"Custom model not found, using pretrained {model_path}"
    model = YOLO(model_path, task="detect")
    context.user_data.model = model

    context.logger.info("Init context...100%")


def handler(context, event):
    context.logger.info("Run yolov8 model")
    data = event.body
    buf = io.BytesIO(base64.b64decode(data["image"]))
    threshold = float(data.get("threshold", 0.5))
    context.user_data.model.conf = threshold
    image = Image.open(buf)

    yolo_results = context.user_data.model(image, conf=threshold)
    labels = yolo_results[0].names

    results = []
    for box in yolo_results[0].boxes:
        results.append({
            "confidence": str(float(box.conf.cpu().item())),
            "label": labels.get(int(box.cls.cpu().item()), "unknown"),
            "points": list(map(int, box.xyxy.cpu().numpy()[0])),
            "type": "rectangle",
        })

    return context.Response(
        body=json.dumps(results),
        headers={},
        content_type='application/json',
        status_code=200
    )
