from flask import Flask, request, render_template, send_file
from ultralytics import YOLO
import cv2
import numpy as np
import os

app = Flask(__name__)
model = YOLO('yolov8m-seg.pt')

@app.route('/')
def upload_page():
    return '''
    <h2>Overlapping Object Cropping</h2>
    <form method="POST" action="/process" enctype="multipart/form-data">
      <input type="file" name="image" accept="image/*" required>
      <input type="text" name="primary" placeholder="Primary object (e.g. face)" required>
      <input type="text" name="occluder" placeholder="Occluder (e.g. hand)">
      <button type="submit">Process</button>
    </form>
    '''

@app.route('/process', methods=['POST'])
def process():
    file = request.files['image']
    primary = request.form['primary']
    occluder = request.form.get('occluder', None)
    
    path = os.path.join('uploads', file.filename)
    file.save(path)

    image = cv2.imread(path)
    results = model(image)
    masks = results[0].masks.data.cpu().numpy()
    boxes = results[0].boxes
    names = model.names

    primary_mask, occluder_mask = None, None
    for i, c in enumerate(boxes.cls):
        label = names[int(c)]
        if label == primary:
            primary_mask = masks[i]
        elif label == occluder:
            occluder_mask = masks[i]
    
    if primary_mask is None:
        return "Primary object not found!"

    if occluder_mask is not None:
        final_mask = cv2.subtract(primary_mask, occluder_mask)
    else:
        final_mask = primary_mask

    y, x = np.where(final_mask > 0)
    cropped = image[np.min(y):np.max(y), np.min(x):np.max(x)]
    output_path = os.path.join('uploads', 'result.jpg')
    cv2.imwrite(output_path, cropped)
    return send_file(output_path, mimetype='image/jpeg')

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(host='0.0.0.0', port=8080)
