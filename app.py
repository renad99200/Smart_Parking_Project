from flask import Flask, render_template, send_from_directory, jsonify, request, send_file
import os
import pickle
import numpy as np
from skimage.io import imread
from skimage.transform import resize
import cv2
from werkzeug.utils import secure_filename
from util import get_parking_spots_bboxes, empty_or_not
import io
import base64

app = Flask(__name__, static_folder='static', template_folder='templates')

# تحميل الموديل عند بدء السيرفر
MODEL_PATH = os.path.join('models', 'model.pkl')
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

# إعدادات رفع الصور
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Serve static files (css, js, webfonts)
@app.route('/css/<path:filename>')
def serve_css(filename):
    return send_from_directory(os.path.join(app.root_path, 'css'), filename)

@app.route('/js/<path:filename>')
def serve_js(filename):
    return send_from_directory(os.path.join(app.root_path, 'js'), filename)

@app.route('/webfonts/<path:filename>')
def serve_webfonts(filename):
    return send_from_directory(os.path.join(app.root_path, 'webfonts'), filename)

@app.route('/normalize.css')
def serve_normalize():
    return send_from_directory(app.root_path, 'normalize.css')

# Main page
@app.route('/')
def main():
    page = request.args.get('page', 'index')
    if page == 'help':
        return render_template('help.html')
    else:
        return render_template('index.html')

# API endpoint for parking info (dynamic data)
@app.route('/api/parking-info')
def parking_info():
    data = {
        'available': 12,
        'occupied': 8,
        'route': 'A2 → B1 → C3'
    }
    return jsonify(data)

# API endpoint للتوقع باستخدام الموديل
@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = data['features']  # لازم تبعت {"features": [قيمة1, قيمة2, ...]}
    prediction = model.predict([features])
    return jsonify({'prediction': prediction.tolist()})

# ... باقي الأكواد كما هي ...

@app.route('/api/parking-detect-video', methods=['POST'])
def parking_detect_video():
    if 'video' not in request.files:
        return jsonify({'error': 'يرجى رفع فيديو الباركنج'}), 400
    video_file = request.files['video']
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(video_file.filename))
    video_file.save(video_path)

    # استخدم اسم الماسك الثابت من السيرفر
    mask_path = os.path.join(app.root_path, 'static', 'mask_1920_1080.png')
    if not os.path.exists(mask_path):
        return jsonify({'error': 'الماسك الثابت غير موجود على السيرفر'}), 400

    # قراءة الماسك
    mask = cv2.imread(mask_path, 0)
    if mask is None:
        return jsonify({'error': 'تعذر قراءة الماسك'}), 400

    # قراءة أول فريم من الفيديو
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return jsonify({'error': 'تعذر قراءة أول فريم من الفيديو'}), 400

    # استخراج أماكن الركن
    connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
    spots = get_parking_spots_bboxes(connected_components)

    # تحليل كل مكان ركن ورسم المستطيلات
    spots_status = []
    for spot in spots:
        x1, y1, w, h = spot
        spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
        status = empty_or_not(spot_crop)
        spots_status.append(status)
        color = (0, 255, 0) if status else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), color, 2)

    # رسم عدد الأماكن الفاضية
    cv2.rectangle(frame, (80, 20), (550, 80), (0, 0, 0), -1)
    cv2.putText(frame, f'Available spots: {int(np.sum(spots_status))} / {len(spots_status)}', (100, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # تحويل الصورة إلى base64
    _, img_encoded = cv2.imencode('.png', frame)
    img_base64 = base64.b64encode(img_encoded.tobytes()).decode('utf-8')

    available = int(np.sum(spots_status))
    occupied = len(spots_status) - available

    # حساب التكلفة (مثال: 10 جنيه لكل مكان مشغول)
    COST_PER_SPOT = 10
    cost = occupied * COST_PER_SPOT

    return jsonify({
        'available': available,
        'occupied': occupied,
        'total': len(spots_status),
        'status': spots_status,
        'image_base64': img_base64,
        'cost': cost
    })

if __name__ == '__main__':
    app.run(debug=True)