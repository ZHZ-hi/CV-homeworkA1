import os
import json
import base64
import numpy as np
import cv2
from flask import Flask, render_template, request, jsonify
import io

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def nearest_neighbor_cv(img, new_w, new_h):
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

def bilinear_cv(img, new_w, new_h):
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

def rotate_image_cv(img, angle, method):
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    angle_rad = np.radians(angle)
    cos_a = np.abs(np.cos(angle_rad))
    sin_a = np.abs(np.sin(angle_rad))
    new_w = int(w * cos_a + h * sin_a)
    new_h = int(w * sin_a + h * cos_a)
    
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    rot_mat[0, 2] += (new_w - w) / 2
    rot_mat[1, 2] += (new_h - h) / 2
    
    if method == 'nearest':
        return cv2.warpAffine(img, rot_mat, (new_w, new_h), flags=cv2.INTER_NEAREST, borderValue=(0, 0, 0))
    else:
        return cv2.warpAffine(img, rot_mat, (new_w, new_h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))

def image_to_base64(img):
    if img is None or img.size == 0:
        return ""
    img_encoded = cv2.imencode('.png', img)
    if img_encoded:
        return base64.b64encode(img_encoded[1]).decode('utf-8')
    return ""

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/process', methods=['POST'])
def process():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        img_file = request.files['image']
        img_bytes = img_file.read()
        
        img_array = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'error': 'Failed to decode image'}), 400
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        operation = request.form.get('operation')
        
        if operation == 'colorSpace':
            color_space = request.form.get('colorSpace', 'RGB')
            results = []
            
            if color_space == 'RGB':
                b, g, r = cv2.split(img)
                results = [
                    {'name': 'R通道', 'image': image_to_base64(r)},
                    {'name': 'G通道', 'image': image_to_base64(g)},
                    {'name': 'B通道', 'image': image_to_base64(b)},
                ]
            elif color_space == 'HSV':
                hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
                h, s, v = cv2.split(hsv)
                results = [
                    {'name': 'H通道', 'image': image_to_base64(h)},
                    {'name': 'S通道', 'image': image_to_base64(s)},
                    {'name': 'V通道', 'image': image_to_base64(v)},
                ]
            elif color_space == 'HLS':
                hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
                h, l, s = cv2.split(hls)
                results = [
                    {'name': 'H通道', 'image': image_to_base64(h)},
                    {'name': 'L通道', 'image': image_to_base64(l)},
                    {'name': 'S通道', 'image': image_to_base64(s)},
                ]
            elif color_space == 'YUV':
                yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
                y, u, v = cv2.split(yuv)
                results = [
                    {'name': 'Y通道', 'image': image_to_base64(y)},
                    {'name': 'U通道', 'image': image_to_base64(u)},
                    {'name': 'V通道', 'image': image_to_base64(v)},
                ]
            elif color_space == 'YCrCb':
                ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
                y, cr, cb = cv2.split(ycrcb)
                results = [
                    {'name': 'Y通道', 'image': image_to_base64(y)},
                    {'name': 'Cr通道', 'image': image_to_base64(cr)},
                    {'name': 'Cb通道', 'image': image_to_base64(cb)},
                ]
            
            return jsonify({'results': results})
        
        elif operation == 'interpolation':
            method = request.form.get('method', 'bilinear')
            op = request.form.get('op', 'scale')
            params = json.loads(request.form.get('params', '{}'))
            
            if op == 'scale':
                scale = params.get('scale', 2)
                new_h = int(img.shape[0] * scale)
                new_w = int(img.shape[1] * scale)
                result = nearest_neighbor_cv(img, new_w, new_h) if method == 'nearest' else bilinear_cv(img, new_w, new_h)
            elif op == 'resize':
                new_w = params.get('width', 400)
                new_h = params.get('height', 300)
                result = nearest_neighbor_cv(img, new_w, new_h) if method == 'nearest' else bilinear_cv(img, new_w, new_h)
            elif op == 'rotate':
                angle = params.get('angle', 45)
                result = rotate_image_cv(img, angle, method)
            else:
                result = img
            
            result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
            return jsonify({'result': image_to_base64(result_bgr)})
        
        elif operation == 'channelAdjust':
            color_space = request.form.get('colorSpace', 'HSV')
            params = json.loads(request.form.get('params', '{}'))
            result = img.copy()
            
            if color_space == 'HSV':
                hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
                h, s, v = cv2.split(hsv)
                
                if 'h' in params:
                    h = np.mod(h + params['h'], 180)
                if 's' in params:
                    s = np.clip(s * params['s'] / 100, 0, 255)
                if 'v' in params:
                    v = np.clip(v * params['v'] / 100, 0, 255)
                
                hsv = np.stack([h, s, v], axis=2).astype(np.uint8)
                result = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
                
            elif color_space == 'RGB':
                b, g, r = cv2.split(img.astype(np.float32))
                
                if 'r' in params:
                    r = np.clip(r * params['r'] / 100, 0, 255)
                if 'g' in params:
                    g = np.clip(g * params['g'] / 100, 0, 255)
                if 'b' in params:
                    b = np.clip(b * params['b'] / 100, 0, 255)
                
                result = np.stack([b, g, r], axis=2).astype(np.uint8)
                
            elif color_space == 'YUV':
                yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV).astype(np.float32)
                y, u, v = cv2.split(yuv)
                
                if 'y' in params:
                    y = np.clip(y * params['y'] / 100, 0, 255)
                if 'u' in params:
                    u = np.clip((u - 128) * params['u'] / 100 + 128, 0, 255)
                if 'v' in params:
                    v = np.clip((v - 128) * params['v'] / 100 + 128, 0, 255)
                
                yuv = np.stack([y, u, v], axis=2).astype(np.uint8)
                result = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
                
            elif color_space == 'YCrCb':
                ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb).astype(np.float32)
                y, cr, cb = cv2.split(ycrcb)
                
                if 'y' in params:
                    y = np.clip(y * params['y'] / 100, 0, 255)
                if 'cr' in params:
                    cr = np.clip((cr - 128) * params['cr'] / 100 + 128, 0, 255)
                if 'cb' in params:
                    cb = np.clip((cb - 128) * params['cb'] / 100 + 128, 0, 255)
                
                ycrcb = np.stack([y, cr, cb], axis=2).astype(np.uint8)
                result = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
            
            result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
            return jsonify({'result': image_to_base64(result_bgr)})
        
        return jsonify({'error': 'Invalid operation'}), 400
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')