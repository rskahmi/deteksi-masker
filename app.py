import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, Response

app = Flask(__name__)

# Load U-Net model
model = tf.keras.models.load_model('unet_mask_model.h5')

IMG_SIZE = 128

def preprocess_frame(frame):
    frame_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    img_array = frame_rgb / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def apply_mask(frame, mask):
    mask = (mask > 0.5).astype(np.uint8) * 255
    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
    mask_colored = cv2.applyColorMap(mask, cv2.COLORMAP_SUMMER)
    overlay = cv2.addWeighted(frame, 1, mask_colored, 0.5, 0)
    return overlay

def generate_frames():
    cap = cv2.VideoCapture(0)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_crop = frame[y:y+h, x:x+w]

            # Preprocess
            face_resized = cv2.resize(face_crop, (IMG_SIZE, IMG_SIZE))
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            input_img = np.expand_dims(face_rgb / 255.0, axis=0)

            # Predict
            pred_mask = model.predict(input_img, verbose=0)[0, :, :, 0]

            # Hitung luas mask
            mask_area = np.sum(pred_mask > 0.5)
            mask_ratio = mask_area / (IMG_SIZE * IMG_SIZE)

            threshold = 0.1
            if mask_ratio > threshold:
                label = "Mask Detected"
                color = (0, 255, 0)
            else:
                label = "No Mask"
                color = (0, 0, 255)

            # Overlay mask
            mask_resized = cv2.resize(pred_mask, (w, h))
            mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255
            red_overlay = np.zeros_like(face_crop)
            red_overlay[:, :, 2] = mask_binary

            blended = cv2.addWeighted(face_crop, 1, red_overlay, 0.5, 0)
            frame[y:y+h, x:x+w] = blended

            # Tambah label dan kotak
            cv2.putText(frame, f"{label} ({mask_ratio:.2f})", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        # Encode dan stream
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
