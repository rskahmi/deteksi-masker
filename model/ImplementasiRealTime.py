import cv2
import numpy as np
import tensorflow as tf

# Load model U-Net yang sudah dilatih
model = tf.keras.models.load_model('unet_mask_model.h5')
IMG_SIZE = 128

# Load Haar Cascade untuk deteksi wajah
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def preprocess_face(face):
    face_resized = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
    face_normalized = face_rgb / 255.0
    return face_normalized[np.newaxis, ...]

def overlay_mask_on_face(face, mask):
    mask_resized = cv2.resize(mask, (face.shape[1], face.shape[0]))
    mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255

    red_overlay = np.zeros_like(face)
    red_overlay[:, :, 2] = mask_binary  # channel merah untuk highlight masker

    return cv2.addWeighted(face, 1, red_overlay, 0.5, 0)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_crop = frame[y:y+h, x:x+w]
        input_img = preprocess_face(face_crop)

        pred_mask = model.predict(input_img, verbose=0)[0, :, :, 0]

        # Hitung mask ratio (luas area masker)
        mask_area = np.sum(pred_mask > 0.5)
        mask_ratio = mask_area / (IMG_SIZE * IMG_SIZE)

        # Tentukan threshold klasifikasi
        threshold = 0.1
        if mask_ratio > threshold:
            label = "Mask Detected"
            color = (0, 255, 0)  # hijau
        else:
            label = "No Mask"
            color = (0, 0, 255)  # merah

        mask_overlay = overlay_mask_on_face(face_crop, pred_mask)

        # Tempelkan hasil overlay dan label ke frame asli
        frame[y:y+h, x:x+w] = mask_overlay
        cv2.putText(frame, f"{label} ({mask_ratio:.2f})", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    cv2.imshow("Mask Detection (U-Net + Classification)", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC untuk keluar
        break

cap.release()
cv2.destroyAllWindows()
