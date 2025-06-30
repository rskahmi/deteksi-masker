# 🛡️ Face Mask Detection using Computer Vision
This project is an implementation of a face mask detection system using a machine learning model (CNN/U-Net) built with TensorFlow/Keras. The application can be run through a simple web interface that uses a webcam to perform real-time detection.

# 🚀 Features
📸 Real-time face mask detection from webcam feed. <br/>
🖥️ Modern and responsive web interface. <br/>
🤖 Deep learning model implemented in .h5 format (CNN/U-Net). <br/>
💻 Easy to run locally. <br/>

# 🧠 Technologies Used
-Backend: Flask (Python) <br/>
-Frontend: HTML, CSS, JavaScript <br/>
-Machine Learning: TensorFlow / Keras <br/>
-Computer Vision: OpenCV <br/>
-Webcam API: getUserMedia <br/>

# 🗂️ Dataset
The dataset used to train the model is available at:
https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset/data

## 📁 Project Structure
```
deteksi-masker
├── app.py                    # Main Flask application
├── model_deteksi_masker.h5   # Trained CNN model
├── templates/                # HTML templates
│   ├── index.html
├── static/
└── README.md                 # Project README
```
## ⚙️ Setup & Run
1. Clone this repo:
```bash
git clone https://github.com/rskahmi/deteksi-masker.git
cd deteksi-masker
```
2. Run train_model.py - you will get model with format .h5
```bash
python train_model.py
```
3. Run the app:
```bash
python app.py
```

4. Open your browser and go to: `http://localhost:5000`

---

Made with 💧 by Risky Ahmi
