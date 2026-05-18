# HandSpeak AI 🤟 | Real-Time Sign Language Translator

An advanced, end-to-end Machine Learning and Computer Vision pipeline that translates hand gestures into spoken words in real-time. This project features both a local OpenCV interface and a modern, high-performance Streamlit Web Application utilizing WebRTC for live video streaming.

## 🚀 Key Features
- **Advanced Feature Engineering:** Instead of raw coordinates, the pipeline extracts relative spatial landmarks and calculates 13 critical pairwise Euclidean distances to achieve maximum classification accuracy.
- **Robust Machine Learning:** Powered by a tuned Random Forest Classifier with integrated `StandardScaler` transformations, achieving high cross-validation stability.
- **Streamlit WebRTC Dashboard:** A sleek, fully-styled web UI (Cyberpunk theme) designed for modern browsers with real-time prediction tracking and confidence scores.
- **Thread-Safe Audio Engine:** Multithreaded Text-to-Speech (TTS) utilizing explicit locking mechanisms to prevent audio overlapping and frame drops.
- **Integrated Data & Training Pipeline:** Features a specialized HUD script for manual or automated gesture sample collection and a robust training pipeline with 5-fold cross-validation.

## 📊 Supported Gestures (8 Classes)
The model accurately recognizes and speaks out the following gestures in real-time when the confidence score exceeds 72%:
- `Hello` 👋
- `Peace` ✌️
- `Yes` ✊
- `No` 🗣️
- `Good` 👍
- `Bad` 👎
- `Water` 💧
- `i love you` ❤️

## 📁 Project Structure
├── data/
│   └── hand_data.csv           # Collected 63-landmark raw dataset
├── models/
│   ├── hand_landmarker.task    # MediaPipe base tracking model
│   └── model.pkl               # Serialized dict bundle (Model + Scaler)
├── .gitignore                  # Prevents uploading heavy venv/caches
├── app.py                      # Modern Streamlit Web Application
├── main.py                     # Local OpenCV real-time pipeline
├── signlanguage.py             # Highly functional data collection HUD
├── train_model.py              # ML training pipeline with 5-fold CV
└── requirements.txt            # Project dependencies

## ⚙️ Setup and Installation

1. Clone the repository and navigate into it:
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME

2. Install all the required packages directly into your virtual environment:
pip install -r requirements.txt

3. To run the Local OpenCV pipeline and camera interface:
python main.py

4. To launch the styled Web Application Dashboard on your browser:
streamlit run app.py

5. To run the data collection script for capturing new gesture samples:
python signlanguage.py

6. To retrain the model after updating your dataset:
python train_model.py
