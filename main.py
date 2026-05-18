import cv2
import mediapipe as mp
import pickle
import numpy as np
import pyttsx3
import threading
import time
from collections import deque
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ══════════════════════════════════════════════════════
#  نظام الصوت
# ══════════════════════════════════════════════════════
_engine_lock = threading.Lock()

def speak(text: str):
    def _task(t):
        with _engine_lock:
            try:
                eng = pyttsx3.init()
                eng.setProperty('rate', 165)
                eng.say(t)
                eng.runAndWait()
                eng.stop()
            except Exception as e:
                print(f"[Speak Error] {e}")
    threading.Thread(target=_task, args=(text,), daemon=True).start()

# ══════════════════════════════════════════════════════
#  هندسة الميزات
# ══════════════════════════════════════════════════════
_KEY_PAIRS = [
    (0,4),(0,8),(0,12),(0,16),(0,20),
    (4,8),(8,12),(12,16),(16,20),
    (4,12),(8,16),(4,20),(8,20)
]

def extract_features(landmarks) -> np.ndarray:
    coords   = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    relative = coords - coords[0]
    distances = [np.linalg.norm(coords[a] - coords[b]) for a, b in _KEY_PAIRS]
    return np.concatenate([relative.flatten(), distances])

# ══════════════════════════════════════════════════════
#  تحميل الموديل
# ══════════════════════════════════════════════════════
with open('models/model.pkl', 'rb') as f:
    bundle = pickle.load(f)

if isinstance(bundle, dict):
    model, scaler = bundle['model'], bundle['scaler']
else:
    model, scaler = bundle, None

# ══════════════════════════════════════════════════════
#  إعداد MediaPipe
# ══════════════════════════════════════════════════════
base_options = python.BaseOptions(model_asset_path='models/hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.6,
)
landmarker = vision.HandLandmarker.create_from_options(options)

# ══════════════════════════════════════════════════════
#  إعداد الكاميرا
# ══════════════════════════════════════════════════════
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# ══════════════════════════════════════════════════════
#  متغيرات الحالة
# ══════════════════════════════════════════════════════
current_label  = ""
last_raw_pred  = ""
stable_count   = 0
STABLE_FRAMES  = 5
MIN_CONFIDENCE = 0.72

fps_times  = deque(maxlen=30)
history    = deque(maxlen=5)

WORDS_COLORS = {
    "Hello":      (108, 230, 108),
    "Peace":      (108, 200, 255),
    "Yes":        (108, 255, 200),
    "No":         (255, 108, 108),
    "Good":       (200, 255, 108),
    "Bad":        (255, 150, 50),
    "Water":      (50, 200, 255),
    "i love you": (255, 108, 200),
}

print("✅ النظام جاهز | اضغط Q للخروج")

while cap.isOpened():
    t0 = time.time()
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w  = frame.shape[:2]
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    ts_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect_for_video(mp_img, ts_ms)

    # خلفية شفافة للـ UI
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), (10, 10, 20), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    if result.hand_landmarks:
        for hand_lms in result.hand_landmarks:
            feats = extract_features(hand_lms)
            if scaler is not None:
                feats = scaler.transform([feats])[0]

            pred  = model.predict([feats])[0]
            probs = model.predict_proba([feats])[0]
            conf  = float(np.max(probs))

            # منطق التثبيت
            if pred == last_raw_pred:
                stable_count += 1
            else:
                stable_count  = 0
                last_raw_pred = pred

            if stable_count >= STABLE_FRAMES and conf >= MIN_CONFIDENCE:
                if pred != current_label:
                    speak(pred)
                    current_label = pred
                    history.appendleft({'word': pred, 'time': time.strftime('%H:%M')})
                    stable_count = 0

            # رسم نقاط اليد
            color = WORDS_COLORS.get(current_label, (108, 99, 255))
            for lm in hand_lms:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 5, color, -1)
                cv2.circle(frame, (cx, cy), 7, (255,255,255), 1)

            # عرض التوقع
            if conf >= MIN_CONFIDENCE:
                txt   = f"{pred}   {conf*100:.0f}%"
                color_bgr = color
                (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
                cv2.rectangle(frame, (20, 15), (tw + 50, th + 45), (10,10,25), -1)
                cv2.rectangle(frame, (20, 15), (tw + 50, th + 45), color_bgr, 2)
                cv2.putText(frame, txt, (35, th + 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_bgr, 2)
    else:
        # اليد اختفت
        current_label = ""
        stable_count  = 0
        last_raw_pred = ""
        cv2.putText(frame, "No Hand", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 120), 2)

    # FPS
    fps_times.append(time.time() - t0)
    fps = 1 / (sum(fps_times)/len(fps_times)) if fps_times else 0
    cv2.putText(frame, f"FPS: {fps:.0f}", (w - 110, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 100), 2)

    # عرض التاريخ
    for i, item in enumerate(history):
        alpha = 1.0 - (i * 0.18)
        color = (int(180*alpha), int(180*alpha), int(200*alpha))
        cv2.putText(frame, item['word'], (20, h - 30 - i*30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("HandSpeak AI — Sign Language Translator", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ تم إغلاق النظام.")