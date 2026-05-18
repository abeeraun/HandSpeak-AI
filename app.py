import streamlit as st
import av
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
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# ══════════════════════════════════════════════════════════════
#  إعدادات الصفحة
# ══════════════════════════════════════════════════════════════
st.set_page_config(page_title="HandSpeak AI", page_icon="🤟", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=JetBrains+Mono&display=swap');

:root {
    --bg: #0a0a0f;
    --card: #12121a;
    --accent: #6c63ff;
    --green: #00d4aa;
    --red: #ff6b6b;
    --text: #f0f0ff;
    --muted: #6b6b8a;
    --border: rgba(108,99,255,0.2);
}

html, body, .stApp { background: var(--bg) !important; font-family: 'Space Grotesk', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem !important; max-width: 100% !important; }

h1 { color: var(--text) !important; font-size: 2rem !important; font-weight: 700 !important; letter-spacing: -1px !important; }
h1 span { color: var(--accent); }

.card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
}

.pred-word {
    font-size: 48px;
    font-weight: 700;
    color: var(--accent);
    text-align: center;
    min-height: 60px;
    text-shadow: 0 0 40px rgba(108,99,255,0.5);
    letter-spacing: -1px;
}

.conf-bar-bg {
    background: rgba(255,255,255,0.05);
    border-radius: 4px;
    height: 8px;
    margin-top: 8px;
    overflow: hidden;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, #6c63ff, #00d4aa);
    transition: width 0.3s;
}

.word-chip {
    display: inline-block;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 8px;
    padding: 6px 12px;
    margin: 4px;
    font-size: 13px;
    color: var(--text);
}
.word-chip.active {
    border-color: var(--accent);
    background: rgba(108,99,255,0.15);
    color: var(--accent);
    font-weight: 600;
}

.section-label {
    color: var(--muted);
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 2px;
    font-weight: 500;
    margin-bottom: 12px;
}

.hist-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 7px 12px;
    background: rgba(255,255,255,0.02);
    border-radius: 8px;
    margin-bottom: 5px;
    border: 1px solid rgba(255,255,255,0.04);
}
.hist-word { color: var(--text); font-size: 14px; font-weight: 500; }
.hist-time { color: var(--muted); font-size: 11px; font-family: 'JetBrains Mono', monospace; }

/* webrtc video */
.stVideo video { border-radius: 12px; width: 100% !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  هندسة الميزات
# ══════════════════════════════════════════════════════════════
_KEY_PAIRS = [
    (0,4),(0,8),(0,12),(0,16),(0,20),
    (4,8),(8,12),(12,16),(16,20),
    (4,12),(8,16),(4,20),(8,20)
]

def extract_features(landmarks):
    coords   = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    relative = coords - coords[0]
    dists    = [np.linalg.norm(coords[a] - coords[b]) for a, b in _KEY_PAIRS]
    return np.concatenate([relative.flatten(), dists])

# ══════════════════════════════════════════════════════════════
#  تحميل الموديل (مرة واحدة فقط)
# ══════════════════════════════════════════════════════════════
@st.cache_resource
def load_model():
    with open('models/model.pkl', 'rb') as f:
        bundle = pickle.load(f)
    if isinstance(bundle, dict):
        return bundle['model'], bundle['scaler']
    return bundle, None

@st.cache_resource
def load_landmarker():
    base_options = python.BaseOptions(model_asset_path='models/hand_landmarker.task')
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.IMAGE,   # IMAGE mode للـ webrtc
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
    )
    return vision.HandLandmarker.create_from_options(options)

model, scaler      = load_model()
landmarker_shared  = load_landmarker()

# ══════════════════════════════════════════════════════════════
#  الصوت
# ══════════════════════════════════════════════════════════════
_engine_lock = threading.Lock()

def speak(text: str):
    def _t(s):
        with _engine_lock:
            try:
                e = pyttsx3.init()
                e.setProperty('rate', 165)
                e.say(s)
                e.runAndWait()
                e.stop()
            except: pass
    threading.Thread(target=_t, args=(text,), daemon=True).start()

# ══════════════════════════════════════════════════════════════
#  معالج الفيديو (webrtc)
# ══════════════════════════════════════════════════════════════
WORDS_MAP = {
    "Hello":"👋", "Peace":"✌️", "Yes":"✊", "No":"🗣️",
    "Good":"👍", "Bad":"👎", "Water":"💧", "i love you":"❤️"
}
MIN_CONFIDENCE = 0.72
STABLE_FRAMES  = 5

class SignProcessor(VideoProcessorBase):
    def __init__(self):
        self.current_label  = ""
        self.last_raw_pred  = ""
        self.stable_count   = 0
        self.word           = ""
        self.conf           = 0.0
        self.history        = deque(maxlen=6)
        self._lock          = threading.Lock()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w = img.shape[:2]

        rgb    = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker_shared.detect(mp_img)   # IMAGE mode

        with self._lock:
            if result.hand_landmarks:
                for hand_lms in result.hand_landmarks:
                    feats = extract_features(hand_lms)
                    if scaler is not None:
                        feats = scaler.transform([feats])[0]
                    pred  = model.predict([feats])[0]
                    probs = model.predict_proba([feats])[0]
                    conf  = float(np.max(probs))

                    if pred == self.last_raw_pred:
                        self.stable_count += 1
                    else:
                        self.stable_count  = 0
                        self.last_raw_pred = pred

                    if self.stable_count >= STABLE_FRAMES and conf >= MIN_CONFIDENCE:
                        self.word = pred
                        self.conf = conf
                        if pred != self.current_label:
                            speak(pred)
                            self.current_label = pred
                            self.history.appendleft({'word': pred, 'time': time.strftime('%H:%M:%S')})
                            self.stable_count = 0

                    # رسم على الصورة
                    color = (108, 99, 255)
                    for lm in hand_lms:
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(img, (cx, cy), 5, color, -1)
                        cv2.circle(img, (cx, cy), 7, (255, 255, 255), 1)

                    if self.word and conf >= MIN_CONFIDENCE:
                        txt = f"{self.word}  {conf*100:.0f}%"
                        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
                        cv2.rectangle(img, (15, 15), (tw + 35, th + 40), (10,10,25), -1)
                        cv2.rectangle(img, (15, 15), (tw + 35, th + 40), (108, 99, 255), 2)
                        cv2.putText(img, txt, (25, th + 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 190, 255), 2)
            else:
                self.current_label = ""
                self.stable_count  = 0
                self.last_raw_pred = ""
                self.word = ""
                self.conf = 0.0
                cv2.putText(img, "No Hand Detected", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 100), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ══════════════════════════════════════════════════════════════
#  الواجهة
# ══════════════════════════════════════════════════════════════
st.markdown("<h1>Hand<span>Speak</span> AI 🤟</h1>", unsafe_allow_html=True)
st.markdown('<p style="color:#6b6b8a; margin-top:-10px; margin-bottom:24px;">مترجم لغة الإشارة — الوقت الحقيقي</p>', unsafe_allow_html=True)

col_cam, col_info = st.columns([3, 1], gap="large")

with col_cam:
    ctx = webrtc_streamer(
        key="sign-language",
        video_processor_factory=SignProcessor,
        rtc_configuration=RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        ),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col_info:
    pred_ph    = st.empty()
    words_ph   = st.empty()
    history_ph = st.empty()

    # تحديث لوحة المعلومات
    while True:
        time.sleep(0.15)

        word = ""
        conf = 0.0
        hist = []

        if ctx.video_processor:
            with ctx.video_processor._lock:
                word = ctx.video_processor.word
                conf = ctx.video_processor.conf
                hist = list(ctx.video_processor.history)

        # بطاقة التوقع
        conf_pct = conf * 100
        pred_ph.markdown(f"""
        <div class="card" style="text-align:center;">
            <div class="section-label">الكلمة المترجمة</div>
            <div class="pred-word">{'—' if not word else word}</div>
            <div style="font-size:24px; margin-top:4px;">{WORDS_MAP.get(word,'') if word else ''}</div>
            <div style="display:flex;justify-content:space-between;color:#6b6b8a;font-size:12px;margin-top:14px;">
                <span>الثقة</span>
                <span style="color:#00d4aa;font-family:'JetBrains Mono',monospace;font-weight:600;">{conf_pct:.1f}%</span>
            </div>
            <div class="conf-bar-bg">
                <div class="conf-bar-fill" style="width:{min(conf_pct,100):.0f}%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # شبكة الكلمات
        chips = "".join(
            f'<span class="word-chip {"active" if w == word else ""}">{e} {w}</span>'
            for w, e in WORDS_MAP.items()
        )
        words_ph.markdown(f"""
        <div class="card">
            <div class="section-label">الكلمات المدعومة</div>
            {chips}
        </div>
        """, unsafe_allow_html=True)

        # التاريخ
        rows_html = "".join(
            f'<div class="hist-row"><span class="hist-word">{WORDS_MAP.get(i["word"],"")} {i["word"]}</span>'
            f'<span class="hist-time">{i["time"]}</span></div>'
            for i in hist
        ) or '<div style="color:#6b6b8a;font-size:13px;padding:8px 0;">لا يوجد تاريخ بعد</div>'
        history_ph.markdown(f"""
        <div class="card">
            <div class="section-label">التاريخ</div>
            {rows_html}
        </div>
        """, unsafe_allow_html=True)