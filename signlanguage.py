import cv2
import mediapipe as mp
import csv
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ══════════════════════════════════════════════════════
#  إعداد MediaPipe
# ══════════════════════════════════════════════════════
base_options = python.BaseOptions(model_asset_path='models/hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.6,
)
landmarker = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

label     = input("أدخل اسم الحركة (مثلاً: Thank You): ").strip()
TARGET    = int(input("كم عينة تريد جمعها؟ [100]: ") or 100)
AUTO_MODE = input("وضع التقاط تلقائي؟ (y/n) [y]: ").strip().lower() != 'n'
AUTO_INTERVAL = 0.15  # ثانية بين كل التقاط تلقائي

img_count   = 0
last_auto_t = 0

print(f"\n✅ جاهز لتسجيل ({label}) — هدف: {TARGET} عينة")
print("  [S] لحفظ يدوي | [A] لتفعيل/تعطيل التلقائي | [Q] للخروج\n")

with open('data/hand_data.csv', mode='a', newline='') as f:
    writer = csv.writer(f)

    while cap.isOpened() and img_count < TARGET:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ts_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(mp_img, ts_ms)

        # خلفية شفافة
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 110), (10, 10, 20), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # شريط التقدم
        progress   = img_count / TARGET
        bar_w      = w - 40
        filled     = int(bar_w * progress)
        cv2.rectangle(frame, (20, 80), (20 + bar_w, 95), (30, 30, 50), -1)
        cv2.rectangle(frame, (20, 80), (20 + filled, 95), (108, 99, 255), -1)

        # نصوص المعلومات
        cv2.putText(frame, f"Label: {label}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (108, 230, 200), 2)
        cv2.putText(frame, f"Captured: {img_count}/{TARGET}  ({progress*100:.0f}%)",
                    (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 255), 2)

        cv2.putText(frame, f"AUTO: {'ON' if AUTO_MODE else 'OFF'}",
                    (w - 160, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 212, 170) if AUTO_MODE else (255, 107, 107), 2)

        captured_this_frame = False

        if result.hand_landmarks:
            for hand_lms in result.hand_landmarks:
                # رسم النقاط
                for lm in hand_lms:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 5, (108, 99, 255), -1)
                    cv2.circle(frame, (cx, cy), 7, (255, 255, 255), 1)

                # التقاط تلقائي
                now = time.time()
                if AUTO_MODE and (now - last_auto_t) >= AUTO_INTERVAL:
                    row = []
                    for lm in hand_lms:
                        row.extend([lm.x, lm.y, lm.z])
                    row.append(label)
                    writer.writerow(row)
                    img_count += 1
                    last_auto_t = now
                    captured_this_frame = True
                    print(f"  [{img_count}/{TARGET}] ✅ {label}")
        else:
            cv2.putText(frame, "⚠ No Hand Detected", (w//2 - 140, h//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 107, 107), 2)

        # وميض عند الالتقاط
        if captured_this_frame:
            cv2.rectangle(frame, (0, 0), (w, h), (108, 99, 255), 4)

        cv2.imshow(f"Data Collection — {label}", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s') and result.hand_landmarks:
            for hand_lms in result.hand_landmarks:
                row = []
                for lm in hand_lms:
                    row.extend([lm.x, lm.y, lm.z])
                row.append(label)
                writer.writerow(row)
                img_count += 1
                print(f"  [{img_count}/{TARGET}] ✅ {label}")
        elif key == ord('a'):
            AUTO_MODE = not AUTO_MODE
            print(f"  Auto mode: {'ON' if AUTO_MODE else 'OFF'}")
        elif key == ord('q'):
            break

print(f"\n✅ تم جمع {img_count} عينة للحركة ({label})")
cap.release()
cv2.destroyAllWindows()