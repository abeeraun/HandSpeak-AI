import csv
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import os

print("جاري تحميل البيانات...")
rows, labels = [], []
with open('data/hand_data.csv', newline='', encoding='utf-8') as f:
    for row in csv.reader(f):
        if not row:
            continue
        label = row[-1].strip()
        if label == 'None':
            continue
        try:
            coords = [float(x) for x in row[:-1]]
            if len(coords) == 63:
                rows.append(coords)
                labels.append(label)
        except ValueError:
            pass

X_raw = np.array(rows)
y     = np.array(labels)

unique, counts = np.unique(y, return_counts=True)
print(f"البيانات: {len(rows)} عينة | {len(unique)} فئات")
for lbl, cnt in zip(unique, counts):
    print(f"  {lbl}: {cnt}")

# ── 2. هندسة الميزات ──────────────────────────────────────────────────────────
_KEY_PAIRS = [
    (0,4),(0,8),(0,12),(0,16),(0,20),
    (4,8),(8,12),(12,16),(16,20),
    (4,12),(8,16),(4,20),(8,20)
]

def add_features(X_arr):
    out = []
    for row in X_arr:
        coords   = row.reshape(21, 3)
        relative = coords - coords[0]
        dists    = [np.linalg.norm(coords[a] - coords[b]) for a, b in _KEY_PAIRS]
        out.append(np.concatenate([relative.flatten(), dists]))
    return np.array(out)

X = add_features(X_raw)

# ── 3. تقسيم + تحجيم ─────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# ── 4. تدريب ──────────────────────────────────────────────────────────────────
print("\nجاري التدريب...")
model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

test_acc  = model.score(X_test, y_test) * 100
cv_scores = cross_val_score(model, X_train, y_train, cv=5) * 100
print(f"دقة الاختبار : {test_acc:.2f}%")
print(f"CV (5-fold)  : {cv_scores.mean():.2f}% +/- {cv_scores.std():.2f}%")

# ── 5. حفظ ───────────────────────────────────────────────────────────────────
os.makedirs('models', exist_ok=True)
with open('models/model.pkl', 'wb') as f:
    pickle.dump({'model': model, 'scaler': scaler}, f)
print("تم الحفظ في models/model.pkl")