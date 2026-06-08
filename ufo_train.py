import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

np.random.seed(42)
n_samples = 20000

countries = ['us', 'ca', 'gb', 'au', 'de', 'fr', 'other']
proportions = [0.40, 0.13, 0.12, 0.08, 0.08, 0.09, 0.10]
centers = {
    'us':    (39.8,  -98.6),
    'ca':    (56.1, -106.3),
    'gb':    (55.4,   -3.4),
    'au':    (-25.3, 133.8),
    'de':    (51.2,   10.4),
    'fr':    (46.6,    2.2),
    'other': (20.0,    0.0),
}

counts = [int(p * n_samples) for p in proportions]
counts[-1] += n_samples - sum(counts)  # absorb rounding remainder

lat_list, lon_list, country_list = [], [], []
for country, count in zip(countries, counts):
    clat, clon = centers[country]
    lats = np.clip(np.random.normal(clat, 10, count), -90, 90)
    lons = np.clip(np.random.normal(clon, 13, count), -180, 180)
    lat_list.append(lats)
    lon_list.append(lons)
    country_list.extend([country] * count)

latitudes      = np.concatenate(lat_list)
longitudes     = np.concatenate(lon_list)
country_arr    = np.array(country_list)
duration_secs  = np.clip(np.random.exponential(300, n_samples), 1, 7200).astype(int)
hour           = np.clip(np.random.normal(21, 3, n_samples), 0, 23).astype(int)
month          = np.random.randint(1, 13, n_samples)
day_of_week    = np.random.randint(0, 7, n_samples)

idx = np.random.permutation(n_samples)
latitudes, longitudes, duration_secs = latitudes[idx], longitudes[idx], duration_secs[idx]
hour, month, day_of_week, country_arr = hour[idx], month[idx], day_of_week[idx], country_arr[idx]

X = pd.DataFrame({
    'latitude':         latitudes,
    'longitude':        longitudes,
    'duration_seconds': duration_secs,
    'hour':             hour,
    'month':            month,
    'day_of_week':      day_of_week,
})

le = LabelEncoder()
y = le.fit_transform(country_arr)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

models = {
    'LogisticRegression':         LogisticRegression(max_iter=1000, random_state=42),
    'RandomForestClassifier':     RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42),
    'GradientBoostingClassifier': GradientBoostingClassifier(random_state=42),
}

results = {}

for name, clf in models.items():
    print(f"\n{'='*60}\nModel: {name}\n{'='*60}")

    cv_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')
    print(f"CV 5-fold train — mean: {cv_scores.mean():.4f}  std: {cv_scores.std():.4f}")

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    results[name] = {'model': clf, 'accuracy': acc, 'cv_mean': cv_scores.mean()}

best_name = max(results, key=lambda n: (results[n]['accuracy'], results[n]['cv_mean']))
best = results[best_name]

print(f"\n{'='*60}")
print(f"Winner: {best_name}  |  test accuracy: {best['accuracy']:.4f}  |  cv mean: {best['cv_mean']:.4f}")
print('='*60)

joblib.dump(best['model'], 'ufo_model.pkl')
joblib.dump(le, 'ufo_label_encoder.pkl')
print("Saved: ufo_model.pkl, ufo_label_encoder.pkl")

if hasattr(best['model'], 'feature_importances_'):
    importances = pd.Series(best['model'].feature_importances_, index=X.columns)
    print("\nFeature importances (sorted):")
    print(importances.sort_values(ascending=False).to_string())

report_lines = [
    f"Winner: {best_name}",
    f"Test accuracy: {best['accuracy']:.4f}",
    "",
    "All models:",
]
for name, r in results.items():
    report_lines.append(f"  {name}: accuracy={r['accuracy']:.4f}, cv_mean={r['cv_mean']:.4f}")

with open('ufo_training_report.txt', 'w') as f:
    f.write('\n'.join(report_lines) + '\n')
print("Saved: ufo_training_report.txt")
