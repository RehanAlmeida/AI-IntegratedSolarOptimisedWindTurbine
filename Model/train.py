import os
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
)

# ✅ Create directory to save charts
analysis_folder = "modelanalysis"
os.makedirs(analysis_folder, exist_ok=True)

# ✅ Step 1: Load Dataset
file_path = "bandraweatherdataset.csv"
df = pd.read_csv(file_path)

# ✅ Step 2: Select Relevant Features
df = df[['wind_speed', 'humidity', 'temperature', 'wind_direction']]

# ✅ Step 3: Generate Labels (0 = Solar, 1 = Wind)
df['label'] = np.where(
    (df['wind_speed'] > 2.0) & (df['temperature'] < 30), 1,  # Wind Power
    0  # Solar Power
)

# ✅ Step 4: Split Data into Train & Test
X = df[['wind_speed', 'humidity', 'temperature', 'wind_direction']]
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Step 5: Train Random Forest Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ✅ Step 6: Model Evaluation
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]  # Probabilities for class 1 (Wind)

# 🔹 Accuracy Score
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# 🔹 Classification Report
report = classification_report(y_test, y_pred)
print("\nClassification Report:\n", report)

# 🔹 Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", conf_matrix)

# ✅ Step 7: Save Evaluation Charts
# 📌 1. Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Solar', 'Wind'], yticklabels=['Solar', 'Wind'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.savefig(os.path.join(analysis_folder, "confusion_matrix.png"))
plt.close()

# 📌 2. AUC-ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.savefig(os.path.join(analysis_folder, "roc_curve.png"))
plt.close()

# 📌 3. Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_prob)
plt.figure()
plt.plot(recall, precision, color='green', lw=2)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.savefig(os.path.join(analysis_folder, "precision_recall_curve.png"))
plt.close()

# 📌 4. Feature Importance Plot
feature_importance = model.feature_importances_
plt.figure()
sns.barplot(x=feature_importance, y=X.columns, palette="viridis")
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("Feature Importance")
plt.savefig(os.path.join(analysis_folder, "feature_importance.png"))
plt.close()

# 📌 5. Histogram of Predictions
plt.figure()
sns.histplot(y_pred, bins=3, kde=False, color="blue")
plt.xlabel("Predicted Class")
plt.ylabel("Count")
plt.title("Histogram of Predictions")
plt.savefig(os.path.join(analysis_folder, "prediction_histogram.png"))
plt.close()

# 📌 6. Boxplot of Temperature by Class
plt.figure()
sns.boxplot(x=y, y=df["temperature"], palette="Set3")
plt.xlabel("Class (0 = Solar, 1 = Wind)")
plt.ylabel("Temperature")
plt.title("Temperature Distribution by Class")
plt.savefig(os.path.join(analysis_folder, "temperature_boxplot.png"))
plt.close()

# 📌 7. Scatter Plot of Wind Speed vs Temperature
plt.figure()
sns.scatterplot(x=df["wind_speed"], y=df["temperature"], hue=df["label"], palette="coolwarm")
plt.xlabel("Wind Speed")
plt.ylabel("Temperature")
plt.title("Wind Speed vs Temperature")
plt.savefig(os.path.join(analysis_folder, "wind_speed_vs_temperature.png"))
plt.close()

# 📌 8. Pair Plot of Features
pairplot = sns.pairplot(df, hue="label", palette="husl")
pairplot.savefig(os.path.join(analysis_folder, "pairplot.png"))
plt.close()

# 📌 9. Violin Plot of Humidity by Class
plt.figure()
sns.violinplot(x=y, y=df["humidity"], palette="pastel")
plt.xlabel("Class (0 = Solar, 1 = Wind)")
plt.ylabel("Humidity")
plt.title("Humidity Distribution by Class")
plt.savefig(os.path.join(analysis_folder, "humidity_violin_plot.png"))
plt.close()

# 📌 10. Correlation Heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.savefig(os.path.join(analysis_folder, "correlation_heatmap.png"))
plt.close()

# # ✅ Step 8: Save Model
# joblib.dump(model, "solar_wind_classifier.pkl")

# # ✅ Step 9: Load Model & Test on New Sample
# model = joblib.load("solar_wind_classifier.pkl")
# sample_data = np.array([[1.5, 60, 29, 180]])  # wind_speed, humidity, temperature, wind_direction
# prediction = model.predict(sample_data)
# print("\nPredicted Class:", "Wind" if prediction[0] == 1 else "Solar")

# print(f"\n✅ All evaluation charts are saved in '{analysis_folder}' folder.")



"""
Model Parameters:


Total unique dates: 155

Descriptive Statistics:
          wind_speed   humidity  temperature  wind_direction
min         0.000000  21.370000    21.120000        0.000000
max         3.720000  95.950000    38.860000      359.900000
mean        0.461802  73.082204    28.355154      195.927966
median      0.350000  75.380000    28.460000      189.100000
<lambda>    0.000000  78.990000    26.080000      187.900000 

Unique values in categorical columns:
city_name: 1
locality_name: 1
rain_intensity: 23
rain_accumulation: 642




Accuracy: 1.0000

Classification Report:
               precision    recall  f1-score   support

           0       1.00      1.00      1.00     39989
           1       1.00      1.00      1.00        65

    accuracy                           1.00     40054
   macro avg       1.00      1.00      1.00     40054
weighted avg       1.00      1.00      1.00     40054


Confusion Matrix:
 [[39989     0]
 [    0    65]]

"""