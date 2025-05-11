# Lumbar Spine Degeneration Classification using EfficientNet

This project focuses on classifying **25 lumbar spine degenerative conditions** from **MRI DICOM images** using a deep learning model built on **EfficientNet**. The pipeline includes data preparation, model training, evaluation, and visual explainability.

---

## 📁 Dataset

The dataset includes the following files:
- `reduced_merged_train_data.csv`
- `test_series_descriptions.csv`
- `train_images/`
- `test_images/`

---

## 🧠 Model Architecture

- **Backbone:** Pretrained **EfficientNetB3**
- **Input:** MRI images (Axial, Sagittal T1, Sagittal T2 views)
- **Output:** 25 multi-label outputs for spinal condition severities
  - Conditions: Neural Foraminal Narrowing (L/R), Subarticular Stenosis (L/R), Spinal Canal Stenosis
  - Levels: L1/L2 to L5/S1
  - Classes: `normal_mild`, `moderate`, `severe`

The EfficientNet backbone is fine-tuned with a custom classification head for multi-output prediction. Dropout and batch normalization are applied for regularization.

---

## ⚙️ Training & Evaluation

- **Loss Function:** Categorical cross-entropy (multi-output)
- **Optimizer:** Adam
- **Metrics:** Accuracy, F1-score, Precision, Recall, Weighted Log Loss
- **Data Augmentation:** Applied using `ImageDataGenerator` for better generalization
- **Explainability:** Implemented using **Grad-CAM** to highlight regions influencing predictions

---

## 🔍 Output

- `test_predictions.csv` — Final test predictions in required format
- `training_log.csv`- Traning and validation test info
- `training_accuracy.png` 
- `training_loss.png`
- `gradcam_visuals/` — Grad-CAM heatmaps for model explainability



