# 🔬 HistoAI — Breast Cancer Histopathology Classification using CNN

HistoAI is an AI-powered diagnostic assistant that classifies breast histopathology images as **Benign** or **Malignant** using a Convolutional Neural Network (CNN) trained on the **BreakHis (Breast Cancer Histopathological Database)** dataset.

The application provides an end-to-end clinical-style workflow, allowing users to upload microscopic tissue images, receive AI predictions with confidence scores, and visualize model attention through Grad-CAM explainability maps.

---

## 🚀 Live Demo

🔗 **Streamlit App:** https://breakhis-cnn-classifier.streamlit.app/

## 📸 Screenshots

### Home Page

<img width="2950" height="1732" alt="image" src="https://github.com/user-attachments/assets/5b169c9c-2c1f-4006-a819-f6c6df99e262" />
<img width="2950" height="868" alt="image" src="https://github.com/user-attachments/assets/7cdf0879-288e-465d-b6d2-c9febbc043f7" />

### Image Upload

<img width="2950" height="1730" alt="image" src="https://github.com/user-attachments/assets/78f80424-ad9a-4302-8a43-63362cee816a" />
<img width="2950" height="1730" alt="image" src="https://github.com/user-attachments/assets/37599a47-77aa-44f9-9ff1-7eafe3e1436f" />
<img width="2950" height="1730" alt="image" src="https://github.com/user-attachments/assets/e019da13-1a65-49da-91c4-2ba93ab95d5b" />

### Diagnostic Results

<img width="2950" height="1730" alt="image" src="https://github.com/user-attachments/assets/caa85dfa-2b12-4785-b840-a3303ea446f3" />
<img width="2950" height="1394" alt="image" src="https://github.com/user-attachments/assets/2e613ee2-9758-4cd0-97b8-b972f01e1ba5" />

### Confidence Analysis Dashboard

<img width="2950" height="1750" alt="image" src="https://github.com/user-attachments/assets/b8cb5717-75d2-43f7-a1f3-9863e794e030" />
<img width="2950" height="1030" alt="image" src="https://github.com/user-attachments/assets/c9301dd8-ab27-481e-8eb1-a0e67be22b49" />

---

## 📌 Features

### 🧠 Deep Learning-Based Classification

* CNN-based breast cancer histopathology classifier
* Binary classification:

  * Benign
  * Malignant
* Confidence score generation
* Threshold-based prediction optimization

### 🔍 Explainable AI

* Grad-CAM visualization
* Highlights image regions influencing model decisions
* Improves prediction transparency

### 🏥 Clinical-Style Interface

* Multi-page diagnostic workflow
* Confidence analysis dashboard
* Probability distribution visualization
* Detailed diagnostic findings
* Modern pathology-inspired UI

### 🛡 Input Validation

* Histopathology image verification
* Rejects non-medical or out-of-distribution images
* Uncertainty detection for ambiguous samples

### ⚡ Efficient Deployment

* Streamlit-based web application
* Lightweight inference pipeline
* Cached model loading for faster predictions

---

## 🧬 Dataset

This project uses the **BreakHis Dataset (Breast Cancer Histopathological Database)**.

### Dataset Characteristics

* Breast tumor histopathology images
* Benign and Malignant classes
* Multiple magnification factors:

  * 40X
  * 100X
  * 200X
  * 400X
* Thousands of annotated microscopic images

The dataset is widely used for evaluating deep learning models in digital pathology and breast cancer classification research.

---

## 🏗 Model Architecture

The CNN architecture consists of:

* 5 Convolutional Blocks
* Batch Normalization
* ReLU Activation
* Max Pooling
* Global Average Pooling
* Fully Connected Dense Layers
* Dropout Regularization
* Sigmoid Output Layer

### Architecture Flow

Input Image (224×224×3)
↓
Conv2D (32)
↓
Conv2D (64)
↓
Conv2D (128)
↓
Conv2D (256)
↓
Conv2D (512)
↓
Global Average Pooling
↓
Dense (256)
↓
Dropout (0.5)
↓
Dense (128)
↓
Dropout (0.3)
↓
Sigmoid Output
↓
Benign / Malignant Prediction

---

## 📊 Model Performance

| Metric    | Score |
| --------- | ----- |
| Accuracy  | 89.1% |
| Precision | 99.8% |
| Recall    | 84.3% |
| F1 Score  | 91.4% |
| AUC       | 97.8% |

These metrics are displayed directly within the application to provide transparency regarding model performance.

---

## 🖥 Application Workflow

### 1️⃣ Upload Histopathology Image

Users upload a breast tissue microscopic image.

### 2️⃣ AI Analysis

The CNN processes the image and generates:

* Prediction label
* Confidence score
* Class probabilities

### 3️⃣ Results Dashboard

Displays:

* Predicted class
* Confidence ring
* Probability distribution
* Diagnostic interpretation

### 4️⃣ Explainability

Users can switch between:

* Original Image
* Grad-CAM Heatmap

to understand what influenced the model's prediction.

---

## 📂 Project Structure

```bash
HistoAI/
│
├── app.py
├── model_utils.py
├── requirements.txt
├── breakhis_sklearn.joblib
├── CNN_Classifier.ipynb
├── train_sklearn_model.py
│
├── screenshots/
│   ├── home.png
│   ├── upload.png
│   ├── results.png
│   └── confidence.png
│
└── README.md
```

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/histoai.git
cd histoai
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
streamlit run app.py
```

---

## 🛠 Tech Stack

### Machine Learning

* TensorFlow / Keras
* NumPy
* Scikit-learn

### Computer Vision

* Pillow (PIL)
* Grad-CAM Visualization

### Web Application

* Streamlit

### Model Persistence

* Joblib

---

## ⚠️ Medical Disclaimer

This project is intended for educational, research, and demonstration purposes only.

The predictions generated by HistoAI are not medical diagnoses and should not be used as a substitute for professional medical advice, pathology review, or clinical decision-making.

Always consult qualified healthcare professionals for diagnosis and treatment decisions.

---

## 👩‍💻 Author

**Ayesha Tariq**

AI & Machine Learning Enthusiast • Building impactful AI solutions for real-world healthcare applications.

If you found this project interesting, consider giving the repository a ⭐.

## 📄 License

This project is licensed under the MIT License.
