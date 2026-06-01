# ✍️ Handwritten Character Recognition using CNN

A Deep Learning project that recognizes handwritten digits and characters using Convolutional Neural Networks (CNNs). The model is trained on benchmark datasets such as MNIST and EMNIST and can be deployed through a Streamlit web application for real-time predictions.

## 📌 Project Overview

Handwritten Character Recognition (HCR) is a computer vision task that involves identifying handwritten digits or alphabets from images. This project leverages image preprocessing techniques and Convolutional Neural Networks (CNNs) to classify handwritten characters with high accuracy.

The system can be extended to recognize complete words and sentences using sequence models such as CRNN (Convolutional Recurrent Neural Networks).

---

## 🎯 Objective

To build a robust handwritten character recognition system capable of:

* Recognizing handwritten digits (0-9)
* Recognizing handwritten alphabets (A-Z, a-z)
* Predicting characters from uploaded images
* Demonstrating predictions through an interactive Streamlit application

---

## 🧠 Model Architecture

The project uses a Convolutional Neural Network (CNN) consisting of:

* Convolution Layers
* ReLU Activation
* Max Pooling
* Batch Normalization
* Dropout
* Fully Connected Layers
* Softmax Output Layer

Example Architecture:

Input Image (28×28)
↓
Conv2D → ReLU
↓
MaxPool
↓
Conv2D → ReLU
↓
MaxPool
↓
Flatten
↓
Dense Layer
↓
Dropout
↓
Output Layer

---

## 📊 Dataset

### MNIST

* 70,000 handwritten digit images
* 28×28 grayscale images
* Classes: 0–9

### EMNIST

* Extended MNIST dataset
* Handwritten letters and digits
* Multiple splits available:

  * Letters
  * Balanced
  * ByClass
  * Digits

Dataset Sources:

* https://www.kaggle.com/datasets
* https://www.nist.gov/itl/products-and-services/emnist-dataset

---

## 🛠️ Tech Stack

### Deep Learning

* TensorFlow / Keras
* NumPy
* Pandas

### Visualization

* Matplotlib
* Seaborn

### Deployment

* Streamlit

### Development

* Google Colab
* Kaggle Notebooks
* VS Code

---

## 📂 Project Structure

```text
Handwritten-Character-Recognition/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── EDA.ipynb
│   └── Training.ipynb
│
├── models/
│   └── handwritten_characters_cnn.keras
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── predict.py
│   └── utils.py
│
├── app.py
│
├── images/
│   ├── demo.png
│   └── architecture.png
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Installation

Clone Repository

```bash
git clone https://github.com/yourusername/Handwritten-Character-Recognition.git

cd Handwritten-Character-Recognition
```

Create Virtual Environment

```bash
python -m venv venv
```

Activate Environment

Windows

```bash
venv\Scripts\activate
```

Linux/Mac

```bash
source venv/bin/activate
```

Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Training

```bash
python src/train.py
```

For local training in VS Code on Windows:

```bash
python -m venv myenv
myenv\Scripts\activate
pip install -r requirements.txt
python src/train.py
```

The first run downloads and prepares MNIST digits and `emnist/letters`
automatically inside `data/raw/`. You do not need to download either dataset
manually. The best model is saved as `models/handwritten_characters_cnn.keras`.

To test one image or launch the local web app:

```bash
python src/predict.py path\to\character.png
streamlit run app.py
```

or train directly using:

* Google Colab
* Kaggle Notebook
* VS Code Jupyter

The trained model will be saved inside:

```text
models/handwritten_characters_cnn.keras
```

---

## 🖥️ Running Streamlit App

```bash
streamlit run app.py
```

Features:

* Upload handwritten image
* Predict character
* Display confidence score
* Visualization of prediction

---

## 📈 Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

Expected Performance:

| Dataset | Accuracy |
| ------- | -------- |
| MNIST   | 98%+     |
| EMNIST  | 90–95%   |

---

## 🔮 Future Improvements

* Word Recognition
* Sentence Recognition
* CRNN Architecture
* Transformer-Based OCR
* Real-Time Camera Input
* ONNX/TensorRT Optimization

---

## 📷 Demo

Add screenshots of:

* Training Results
* Streamlit Interface
* Prediction Examples

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome.

Feel free to fork the repository and submit pull requests.

---

## 📜 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

Ishank Mishra

Machine Learning | Deep Learning | AI Enthusiast

If you found this project useful, consider giving it a ⭐ on GitHub.
