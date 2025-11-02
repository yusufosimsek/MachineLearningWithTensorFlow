# Machine Learning with TensorFlow 🧠

This repository contains projects and examples built with **TensorFlow**, aiming to explore and understand core **machine learning** concepts.  
It includes scripts and notebooks for data preprocessing, model training, evaluation, and saving.

---

## 🚀 Technologies Used
- **Python 3.x**
- **TensorFlow**
- **NumPy**
- **Pandas**
- **Matplotlib / Seaborn**
- **Scikit-Learn**

---

## 📂 Project Structure
```
MachineLearningWithTensorFlow/
│
├── data/                 # Datasets (CSV, txt, etc.)
├── notebooks/            # Jupyter Notebook files
├── models/               # Saved models (.h5, .keras, .pkl, etc.)
├── src/                  # Source code (model definitions, training scripts)
├── README.md             # This file
└── requirements.txt      # Required libraries
```

---

## ⚙️ Installation
To run this project locally:

```bash
# Clone the repository
git clone https://github.com/yusufosimsek/MachineLearningWithTensorFlow.git

# Move into the directory
cd MachineLearningWithTensorFlow

# Install dependencies
pip install -r requirements.txt
```

---

## 🧩 Example Usage
```python
from tensorflow import keras
import numpy as np

# Load a saved model
model = keras.models.load_model('models/my_model.keras')

# Make a prediction
prediction = model.predict(np.array([[5.1, 3.5, 1.4, 0.2]]))
print(prediction)
```

---

## 🧠 Topics Covered
- Data Preprocessing  
- Regression Models  
- Classification Algorithms  
- Deep Learning (Neural Networks)  
- Model Saving and Loading  

---

## 📜 License
This project is created for **educational and open-source** purposes.  
Feel free to use, modify, and distribute it as you wish.

---

### ✨ Author
**Yusuf Onur Şimşek**  
📘 [GitHub Profile](https://github.com/yusufosimsek)
