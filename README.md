# Iris Flower Classification using Machine Learning

This project implements a **machine learning pipeline** to classify Iris flowers into three species (*Setosa, Versicolor, Virginica*) based on sepal and petal measurements.

The notebook provides a clean, reproducible workflow with **EDA, preprocessing, model comparison, hyperparameter tuning, evaluation, and model persistence**.

---

## 📌 Project Overview

* **Dataset**: The classic [Iris dataset](https://archive.ics.uci.edu/dataset/53/iris), widely used for beginner machine learning tasks.
* **Objective**: Predict the flower species based on four input features:

  * Sepal Length
  * Sepal Width
  * Petal Length
  * Petal Width

---

## ⚙️ Features

* Data loading and quick **Exploratory Data Analysis (EDA)**
* **Preprocessing pipeline**: Missing value handling + scaling
* Comparison of multiple models:

  * Logistic Regression
  * Random Forest
  * Support Vector Classifier
* **Cross-validation** for fair performance comparison
* **GridSearchCV** hyperparameter tuning
* Evaluation using accuracy, classification report, and confusion matrix
* **Model persistence** with `joblib`

---

## 🧰 Tech Stack

* **Languages**: Python 3
* **Libraries**:

  * `pandas`, `numpy`, `matplotlib`
  * `scikit-learn`
  * `joblib`

---

## 🚀 How to Run

1. Clone this repository:

   ```bash
   git clone https://github.com/your-username/iris-flower-classification.git
   cd iris-flower-classification
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Open the notebook:

   ```bash
   jupyter notebook iris_ml_classification_clean.ipynb
   ```

---

## 📊 Results

* Achieved **high accuracy (>95%)** across models.
* Best model: **Random Forest (with tuned hyperparameters)**.
* Example confusion matrix:

|                       | Predicted Setosa | Predicted Versicolor | Predicted Virginica |
| --------------------- | ---------------- | -------------------- | ------------------- |
| **Actual Setosa**     | ✅                |                      |                     |
| **Actual Versicolor** |                  | ✅                    |                     |
| **Actual Virginica**  |                  |                      | ✅                   |

---

## 💾 Model Deployment

The final tuned model is saved as:

```bash
iris_best_model.joblib
```

You can load it in Python:

```python
import joblib
model = joblib.load("iris_best_model.joblib")
prediction = model.predict([[5.1, 3.5, 1.4, 0.2]])  # Example input
```

---

## 📈 Next Steps

* Add **RandomizedSearchCV** for faster hyperparameter tuning.
* Integrate **model explainability** (SHAP, permutation importance).
* Build a **FastAPI/Streamlit app** for user-friendly predictions.

---

## 📚 References

* UCI Machine Learning Repository — [Iris Dataset](https://archive.ics.uci.edu/dataset/53/iris)
* Scikit-learn Documentation — [https://scikit-learn.org/](https://scikit-learn.org/)

---

✨ *This project demonstrates a complete ML workflow on a beginner-friendly dataset and serves as a template for future classification problems.*

---
