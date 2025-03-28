
# 🚢 Titanic Survival Prediction

This project predicts whether a passenger survived the Titanic disaster using Machine Learning. It implements a full pipeline including data cleaning, exploratory data analysis (EDA), model training, and evaluation.

## 📁 Project Structure
```
├── Titanic_Survival_Prediction_with_Model.py  # Main Python script (train & save model)
├── tested.csv                                 # Input dataset (Titanic data)
├── titanic_model.pkl                          # Saved Random Forest model
├── survival_distribution.png                  # Survival distribution plot
├── correlation_heatmap.png                    # Feature correlation heatmap
└── confusion_matrix.png                       # Model confusion matrix
```

## 🧰 Requirements

Ensure you have Python 3.x installed. You can install required packages using:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

## ▶️ How to Run

1. Ensure `tested.csv` is present in the same directory as the script.  
2. Execute the Python script:

```bash
python Titanic_Survival_Prediction_with_Model.py
```

## 📊 Outputs

1. **titanic_model.pkl**: Saved trained Random Forest model.  
2. **survival_distribution.png**: Visualizes survival rates.  
3. **correlation_heatmap.png**: Displays feature correlations.  
4. **confusion_matrix.png**: Shows model predictions vs. actual outcomes.  

## 🔍 How to Load and Use the Saved Model

You can load and use the saved model (`titanic_model.pkl`) for future predictions:

```python
import joblib
import pandas as pd

# Load the model
model = joblib.load('titanic_model.pkl')

# Example input for prediction (modify as per your dataset schema)
sample_input = pd.DataFrame({
    'Pclass': [3],
    'Sex': [1],  # 0: female, 1: male
    'Age': [29],
    'SibSp': [0],
    'Parch': [0],
    'Fare': [7.25],
    'Embarked': [1]  # 0: C, 1: Q, 2: S (encoded)
})

# Predict survival outcome
prediction = model.predict(sample_input)
print("Survival Prediction:", prediction)
```

## 📌 Project Goal

- Perform end-to-end Machine Learning on Titanic data.
- Build a robust and interpretable Random Forest model.
- Save the trained model for future inference.

## 📧 Contact

For any questions or improvements, feel free to connect!

