# Telecom-Customer-Churn-Prediction-App
![image](https://github.com/user-attachments/assets/eb6a387e-66f6-4ead-a780-45bad694f5ff)
# Customer Churn Prediction Web App

## Overview

This interactive web application, built using Streamlit, predicts customer churn for telecommunication companies. It enables data-driven retention strategies by allowing businesses to identify at-risk customers and take proactive measures.

## Features

- **Machine Learning Models:** Implements Logistic Regression, Random Forest, and XGBoost with hyperparameter tuning via GridSearchCV.
- **Performance:** Achieves an AUC score of 0.85, ensuring reliable churn predictions.
- **Data Preprocessing:** Handles missing values, scales features, and encodes categorical variables for a robust and clean pipeline.
- **User-Friendly Interface:** Provides a seamless experience, allowing users to input customer attributes such as tenure, payment methods, and service usage for real-time predictions.

## Installation

To set up and run the web app locally, follow these steps:

### Prerequisites

Ensure you have the following installed:

- Python (>= 3.7)
- pip
- Virtual environment (optional but recommended)

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/customer-churn-prediction.git
   cd customer-churn-prediction
   ```
2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate   # On macOS/Linux
   venv\Scripts\activate      # On Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Usage

1. Launch the web app.
2. Input customer details such as tenure, payment methods, contract type, service usage, and demographics.
3. Click the **Predict** button to see the churn probability.
4. Use the insights to devise retention strategies.

## Project Structure

```
customer-churn-prediction/
│-- app.py                # Main Streamlit app
│-- model.py              # ML models and prediction logic
│-- preprocess.py         # Data preprocessing pipeline
│-- requirements.txt      # Python dependencies
│-- data/                 # Sample datasets (if applicable)
│-- README.md             # Project documentation
```

## Model Performance

The model was trained on the Telco Customer Churn dataset and optimized using GridSearchCV, achieving an AUC score of **0.85**. The pipeline ensures reliable and interpretable predictions for business decision-making.

## Technologies Used

- **Programming Language:** Python
- **Framework:** Streamlit
- **Machine Learning:** Scikit-learn, XGBoost
- **Data Processing:** Pandas, NumPy

## Future Improvements

- Deploy the app using cloud services such as AWS or Heroku.
- Integrate explainable AI (e.g., SHAP) to enhance model interpretability.
- Extend the dataset with additional customer features for improved accuracy.

## Contributing

Contributions are welcome! Feel free to submit issues or pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

Developed by [Your Name](https://github.com/yourusername)

Give me to copy the content


