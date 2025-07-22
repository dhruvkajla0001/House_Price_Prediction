<!DOCTYPE html>
<html>
<head>
    <title>House Price Prediction Project</title>
</head>
<body>

<h1>House Price Prediction Project</h1>

<p>
This project predicts house prices using machine learning. 
It is based on the Ames Housing Dataset (from Kaggle) and uses 
<strong>XGBoost</strong> along with a <strong>data preprocessing pipeline</strong>. 
A <strong>Streamlit web app</strong> is provided for interactive predictions.
</p>

<h2>Features</h2>
<ul>
    <li>Predicts house prices based on property details.</li>
    <li>Trained XGBoost model with full preprocessing (numerical scaling + categorical encoding).</li>
    <li>Default values for all features (derived from the training dataset).</li>
    <li>Streamlit app for quick and interactive price estimation.</li>
</ul>

<h2>Project Structure</h2>
<ul>
    <li><code>train.csv</code> - Original Kaggle dataset used for training.</li>
    <li><code>final_xgb_model.pkl</code> - Saved trained pipeline (preprocessor + XGBoost).</li>
    <li><code>app.py</code> - Streamlit application for prediction.</li>
    <li><code>README.html</code> - Project documentation.</li>
</ul>

<h2>How to Run</h2>
<ol>
    <li>Install dependencies:
        <pre><code>pip install streamlit pandas numpy scikit-learn xgboost joblib</code></pre>
    </li>
    <li>Ensure the following files are in the same directory:
        <ul>
            <li><code>train.csv</code></li>
            <li><code>final_xgb_model.pkl</code></li>
            <li><code>app.py</code></li>
        </ul>
    </li>
    <li>Run the app:
        <pre><code>streamlit run app.py</code></pre>
    </li>
    <li>Open the link in your browser (usually <code>http://localhost:8501</code>).</li>
</ol>

<h2>Usage</h2>
<p>
The app allows you to enter a few key features (Overall Quality, Living Area, Garage Capacity, Basement Area). 
Other columns use typical defaults from the dataset to ensure compatibility with the trained model.
</p>

</body>
</html>
