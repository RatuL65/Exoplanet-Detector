# 🪐 Exoplanet Detection AI

**Discover new worlds with the power of Machine Learning! This interactive web app uses a highly accurate AI model to predict whether a celestial object is an exoplanet based on data from NASA's Kepler mission.**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://exoplanet-detector-reduan.streamlit.app/)



---

## ## Project Overview

This project tackles the challenge of identifying exoplanets from the vast datasets provided by NASA's space telescopes. Instead of manual analysis, this tool leverages a **Random Forest Classifier** trained on the cumulative Kepler Object of Interest (KOI) data to automatically classify planetary candidates. The model achieves an impressive **99.25% accuracy**, providing a reliable tool for both astronomers and space enthusiasts.

The entire application is built in Python and deployed as an interactive web app using **Streamlit**.

---

## ## Features ✨

- **Interactive UI**: Adjust 19 different astronomical features using sliders and input boxes to describe a potential planet.
- **High-Accuracy Model**: Utilizes a Scikit-learn Random Forest model trained on thousands of confirmed exoplanets and false positives.
- **Instant Predictions**: Get a real-time prediction—either **CONFIRMED Exoplanet** or **FALSE POSITIVE**.
- **AI Transparency**: A **Feature Importance** chart shows you which data points the AI valued most for its decision.
- **Celebratory Animations**: Discovering a new world is exciting, and our app celebrates every "CONFIRMED" prediction with a fun animation!

---

## ## How to Use

1.  **Open the App**: Navigate to the [live Streamlit application](https://exoplanet-detector-reduan.streamlit.app/).
2.  **Adjust Features**: Use the sidebar on the left to input the data for a celestial object of interest.
3.  **Analyze**: Click the **"Analyze Celestial Object"** button in the main panel.
4.  **View Results**: The app will display the AI's prediction, the confidence level, and a detailed feature importance plot.

---

## ## Technologies Used

- **Python**: The core programming language.
- **Pandas**: For data manipulation and cleaning.
- **Scikit-learn**: For building and training the machine learning model.
- **Streamlit**: For creating and deploying the interactive web app.
- **Matplotlib**: For generating the feature importance plot.
- **Joblib**: For saving and loading the trained model.
