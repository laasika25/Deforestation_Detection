# 🔥 Fire Incident Classification using Machine Learning
An end-to-end machine learning application designed to automatically detect and classify forest fire incidents using historical satellite data. The goal is to support faster decision-making and minimize environmental damage through intelligent fire monitoring.

## 📘 Project Summary
This project converts raw MODIS satellite data (2021–2023) into a smart, interactive ML-based classification system. A Streamlit interface is provided to allow users to upload data and visualize model predictions in real time.

### 🛠️ Tech Stack
Language: Python 3 

Libraries: pandas, numpy, matplotlib, seaborn, folium

ML Frameworks: scikit-learn, imbalanced-learn (SMOTE), xgboost, joblib

User Interface: Streamlit

Tools: Google Colab, Visual Studio Code


### 📊 Model Performance
Algorithm	Accuracy:-


Logistic Regression	84%

Decision Tree	89%

K-Nearest Neighbors	87%

Random Forest	96%

✅ Random Forest provided the best performance and is used in the deployed interface.



### Set up the Project Locally

```bash
# Clone the repository

git clone https://github.com/laasika25/Deforestation_Detection.git
cd Deforestation_Detection

# Create virtual environment
python -m venv env
env\Scripts\activate          # On Windows
# source env/bin/activate    # On Mac/Linux

# Install dependencies
pip install -r requirements.txt
pip install streamlit pandas numpy scikit-learn matplotlib seaborn joblib folium imbalanced-learn

# Launch the Streamlit interface
streamlit run app.py
```


## Streamlit Model:-
<img width="569" height="599" alt="image" src="https://github.com/user-attachments/assets/59d9cae6-09db-4160-8ed7-06f54a2af656" />

## 🔮 Future Enhancements

Integration with live satellite feeds for real-time alerts

Mobile/web notifications for forest authorities

Include weather and environmental factors for richer prediction
