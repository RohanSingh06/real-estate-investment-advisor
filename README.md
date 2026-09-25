\# Real Estate Investment Advisor



\## Predicting Property Profitability \& Future Value



A machine learning project that analyzes Indian real estate properties and predicts:



1\. Whether a property is a Good Investment

2\. The estimated property price after 5 years



\---



\## Project Overview



The project combines data analysis, feature engineering, classification, regression, MLflow experiment tracking, and Streamlit to build an interactive Real Estate Investment Advisor.



\### Domain



Real Estate / Investment / Financial Analytics



\---



\## Skills Used



\- Python

\- Pandas

\- NumPy

\- Data Analysis

\- Exploratory Data Analysis

\- Feature Engineering

\- Machine Learning

\- Classification

\- Regression

\- Feature Scaling

\- Model Evaluation

\- Streamlit

\- MLflow



\---



\## Dataset



Dataset:



`india\_housing\_prices.csv`



The dataset contains property information including:



\- State

\- City

\- Locality

\- Property Type

\- BHK

\- Size

\- Price

\- Price per SqFt

\- Year Built

\- Furnished Status

\- Floor Information

\- Nearby Schools

\- Nearby Hospitals

\- Public Transport

\- Parking

\- Security

\- Amenities

\- Facing

\- Owner Type

\- Availability Status



\---



\## Project Workflow



\### Phase 1 — Project Setup



Project structure and environment setup.



\### Phase 2 — Data Inspection



Dataset inspection, shape, columns, data types, missing values and duplicates.



\### Phase 3 — Data Cleaning



Data quality checks and cleaning.



\### Phase 4 — Feature Engineering



Created features including:



\- Price per SqFt

\- Amenity Density Score

\- Price per BHK

\- Size per BHK

\- Property Age Category

\- Floor Consistency



\### Phase 5 — Target Creation



Created:



\- `Good\_Investment`

\- `Future\_Price\_5Y`



\### Phase 6 — Exploratory Data Analysis



Completed the required real-estate EDA questions covering:



\- Price

\- Size

\- Location

\- Property type

\- BHK

\- Amenities

\- Ownership

\- Transport

\- Furnishing

\- Parking



\### Phase 7 — ML Preprocessing



Implemented:



\- Numerical imputation

\- Categorical imputation

\- Scaling

\- One-hot encoding

\- Train/test splitting



\### Phase 8 — Classification



Five classification models:



\- Logistic Regression

\- Decision Tree

\- Random Forest

\- Extra Trees

\- XGBoost



Target:



`Good\_Investment`



\### Phase 9 — Regression



Five regression models:



\- Linear Regression

\- Decision Tree Regressor

\- Random Forest Regressor

\- Extra Trees Regressor

\- XGBoost Regressor



Target:



`Future\_Price\_5Y`



\### Phase 10 — MLflow



Tracked:



\- Experiments

\- Parameters

\- Metrics

\- Model artifacts



\### Phase 11 — Streamlit



Developed an interactive application for:



\- Property input

\- Investment prediction

\- Investment confidence

\- 5-year price prediction

\- Property analysis



\---



\## Project Structure



```text

Real Estate Investment Advisor/

│

├── app/

├── data/

│   ├── raw/

│   └── processed/

├── models/

├── notebooks/

├── src/

├── requirements.txt

├── README.md

└── .gitignore

