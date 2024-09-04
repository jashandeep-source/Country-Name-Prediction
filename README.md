**Country Name Prediction Using Geospatial and Socioeconomic Factors**

## In this project, analysis and prediciton of the cities data in combination with unemployment data and international toursim data

**Libraries used:**
* Anaconda: https://www.anaconda.com

**Anaconda has the following:**
* Pandas
* numpy
* sklearn
* matplotlib

### Note : If you don't have any of the above say sklearn use pip install scikit-learn to install it

**Python Configuration and Modules used:**
* Python 3.11.5
* import pandas as pd
* import numpy as np
* import os
* import seaborn as sns
* import matplotlib.pyplot as plt
* from scipy import stats
* from sklearn.model_selection import train_test_split, GridSearchCV
* from sklearn.linear_model import LinearRegression, Ridge
* from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
* from sklearn.neighbors import KNeighborsRegressor
* from sklearn.svm import SVR
* from sklearn.metrics import mean_squared_error, r2_score
* from sklearn.preprocessing import StandardScaler
* from sklearn.ensemble import RandomForestClassifier
* from sklearn.ensemble import GradientBoostingClassifier
* from sklearn.pipeline import make_pipeline
* from sklearn.neighbors import KNeighborsClassifier
* from sklearn.metrics import classification_report

### Order of execution & Commands (and arguments):

### Data Cleaning and Exploration
* Run data_ETL.ipynb first
* Run intial_data_analysis.ipynb

### Data Analysis
#### For toursim: 
* Run tourism_analysis.ipynb 

#### For Unemployment: 
* Run unemployment_analysis.py as
***python3 unemployment_analysis.py***

### Data Prediction
#### For Country Name: 
* Run countryname_prediciton.ipynb

### Files produced/expected
### Data Cleaning
* Data/merged_data.csv

### Data Analysis
* Histogram_unemployment.png
* Population Vs Unemployment.pmg

### Data Prediction
#### For Country Name
* country_predictions/validation_results.csv





