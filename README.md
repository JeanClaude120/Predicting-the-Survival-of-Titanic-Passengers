# Predicting the Survival Chance of Titanic Passengers


## Table of Contents

- [Project Overview](#project-overview)




### Project Overview

This project aimed to develop a predictive model that estimates the likelihood of survival for passengers aboard the Royal Mail Ship Titanic, based on key personal and travel-related features. The model leverages machine learning techniques to analyse patterns and correlations between variables such as age, sex, passenger class, fare paid, and cabin location, among others.

### Data Source

The primary dataset used for this analysis is "titanic_dataset.csv", which contains detailed information about over 500 passengers. The dataset is also publicly available on Kaggle.

### Tools

- Excel - Data preparation
- Python in Google Colab (pandas, matplotlib, sklearn, seaborn): data cleaning, analysis, visualisation and regression

### Step by Step

- Importing a CSV into Google Colab
- Data exploration
Data preparation: Null/empty values, encoding strings into numbers, replacing missing ages with median, removing some columns, encoding male/female into numbers
- Train-test split
- Fitting the model
- Testing the model and calculating the accuracy and the confusion matrix

### Data Analysis

``` Python
# create train test split
from sklearn.model_selection import train_test_split

X = df.drop('Survived', axis=1)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)

Shape of X_train: (712, 6)
Shape of X_test: (179, 6)
Shape of y_train: (712,)
Shape of y_test: (179,)
```
```Python
# Fit a logistic regression
From sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Initialise and train the Logistic Regression model
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)
```
### Results

After preprocessing and splitting the data, the model was trained using Logistic Regression, which is a widely used classification algorithm for binary outcomes. Below are the key outcomes from the model evaluation:

- Training Data Size: 712 samples

- Testing Data Size: 179 samples

- Features Used: 5 (age, sex, fare, class, siblings, parents-on-board )

- Model Used: Logistic Regression

- Accuracy Score: ~ 0.8101






  
