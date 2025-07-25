# Titanic Survival Prediction Using Machine Learning

<div align="center">
  <img width="672" height="422" alt="Screenshot 2025-07-11 at 20 46 45" src="https://github.com/user-attachments/assets/3817c52e-ea7a-4fed-b20b-b5f0e1926ce2" />
</div>


## Table of Contents

- [Project Overview](#project-overview)
- [Data Source](#data-source)
- [Tools](#tools)
- [Step by step](#step-by-step)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Data Analysis](#data-analysis)
- [Results](#results)
- [Applications](#applications)
- [Recommendations](#recommendations)


### Project Overview

The sinking of the Royal Mail Ship Titanic in April 1912 is one of the world's most notorious ocean voyage accidents in history. This project aimed to develop a predictive model that estimates the likelihood of survival for passengers aboard the RMS Titanic, based on key personal and travel-related features. The model leverages machine learning techniques to analyse patterns and correlations between variables such as age, sex, passenger class, fare paid, and cabin location, among others.

### Data Source

The primary dataset used for this analysis is "titanic_dataset.csv", which contains detailed information about over 1500 passengers. The dataset is also publicly available on Kaggle.

### Tools

- Excel - Data preparation
- Python in Google Colab (pandas, matplotlib, sklearn, seaborn): data cleaning, analysis, visualisation and regression

### Step by step

- Downloaded CSV file on Kaggle.com
- Open it in Excel to have a quick look at consistency in formatting
- Imported a CSV into Google Colab

### Exploratory Data Analysis 
  
  - Check Null values
      
``` Python
# df.isnull().sum()
```

  - Remove the columns we don't need for analysis
    
``` Python
columns_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked']
df = df.drop(columns=columns_to_drop)
data.head()
```

  - Replace missing ages with the median
    
``` Python
median_age = df['Age'].median()
df['Age'].fillna(median_age, inplace=True)
```

  - Encoding male/female into numbers
     
``` Python
df['Sex'] = df['Sex'].map({'female': 0, 'male': 1})
df.head()
```
- Plot the total survived and dead from the dataset

``` Python
import matplotlib.pyplot as plt
# Calculate the counts of survived and dead
survived_counts = df['Survived'].value_counts()
# Create a bar plot
plt.figure(figsize=(6, 4))
survived_counts.plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Total Survived vs. Dead')
plt.xlabel('Survived')
plt.ylabel('Count')
plt.xticks([0, 1], ['Dead', 'Survived'], rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
```

<img width="629" height="401" alt="survived vs dead" src="https://github.com/user-attachments/assets/9b88b2d4-489f-4c25-a3be-197e77564d4a" />

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

```Python
# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the model: {accuracy:.4f}")
Accuracy of the model: 0.8101
```

- Confusion matrix
  
``` Python
import matplotlib.pyplot as plt
# Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred)
```
- Print the confusion matrix
  
``` Python
print("Confusion Matrix:")
print(cm)
```

- Visualize the confusion matrix

``` Python
import seaborn as sns
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Dead', 'Survived'], yticklabels=['Dead', 'Survived'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
```

<img width="543" height="460" alt="confusion matrix" src="https://github.com/user-attachments/assets/797fbaf0-c284-4b6c-a8b3-523a2a278770" />


- Description of confusion matrix:
  
   - True Negatives (TN):	The number of people who were actually Dead and the model predicted Dead.
   - False Positives (FP):	The number of people who were actually Dead but the model predicted Survived (Type I error).
   - False Negatives (FN):	The number of people who were actually Survived but the model predicted Dead (Type II error).
   - True Positives (TP):	The number of people who were actually Survived and the model predicted Survived.

``` Python
- model.coef_[0]
array([-0.92818783, -2.61059783, -0.03145174, -0.3158025 , -0.12520297,
        0.00343413])
```

- Plot the feature importance

``` Python
import pandas as pd
import matplotlib.pyplot as plt
# Get the coefficients (feature importances) from the logistic regression model
feature_importance = model.coef_[0]

# Create a pandas Series for easier plotting
feature_importance_series = pd.Series(feature_importance, index=X.columns)

# Sort the features by importance
sorted_feature_importance = feature_importance_series.sort_values(ascending=False)

# Plot the feature importance
plt.figure(figsize=(10, 6))
sorted_feature_importance.plot(kind='bar')
plt.title('Feature Importance from Logistic Regression')
plt.xlabel('Features')
plt.ylabel('Coefficient Value')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
```

<img width="1015" height="583" alt="feature importance" src="https://github.com/user-attachments/assets/6abed736-6cf8-4ca5-a45e-485e4359d3ba" />


### Results

After preprocessing and splitting the data, the model was trained using Logistic Regression, which is a widely used classification algorithm for binary outcomes. Below are the key outcomes from the model evaluation:

- Training Data Size: 712 samples

- Testing Data Size: 179 samples

- Features Used: 5 (age, sex, fare, class, siblings, parents-on-board )

- Model Used: Logistic Regression

- Accuracy Score: ~ 0.8101

### Applications

- The streamlined workflow can be adapted for similar binary classification problems in healthcare, finance, and customer retention.
- This Titanic survival prediction model is ideal for teaching the whole data science workflow from data wrangling to machine learning in a historical and well-documented context.
- This project helps identify which features, such as sex, class, and fare, are most influential in predicting survival, offering insights into real-world applications like risk modelling and classification systems.

### Recommendations

- We can try other classification models, such as Random Forest, Gradient Boosting, or Support Vector Machines, to compare performance and hopefully improve accuracy.
- We can include additional engineered features like family size, ticket prefixes, or cabin sections; these might carry hidden patterns affecting survival chances.
- We can implement k-fold cross-validation to ensure model robustness and reduce overfitting or underfitting issues.
- When presenting to a non-technical audience, we can use ROC curves, feature importance plots, and survival probability charts to enhance the interpretability of the model results.





  
