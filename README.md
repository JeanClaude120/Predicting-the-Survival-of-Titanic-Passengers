# Predicting the Survival Chance of Titanic Passengers


## Table of Contents

- [Project Overview](#project-overview)




### Project Overview

The sinking of the Royal Mail Ship Titanic in April 1912 is one of the world's most notorious ocean voyage accidents in history. This project aimed to develop a predictive model that estimates the likelihood of survival for passengers aboard the RMS Titanic, based on key personal and travel-related features. The model leverages machine learning techniques to analyse patterns and correlations between variables such as age, sex, passenger class, fare paid, and cabin location, among others.

### Data Source

The primary dataset used for this analysis is "titanic_dataset.csv", which contains detailed information about over 1500 passengers. The dataset is also publicly available on Kaggle.

### Tools

- Excel - Data preparation
- Python in Google Colab (pandas, matplotlib, sklearn, seaborn): data cleaning, analysis, visualisation and regression

### Methodology: Step by step

- Downloaded CSV file on Kaggle.com
- Open it in Excel to have a quick look at consistency in formatting
- Imported a CSV into Google Colab
- Exploratory Data Analysis (EDA)
  
  - Check Null values
      
``` Python
# data.isnull().sum()
```

  - Remove the columns we don't need for analysis
    
``` Python
columns_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked']
data = data.drop(columns=columns_to_drop)
data.head()
```

  - Replace missing ages with the median
    
``` Python
median_age = df['Age'].median()
data['Age'].fillna(median_age, inplace=True)
```

  - Encoding male/female into numbers
     
``` Python
data['Sex'] = data['Sex'].map({'female': 0, 'male': 1})
data.head()
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

- Visualize the confusion matrix using

``` Python
import seaborn as sns
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Dead', 'Survived'], yticklabels=['Dead', 'Survived'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
```

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





  
