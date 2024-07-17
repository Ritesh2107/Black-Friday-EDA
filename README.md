Black Friday Purchase Analysis
Project Overview
A retail company, "ABC Private Limited," aims to understand customer purchase behavior, specifically the purchase amount for various products across different categories. The provided dataset includes customer demographics (age, gender, marital status, city type, stay in current city), product details (product ID and product category), and the total purchase amount from the last month. The goal is to build a model to predict the purchase amount, enabling personalized offers for customers.

Dataset
The dataset contains the following columns:

User_ID: Unique identifier for each customer
Product_ID: Unique identifier for each product
Gender: Gender of the customer
Age: Age range of the customer
Occupation: Occupation code of the customer
City_Category: Category of the city (A, B, C)
Stay_In_Current_City_Years: Number of years the customer has stayed in the current city
Marital_Status: Marital status of the customer
Product_Category_1: Category of the product (1)
Product_Category_2: Category of the product (2)
Product_Category_3: Category of the product (3)
Purchase: Purchase amount (only present in the training dataset)
Data Preprocessing
Data Loading:

python
Copy code
import pandas as pd
df_train = pd.read_csv('Dataaset/train.csv')
df_test = pd.read_csv('Dataaset/test.csv')
Data Concatenation:

python
Copy code
df = pd.concat([df_train, df_test])
Handling Missing Values:

Dropped User_ID column.
Mapped gender values to numerical.
Converted age ranges to numerical categories.
One-hot encoded the City_Category.
Filled missing values in Product_Category_2 and Product_Category_3 with their respective modes.
Feature Engineering:

python
Copy code
df['Gender'] = df['Gender'].map({'F': 0, 'M': 1}).fillna(-1)
df['Age'] = df['Age'].map({'0-17': 1, '18-25': 2, '26-35': 3, '36-45': 4, '46-50': 5, '51-55': 6, '55+': 7}).fillna(-1)
df_city = pd.get_dummies(df['City_Category'], drop_first=True)
df = pd.concat([df, df_city], axis=1)
df.drop('City_Category', axis=1, inplace=True)
df['Product_Category_2'] = df['Product_Category_2'].fillna(df['Product_Category_2'].mode()[0])
df['Product_Category_3'] = df['Product_Category_3'].fillna(df['Product_Category_3'].mode()[0])
Model Building
Import Libraries:

python
Copy code
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import plotly.express as px
Initial Data Exploration:

python
Copy code
df.info()
df.describe()
Handling Categorical Features:

Gender, Age, and City Category were converted to numerical values.
Missing values were handled appropriately.
Conclusion
The processed data is now ready for model building to predict the purchase amount. The next steps involve selecting and training machine learning models, evaluating their performance, and optimizing them for better accuracy.

Repository Structure
Copy code
.
├── Dataaset
│   ├── train.csv
│   ├── test.csv
├── Black_friday.ipynb
├── README.md
Usage
To run the notebook and reproduce the results, ensure you have the required libraries installed and execute the notebook cells in sequence.
