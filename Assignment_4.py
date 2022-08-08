import logreg as logreg
import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.mlab as mlab

df = pd.read_csv('C:/Users/i24253/Python_Learn 1/Assignment_4/heart_disease.csv')
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
print(df.head(10))

df.rename(columns={'Education level': 'Education'}, inplace=True)

# print(df.columns)

print(df.shape)


# 1.Histogram
def draw_histograms(dataframe, features, rows, cols):
    fig = plt.figure(figsize=(10, 15))
    for i, feature in enumerate(features):
        ax = fig.add_subplot(rows, cols, i + 1)
        dataframe[feature].hist(bins=15, ax=ax, facecolor='Green')
        ax.set_title(feature + "Distribution", color='blue')
    fig.tight_layout()
    plt.show()


draw_histograms(df, df.columns, 6, 3)

print(df.describe())

# 1.a
count = 0
for i in df.isnull().sum(axis=1):
    if i > 0:
        count = count + 1
print('\n Total number of rows with missing values is ', count)

# 1.b
print('\nMissing values in each columns:\n', df.isnull().sum())

print(df.shape)

df['Gender'].value_counts()

# 2.a
# selected = df['cigsPerDay'], [df['Smoker' == 'Yes']]


print(df['Age'].unique())

df['ageGroup'] = pd.cut(x=df['Age'], bins=[30, 39, 49, 59, 70],
                        labels=['30-39 years', '40-49 years', '50-59 years', '60-70 years'])
print(df.head())

Male_count = 0
Female_count = 0
temp = 0
for index, row in df.iterrows():
    # print(row['Gender'], row['cigsPerDay'])
    if row['cigsPerDay'] == 'NAn':
        if row['Gender'] == 'Male':
            Male_count = Male_count + 1
            temp = temp + 1

        if row['Gender'] == 'Female':
            Female_count = Female_count + 1
            temp = temp + 1
print(temp)

# 2.b
for index, row in df.iterrows():
    if row['Gender'] == 'Male':
        if row['cigsPerDay'] == 0.0:
            print("Male: ", row['ageGroup'])
    else:
        if row['cigsPerDay'] == 0.0:
            print("FeMale: ", row['ageGroup'])

# 2 C.What is the average “cigsPerDay” among the Male smokers, whose age is between 30 and 39 years old; 40 and 49
# years old; 50 and 59 years old? (Round to the nearest integer) 2c For 30 and 39 years old(Male)
c1 = 0
c2 = 0
c3 = 0
sum1c = 0
for index, row in df.iterrows():
    if row['Gender'] == 'Male':
        if row['ageGroup'] == '30-39 years':
            # print ("printing",row['cigsPerDay']+1)
            c1 = c1 + 1
            sum1c = sum1c + row['cigsPerDay']

# print(c1)

average0 = sum1c / c1
print("The average “cigsPerDay” among the Male smokers, whose age is:  "
      "between 30 and 39 years old", (round(average0, 2)))

# For 40 and 49 years old (Male);
sum2c = 0
for index, row in df.iterrows():
    if row['Gender'] == 'Male':
        if row['ageGroup'] == '40-49 years':
            # print ("printing",row['cigsPerDay']+1)
            c2 = c2 + 1
            sum2c = sum2c + row['cigsPerDay']

# print(c1)

average1 = sum2c / c2
print("The average “cigsPerDay” among the Male smokers, whose age is:  "
      "between 40 and 49 years old", (round(average1, 2)))

# For 50 and 59 years old (Male);
sum3c = 0
for index, row in df.iterrows():
    if row['Gender'] == 'Male':
        if row['ageGroup'] == '50-59 years':
            # print ("printing",row['cigsPerDay']+1)
            c3 = c2 + 1
            sum3c = sum3c + row['cigsPerDay']

# print(c1)

average2 = sum3c / c3
print("The average “cigsPerDay” among the Male smokers, whose age is:  "
      "between 40 and 49 years old", (round(average2, 2)))

# 2D. What is the average “cigsPerDay” among the Female smokers, whose age is between 30 and 39 years old; 40 and 49
# years old; 50 and 59 years old; 60 and 69 years old? (Round to the nearest integer)
D1 = 0
D2 = 0
D3 = 0
sum1w = 0
for index, row in df.iterrows():
    if row['Gender'] == 'Female':
        if row['ageGroup'] == '30-39 years':
            # print ("printing",row['cigsPerDay']+1)
            D1 = c1 + 1
            sum1w = sum1w + row['cigsPerDay']

average4 = sum1w / D1
print("The average “cigsPerDay” among the Female smokers, whose age is:  "
      "between 30 and 39 years old", (round(average4, 2)))

# For 40 and 49 years old (Male);
sum2w = 0
for index, row in df.iterrows():
    if row['Gender'] == 'Female':
        if row['ageGroup'] == '40-49 years':
            # print ("printing",row['cigsPerDay']+1)
            D2 = D2 + 1
            sum2w = sum2w + row['cigsPerDay']

average5 = sum2w / D2
print("The average “cigsPerDay” among the Female smokers, whose age is:  "
      "between 40 and 49 years old", (round(average5, 2)))

# For 50 and 59 years old (Male);
sum3w = 0
for index, row in df.iterrows():
    if row['Gender'] == 'Female':
        if row['ageGroup'] == '50-59 years':
            # print ("printing",row['cigsPerDay']+1)
            D3 = D2 + 1
            sum3w = sum3w + row['cigsPerDay']

# print(c1)

average2 = sum3w / D3
print("The average “cigsPerDay” among the Female smokers, whose age is:  "
      "between 40 and 49 years old", (round(average2, 2)))

# 3. A.	For each record that has a missing value in “glucose”, replace the missing values with the average “glucose”
# of the people in that specific gender and age range.
print(df.head(20))
# print(df['glucose'].mean())
df['glucose'].fillna(value=df['glucose'].mean(), inplace=True)
print('Updated Dataframe:')
print(df.head(20))

# B.	For each record that has a missing value in “totChol”, replace the missing values with the average “totChol”
# of the people in that specific gender and age range.
print(df.head(50))
# print(df['totChol'].mean())
df['totChol'].fillna(value=df['totChol'].mean(), inplace=True)

print("Sample of missing value replaced with avg", df.iloc[42][['totChol']].head(50))
print("Sample of missing value replaced with avg", df.iloc[154][['totChol']].head(200))
print('\nUpdated Dataframe or totChol with missing values:')
print((df.head(200)))

# C.Replace the missing values in “BPMeds” with 0.
print('Before Updated Dataframe:')
print((df.head(55)))
df['BPMeds'] = df['BPMeds'].fillna(0)
print('\nUpdated Dataframe for BPMeds with missing values:')
print(df.head(55))

# D.Replace the missing values in “BMI” with the average BMI.
print('Before Updated Dataframe:')
print((df.head(100)))
df['BMI'].fillna(value=df['BMI'].mean(), inplace=True)
print('\nUpdated Dataframe for BMI with missing values:')
print(df['BMI'].loc[[97, 294, 705, 1155, 1161]])

# E.Replace the missing values in “Education level” with the value “Unknown”.
print('Before Updated Dataframe:')
print((df.head(55)))
df['Education'] = df['Education'].fillna('Unknown')
print('\nUpdated Dataframe for column Education level:')
print(df['Education'].loc[[1604]])

# F.Replace the missing values in “heartRate” with the average heart rate value.
print('Before Updated Dataframe:')
print((df.head(100)))
df['heartRate'].fillna(value=df['heartRate'].mean(), inplace=True)
print('\nUpdated Dataframe for heartRate with missing values:')
print(df.loc[[689]])



# 4.
#One Hot Encoding
print("BEFORE ENCODING:\n", df.head())
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
dfle = df
dfle.Gender = le.fit_transform(dfle.Gender)
dfle.Education = le.fit_transform(dfle.Education)
dfle.Smoker = le.fit_transform(dfle.Smoker)

print("AFTER ENCODING:\n", df.head())


'''
#cleanup_nums = {"Gender": {"Male": 0, "Female": 1},
                "Education": {"higher education": 1, "high school": 2, "middle school": 3, "bachelor": 4, },
                "Smoker": {"Yes": 1, "No": 2}}

#df = df.replace(cleanup_nums)
#print("AFTER ENCODING:\n", df.head())
'''
# 5.

corrMatrix = dfle.corr()

print(round(corrMatrix, 2))
plt.figure(figsize=(14, 10))
sn.heatmap(round(corrMatrix, 2), cmap='Purples', annot=True, linecolor='Red', linewidths=1.0)
plt.show()

# 6.

df.dropna(axis=0, inplace=True)
X = df.iloc[:, 0:15]

#print(X)
y = df.iloc[:, 15:16]

#print(y)

print(X.head(5))

print(y.head(5))

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=58)
# X_train, X_test, Y_train, Y_test = train_test_split(predictors, target, test_size=0.30, random_state=58)

# Applying the ML model - Logistic Regression
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

# Training the data
logreg.fit(X_train, y_train)

# Testing the data
y_pred = logreg.predict(X_test)

# Predicting the score
score = logreg.score(X_test, y_test)
print("\nTest accuracy score is:", score)

print("\nclassification_report:\n",classification_report(y_test, y_pred))


print("\nconfusion_matrix",confusion_matrix(y_test, y_pred))
print("\nAccuracy score is ",accuracy_score(y_test, y_pred))


# 7.

print("\n\nlogistic Regression model w/ variables identified in Question 5 :\n")
X = df.drop(['Gender','Education','Smoker','cigsPerDay','BPMeds',
             'prevalentStroke','diabetes','totChol','ageGroup'],1)
#print(X)


y = df.iloc[:, 15:16]

#print("Predictor\n",y.head(5))

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21)
# X_train, X_test, Y_train, Y_test = train_test_split(predictors, target, test_size=0.30, random_state=58)

# Applying the ML model - Logistic Regression
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

# Training the data
logreg.fit(X_train, y_train)

# Testing the data
y_pred = logreg.predict(X_test)

# Predicting the score
score = logreg.score(X_test, y_test)
print("Test accuracy score is:", score)

print("\nclassification_report:\n",classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))





