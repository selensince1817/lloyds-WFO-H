print('\nImporting dependencies...\n\n')
from re import I
import joblib

import click

from sklearn import preprocessing

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from IPython.display import Image  



from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_curve, roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier, export_graphviz

 


import pandas as pd

print('\nPackages installed\n\n')









print('\nImporting model from .ipynb\n')
rfc = joblib.load('my_model.pkl')
print('\nImported\n')

print('\n---------------------------\nInitialise\n---------------------------\n')


age = int(input('\nEnter employee\'s age:\n'))
have_kids = 0

if click.confirm('\nDoes the employee have children?\n', default=True):
    have_kids = 1

gender = int(input('\nWhat is employee\' gender? \nEnter [0] for Female and [1] for Male\n'))
if(gender != 1 and gender != 0):
    print('Incorrect input, try again')
    gender = int(input('\nWhat is employee\' gender? \nEnter [0] for Female and [1] for Male\n'))


is_lives_near = 0
if click.confirm('\nDoes the employee live near the office location?\n', default=True):
    is_lives_near = 1
else:
    is_lives_near = 0



occupation_dict = {
     0: 'Occupation_Business',
     1: 'Occupation_Engineer',
     2: 'Occupation_HR',
     3: 'Occupation_Manager',
     4: 'Occupation_Manager',
     5: 'Occupation_Marketing',
     6: 'Occupation_Recruiter',
     7: 'Occupation_Tutor'
}

occupation = int(input('\nWhat is employee\'s occupation?\nChoose from the following:\n{}\n'.format(occupation_dict)))


df = pd.read_csv('df_template.csv')

occupation_name = occupation_dict[occupation]
print(occupation_name)



data = {
    'Age': age,
    'Gender': gender,
    'have_kids': have_kids,
    'is_office_near_home': is_lives_near,
     occupation_name: int(1)
}

print(data)

df = df.append(data, ignore_index=True)
df.fillna(0, inplace=True)

df.drop(df.columns[0:1], inplace=True, axis=1)
df.to_csv('to_predict.csv')


print('\n\n\n\n\n\n\n\n------------------------\n\n------------------------\n\n------------------------\n\n------------------------\n\n------------------------\n\n\n\n\n\n\n\n')

prediction = rfc.predict(df)
if(prediction[0] == 0):
    print('Employee will most likely enjoy working from office (WFO)')
else:
    print('Employee will most likely enjoy working remotely (WFH)')
 


