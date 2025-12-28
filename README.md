## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
import pandas as pd
df=pd.read_csv("Encoding Data.csv") df

<img width="1018" height="570" alt="image" src="https://github.com/user-attachments/assets/23fa25c6-db04-46ce-aa7d-db00cc9e64e1" />

from sklearn.preprocessing import LabelEncoder,OrdinalEncoder pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm]) e1.fit_transform(df[["ord_2"]])

<img width="740" height="352" alt="image" src="https://github.com/user-attachments/assets/3d49c02e-ba7d-446e-844f-2f5f79e5a9c3" />

df['bo2']=e1.fit_transform(df[["ord_2"]]) df

<img width="722" height="528" alt="image" src="https://github.com/user-attachments/assets/8c280406-1604-4603-a3f8-deff97846995" />

le=LabelEncoder() dfc=df.copy()

dfc['ord_2']=le.fit_transform(dfc['ord_2']) dfc

<img width="644" height="569" alt="image" src="https://github.com/user-attachments/assets/58c18f39-b840-469c-b258-9bfb062183ba" />

from sklearn.preprocessing import OneHotEncoder

ohe=OneHotEncoder(sparse_output=False)

df2=df.copy()

enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))

df2=pd.concat([df2,enc],axis=1)

df2
<img width="550" height="441" alt="image" src="https://github.com/user-attachments/assets/87c6dc84-3721-4995-84fc-68faa29648f4" />

pd.get_dummies(df2,columns=["nom_0"])

<img width="828" height="456" alt="image" src="https://github.com/user-attachments/assets/6e6e5758-6f9f-47e2-be21-49617d308a67" />
pip install --upgrade category_encoders

<img width="1382" height="428" alt="image" src="https://github.com/user-attachments/assets/81f08353-334a-4fd7-8925-0cf4ba14e024" />

from category_encoders import BinaryEncoder

df=pd.read_csv("/content/data.csv")

df

<img width="622" height="439" alt="image" src="https://github.com/user-attachments/assets/6374ec15-ef68-41da-9063-8462ab23088b" />

be=BinaryEncoder()

nd=be.fit_transform(df['Ord_2'])

df

<img width="620" height="444" alt="image" src="https://github.com/user-attachments/assets/9eb35abb-b4f2-49bb-92eb-d5d6c9f504fa" />

dfb=pd.concat([df,nd],axis=1)

dfb
<img width="877" height="441" alt="image" src="https://github.com/user-attachments/assets/e6c13fb4-e6fd-400f-aefa-e20417d130b1" />

from category_encoders import TargetEncoder

te=TargetEncoder()

CC=df.copy()

new=te.fit_transform(X=CC["City"],y=CC["Target"])

CC=pd.concat([CC,new],axis=1)

CC

<img width="709" height="445" alt="image" src="https://github.com/user-attachments/assets/d63c1f5c-46c4-4b6c-8def-e0dadb6a693e" />

import pandas as pd

from scipy import stats

import numpy as np

df=pd.read_csv("/content/Data_to_Transform.csv")

df
<img width="986" height="498" alt="image" src="https://github.com/user-attachments/assets/583868d0-b31d-4974-85ce-2a68cd5d3b02" />


df.skew()

<img width="434" height="243" alt="image" src="https://github.com/user-attachments/assets/cc5a4dc7-eb28-44a4-8e29-391c833116c6" />

np.log(df["Highly Positive Skew"])
/
<img width="470" height="557" alt="image" src="https://github.com/user-attachments/assets/f1239be4-e995-42cc-a847-15120f60efe0" />

np.reciprocal(df["Moderate Positive Skew"])

<img width="536" height="586" alt="image" src="https://github.com/user-attachments/assets/08860d7d-d05f-4286-970c-14d06a83ac3c" />

np.sqrt(df["Highly Positive Skew"])

<img width="597" height="577" alt="image" src="https://github.com/user-attachments/assets/780627ff-566b-4d08-b790-c98e2792c044" />

np.square(df["Highly Positive Skew"])

<img width="651" height="567" alt="image" src="https://github.com/user-attachments/assets/e726b354-a252-4978-a31c-fc24c57a6281" />

df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"]) df

<img width="1276" height="517" alt="image" src="https://github.com/user-attachments/assets/f7db355d-2d18-44a0-b4a8-725cfe40eb75" />

df.skew()

<img width="515" height="294" alt="image" src="https://github.com/user-attachments/assets/042e1e4c-e49d-4216-9c38-dde786204b9c" />



df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])

df.skew()

<img width="615" height="354" alt="image" src="https://github.com/user-attachments/assets/6fb5c6eb-259c-438e-91f8-dbda9f6cc2c9" />

from sklearn.preprocessing import QuantileTransformer


qt=QuantileTransformer(output_distribution='normal')

df["Moderate Negative Skew_1"]=qt.fit_transform

(df[["Moderate Negative Skew"]]) df




<img width="1343" height="556" alt="image" src="https://github.com/user-attachments/assets/a8cb5f84-cb09-457b-8c88-1f102116cac4" />

import seaborn as sns

import statsmodels.api as sm

import matplotlib.pyplot as plt

sm.qqplot(df["Moderate Negative Skew"],line='45')

plt.show()

<img width="910" height="561" alt="image" src="https://github.com/user-attachments/assets/77c3bfd1-96d1-482b-9fd9-75bc1f7eb48b" />

sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')

plt.show()

<img width="929" height="562" alt="image" src="https://github.com/user-attachments/assets/980bd11a-37cd-4a35-87ae-98703f5a6933" />

from sklearn.preprocessing import QuantileTransformer

qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])

sm.qqplot(df["Moderate Negative Skew"],line='45')

plt.show()

<img width="1062" height="557" alt="image" src="https://github.com/user-attachments/assets/7f0d3714-ffd0-4d44-942e-9c681bec5a97" />



df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])

sm.qqplot(df["Highly Negative Skew"],line='45')

plt.show()


<img width="828" height="546" alt="image" src="https://github.com/user-attachments/assets/bc79d62a-db68-41fb-b8af-09c1112009d4" />

df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])

sm.qqplot(df["Highly Negative Skew"],line='45')

plt.show()

<img width="828" height="546" alt="image" src="https://github.com/user-attachments/assets/b3e8cabb-7494-455f-b8c2-f5d6d56dc6f7" />

from sklearn.preprocessing import QuantileTransformer

qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

dt["Age_1"]=qt.fit_transform(dt[["Age"]])

sm.qqplot(dt['Age'],line='45')

plt.show()

<img width="1360" height="620" alt="image" src="https://github.com/user-attachments/assets/e25bdab7-05b8-46b8-8965-edf7f7b5ddef" />

sm.qqplot(df["Highly Negative Skew_1"],line='45')

plt.show()
<img width="1090" height="558" alt="image" src="https://github.com/user-attachments/assets/9091c07a-407d-4c98-ae78-232edd7b9324" />


<img width="831" height="559" alt="image" src="https://github.com/user-attachments/assets/fb931b7e-db31-4630-89f9-0b89fcb7fe18" />


# RESULT:
       The Feature Encoding and Transformation process have been done and the python codes are executed successfully

       
