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
"encoding.csv"
import pandas as pd
df=pd.read_csv("Encoding Data.csv")
df
 <img width="355" height="344" alt="image" src="https://github.com/user-attachments/assets/d658ac34-0f8e-47db-a933-fdd4b6fe6916" />
 from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
     <img width="418" height="220" alt="image" src="https://github.com/user-attachments/assets/8b820b00-9365-4b34-8beb-aef30ce59850" />
     df['bo2']=e1.fit_transform(df[["ord_2"]])
df
<img width="348" height="327" alt="image" src="https://github.com/user-attachments/assets/64098f7e-ef28-4c21-be4a-b505562bbf55" />
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
<img width="333" height="356" alt="image" src="https://github.com/user-attachments/assets/fc7feec6-bfda-4cd4-9e39-e2c9a2309558" />
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]])) # Orders in Alphabetical Order Blue , Green, Red
df2=pd.concat([df2,enc],axis=1)
df2
<img width="682" height="387" alt="image" src="https://github.com/user-attachments/assets/2fe210e2-b0b9-413e-8051-9c9e24e05e36" />
pd.get_dummies(df2,columns=["nom_0"])
<img width="600" height="315" alt="image" src="https://github.com/user-attachments/assets/676826ae-4198-4e8e-9ab6-230ba9e08676" />
"data.csv"
pip install --upgrade category_encoders
<img width="998" height="298" alt="image" src="https://github.com/user-attachments/assets/9f5886f8-8536-4b66-b213-f9affb6d3f1c" />
from category_encoders import BinaryEncoder
df=pd.read_csv("/content/data.csv")
df
<img width="502" height="350" alt="image" src="https://github.com/user-attachments/assets/4fefe66e-333f-4f80-83a3-8c2a02c35061" />
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb
<img width="607" height="357" alt="image" src="https://github.com/user-attachments/assets/8d36f4fd-2b4d-4588-86e3-b0ef16834c5e" />
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
<img width="501" height="385" alt="image" src="https://github.com/user-attachments/assets/4726e978-4134-4d68-99f7-109be25cc503" />
"data to transform"
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform.csv")
df
<img width="675" height="418" alt="image" src="https://github.com/user-attachments/assets/4ca631b4-3ccc-48c3-8660-1112b2ffb74b" />
df.skew()
<img width="272" height="188" alt="image" src="https://github.com/user-attachments/assets/973deff9-c355-4478-8916-c3a5f97db3aa" />
np.log(df["Highly Positive Skew"])
<img width="294" height="391" alt="image" src="https://github.com/user-attachments/assets/e4be351c-2c09-489c-858f-ece6b31fecf0" />
np.reciprocal(df["Moderate Positive Skew"])
<img width="396" height="391" alt="image" src="https://github.com/user-attachments/assets/dabe2430-d202-400f-9b48-ab263c547e5f" />
np.sqrt(df["Highly Positive Skew"])
<img width="320" height="393" alt="image" src="https://github.com/user-attachments/assets/1712eb0d-fbe0-4cec-a463-78bbbced883b" />
np.square(df["Highly Positive Skew"])
<img width="304" height="387" alt="image" src="https://github.com/user-attachments/assets/2fe850e7-f4de-4788-a1b3-325f2f4c2a11" />
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
<img width="865" height="375" alt="image" src="https://github.com/user-attachments/assets/0990515d-1ac9-4492-be59-821e18dc6d0a" />
df.skew()
<img width="307" height="215" alt="image" src="https://github.com/user-attachments/assets/69289086-43a8-4817-a809-8553c944448c" />
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
<img width="639" height="256" alt="image" src="https://github.com/user-attachments/assets/03eceb01-056c-44e1-918e-a6947e99b308" />
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
<img width="1217" height="408" alt="image" src="https://github.com/user-attachments/assets/3cf8a7fe-c855-4f8c-80a1-14128a03bc63" />
import seaborn as sns
import statsmodels.api as sm # STATS MODEL- STATISTICAL MODEL TO VISUALIZE DISTRIBUTION
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45') # QQ - QUANTILE QUANTILE PLOT
plt.show()
<img width="580" height="442" alt="image" src="https://github.com/user-attachments/assets/3fc3055f-c192-45d7-a8c2-0b9519cb25a5" />
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45') # RECIPROCAL
plt.show()
<img width="553" height="393" alt="image" src="https://github.com/user-attachments/assets/d2e48d41-d722-492c-83f4-d5fa35dd80fd" />
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
<img width="616" height="439" alt="image" src="https://github.com/user-attachments/assets/24a544e3-69f0-4e00-a7d1-1a969fbcdea9" />
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
<img width="553" height="410" alt="image" src="https://github.com/user-attachments/assets/f344c4ab-a77e-4f99-9412-9e7717461911" />









# RESULT:
       The Feature Encoding and Transformation process have been done and the python codes are executed successfully

       
