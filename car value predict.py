#importing all libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
import seaborn as sns
sns.set()
#loading the data
raw_data=pd.read_csv("C:\\Users\\HP\\Downloads\\1.04.+Real-life+example.csv")
raw_data.head()
raw_data.describe(include='all')
#determining the variables of interest
data=raw_data.drop(['Model'],axis=1)
data.describe(include='all')
#dealing with missing values
data.isnull().sum()
data_no_mv = data.dropna(axis=0)
data_no_mv .describe(include='all')
# Exploring PDF's
sns.displot(data_no_mv['Price'])
# Dealing with outliers
q=data_no_mv['Price'].quantile(0.99)
data_1 = data_no_mv[data_no_mv['Price']<q]
data_1.describe(include='all')
sns.displot(data_1['Price'])
sns.displot(data_no_mv['Mileage'])
q= data_1['Mileage'].quantile(0.99)
data_2 = data_1[data_1['Mileage']<q]
sns.displot(data_2['Mileage'])
sns.displot(data_no_mv['EngineV'])
data_3 = data_2[data_2['EngineV']<6.5]
sns.displot(data_3['EngineV'])
sns.displot(data_no_mv['Year'])
q = data_3['Year'].quantile(0.01)
data_4 = data_3[data_3['Year']>q]
sns.displot(data_4['Year'])
data_cleaned = data_4.reset_index(drop=True)
data_cleaned.describe(include='all')
#checking the OLS assumption
f,(ax1,ax2,ax3)=plt.subplots(1,3,sharey=True,figsize=(15,3))
ax1.scatter(data_cleaned['Year'],data_cleaned['Price'])
ax1.set_title('Price and Year')
ax2.scatter(data_cleaned['EngineV'],data_cleaned['Price'])
ax2.set_title('Price and EngineV')
ax3.scatter(data_cleaned['Mileage'],data_cleaned['Price'])
ax3.set_title('Price and Mileage')
plt.show()
sns.displot(data_cleaned['Price'])
log_price=np.log(data_cleaned['Price'])
data_cleaned['Log_price'] = log_price
data_cleaned
f,(ax1,ax2,ax3)=plt.subplots(1,3,sharey=True,figsize=(15,3))
ax1.scatter(data_cleaned['Year'],data_cleaned['Log_price'])
ax1.set_title('Log_price and Year')
ax2.scatter(data_cleaned['EngineV'],data_cleaned['Log_price'])
ax2.set_title('Log_price and EngineV')
ax3.scatter(data_cleaned['Mileage'],data_cleaned['Log_price'])
ax3.set_title('Log_price and Mileage')
plt.show()
data_cleaned = data_cleaned.drop(['Price'],axis=1)
data_cleaned.columns.values
from statsmodels.stats.outliers_influence import variance_inflation_factor
variable=data_cleaned[['Mileage','Year','EngineV']]
vif=pd.DataFrame()
vif["VIF"]=[variance_inflation_factor(variable.values,i) for i in range(variable.shape[1])]
vif["features"]=variable.columns
vif
data_no_multicollinearty = data_cleaned.drop(['Year'],axis=1)
data_with_dummies = pd.get_dummies(data_no_multicollinearty,drop_first=True)
data_with_dummies.head()
#Rearrange a bit
data_with_dummies.columns.values
cols =['Mileage', 'EngineV', 'Log_price', 'Brand_BMW',
       'Brand_Mercedes-Benz', 'Brand_Mitsubishi', 'Brand_Renault',
       'Brand_Toyota', 'Brand_Volkswagen', 'Body_hatch', 'Body_other',
       'Body_sedan', 'Body_vagon', 'Body_van', 'Engine Type_Gas',
       'Engine Type_Other', 'Engine Type_Petrol', 'Registration_yes']
data_preproccessed = data_with_dummies[cols]
data_preproccessed.head()
#linear Regression modle
targets =data_preproccessed['Log_price']
inputs = data_preproccessed.drop(['Log_price'],axis=1)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(inputs)
inputs_scaled = scaler.transform(inputs)
# Train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(inputs_scaled,targets,test_size=0.2,random_state=365)
reg=LinearRegression()
reg.fit(x_train,y_train)
y_hat=reg.predict(x_train)
plt.scatter(y_train,y_hat)
plt.xlabel('Target(y_train)',size=18)
plt.ylabel('Prediction(y_hat)',size=18)
plt.xlim(6,13)
plt.ylim(6,13)
plt.show()
sns.displot(y_train-y_hat)
plt.title("Residual PDF",size = 18)
reg.score(x_train,y_train)
# finding weights and bias
reg.intercept_
reg.coef_
reg_summary = pd.DataFrame(inputs.columns.values,columns=['Features'])
reg_summary['Weights']=reg.coef_
reg_summary
data_cleaned['Brand'].unique()
# Testing
y_hat_test=reg.predict(x_test)
plt.scatter(y_test,y_hat_test,alpha=0.2)
plt.xlabel('Target(y_test)',size=18)
plt.ylabel('Prediction(y_hat_test)',size=18)
plt.xlim(6,13)
plt.ylim(6,13)
plt.show()
df_pf=pd.DataFrame(np.exp(y_hat_test),columns=['Prediction'])
df_pf.head()
df_pf['Target']=np.exp(y_test)
df_pf
y_test=y_test.reset_index(drop=True)
y_test.head()
df_pf['Target']=np.exp(y_test)
df_pf
df_pf['Residual'] = df_pf['Target'] - df_pf['Prediction']
df_pf['Difference%'] = np.absolute(df_pf['Residual']/df_pf['Target']*100)
df_pf
df_pf.describe()
df_pf.sort_values(by=['Difference%'])
pd.options.display.max_rows = 999
pd.set_option('display.float_format',lambda x: '%.2f'% x)
print(df_pf.sort_values(by = ['Difference%']))
