from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from pandas import Series
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os
%matplotlib inline

    
import warnings 
warnings.filterwarnings('ignore')

os.chdir(r'../..')

startdate_train = datetime(2021,1,4)
enddate_train = datetime(2023,1,1)

startdate_test= datetime(2023,1,2)
enddate_test = datetime(2023,3,5)

#Final Prediction Period
StartDate_Train_Final = datetime(2021,1,4)
EndDate_Train_Final = datetime(2022,1,1)

StartDate_Test_Final= datetime(2022,1,2)
enddate_test_Final = datetime(2023,3,5)

from statsmodels.tsa.stattools import adfuller

# Perform Dickey-Fuller test (ADF)

# Null Hypothesis      : Series is non-Stationary (Unit Root present)
# Alternate Hypothesis : Series is Stationary (Basically Difference Stationary)


def adf_test(timeseries):
    #Perform Dickey-Fuller test:
    print ('Results of Augmented Dickey-Fuller Test:')
    adftest = adfuller(timeseries) #AIC - Method to use when automatically determining the 
                                                    #lag length among the values 0, 1, …, maxlag.
    adfoutput = pd.Series(adftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    print(adfoutput)
    print("---------------------")
    print("---------------------")
    print('Null Hypothesis      : Series is non-Stationary (Unit Root present)')
    print('Alternate Hypothesis : Series is Stationary')
    print("---------------------")
    print("---------------------")
    if adfoutput[1] <= .05:
        print('p-value is less than .05, means reject Null Hypothesis')
        print('Results : Series is Stationary')
    else:
        print('p-value is greater than .05, means fail to reject Null Hypothesis')
        print('Results : Series is non-Stationary')
        
#     print(adfoutput[1])
    return adfoutput[1]
   
from statsmodels.tsa.stattools import kpss

# Perform Dickey-Fuller test (ADF)

# Null Hypothesis      : Series is Stationary (Basically Trend Stationary)
# Alternate Hypothesis : Series is non-Stationary (Unit Root present)

def kpss_test(timeseries):
    print ('Results of KPSS Test:')
    kpsstest = kpss(timeseries, regression='c')
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','Lags Used'])
    print(kpss_output)
    print("---------------------")
    print("---------------------")
    print('Null Hypothesis      : Series is Stationary (Basically Trend Stationary)')
    print('Alternate Hypothesis : Series is non-Stationary (Unit Root present)')
    print("---------------------")
    print("---------------------")
    if kpss_output[1] < .05:
        print('p-value is less than .05, means reject Null Hypothesis')
        print('Results : Series is non-Stationary')
    else:
        print('p-value is greater than .05, means fail to reject Null Hypothesis')
        print('Results : Series is Stationary (Basically Trend Stationary)')
        
    return kpss_output[1]


df_Input = pd.read_excel(r'input_data\processed_data\ca_accident.xlsx', sheet_name= 'data')

df = df_Input.iloc[0:, 2:]
df.tail()

### Import Daily file for calculating recent weekday trends

# # Import Daily Files
# df_Input_Daily = pd.read_excel(r'Data\Processed Data\Daily_CallVolume_By_Segments.xlsx', sheet_name= 'ISC PH')
# df_Input_Daily = df_Input_Daily[df_Input_Daily['Date'] > pd.to_datetime(enddate_test) - timedelta(weeks = 13)] 
# #13 weeks before today
# df_Input_Daily.Date.min(), df_Input_Daily.Date.max()
# df_Input_Daily = df_Input_Daily.rename(columns = {"Claims":"Claims"})


# # df_Input_Daily


### Divide data into Train & Test

train = df.loc[(df['Date'] >= startdate_train) & (df['Date'] <= enddate_train )]
train = train.set_index('Date')
test = df.loc[(df['Date'] >= startdate_test) & (df['Date'] <= enddate_test )]
test = test.set_index('Date')

train.tail()

train.dtypes

### Create date column dataframe for test period

# from statsmodels.tsa.seasonal import seasonal_decompose
# result = seasonal_decompose(train['Claims'], model ='additive')
# result.plot()
# plt.show()

train.dtypes

train.isnull().sum()

plt.figure(figsize=(20,4))
plt.grid()
sns.lineplot(x=train.index, y='Claims', data=train, markers='o')
# sns.lineplot(x=test.index, y='Claims', data=test, markers='o')
# plt.xticks(train.index, rotation=80)
plt.title('Weekly')
plt.show()

### Checking Stationarity

#### ADF Test
#### KPSS Test

adf_out_1 = adf_test(train['Claims'])
kpss_out_1 = kpss_test(train['Claims'])

d_indicator = 0
D_indicator = 0

if (adf_out_1 >0.05):
    adf_out_2 = adf_test(np.log(train['Claims']))
    d_indicator = 1
    if (adf_out_2 >0.05):
        train["Log Claims"] = np.log(train['Claims'])
        d_indicator = 2
        
if (kpss_out_1 <0.05):
    D_indicator = 1

train["Log Claims"] = np.log(train['Claims'])
train["dif Claims"] = train['Claims'].diff()
train["dif Log Claims"] = train['Log Claims'].diff()

#### Create value for p and q to be used

from statsmodels.tsa.stattools import acf, pacf

if (d_indicator == 0):
    y = train["Claims"]
if (d_indicator == 1):
    y = train["Log Claims"]
if (d_indicator == 2):
    y = train["dif Log Claims"][1:]

lag_pacf = pacf(y, nlags=int((len(y)/2)-1), method='ols')
lag_pacf_abs = np.abs(lag_pacf)

lag_acf = acf(y, nlags=len(y))
lag_acf_abs = np.abs(lag_acf)

min_threshold = -1.96/np.sqrt(len(y))
max_threshold = 1.96/np.sqrt(len(y))

print(min_threshold)
print(max_threshold)

### Finalizing p value for ARIMA and SARIMA

pacf_val = 0

for i in range(1, len(lag_pacf_abs)):
    if (lag_pacf_abs[i] < max_threshold):
        pacf_val = i-1
        break
    
print(pacf_val)

### Finalizing q value for ARIMA and SARIMA

acf_val = 0

for i in range(1, len(lag_acf_abs)):
    if (lag_acf_abs[i] < max_threshold):
        acf_val = i-1
        break
    
print(acf_val)

# plt.figure(figsize=(16, 7))
# plt.plot(lag_pacf,marker='+')
# plt.axhline(y=0,linestyle='--',color='gray')
# plt.axhline(y=-1.96/np.sqrt(len(y)),linestyle='--',color='gray')
# plt.axhline(y=1.96/np.sqrt(len(y)),linestyle='--',color='gray')
# plt.title('Autocorrelation Function')
# plt.xlabel('number of lags')
# plt.ylabel('correlation')
# plt.tight_layout()


# plt.figure(figsize=(16, 7))
# #Plot ACF: 
# plt.plot(lag_acf, marker="o")
# plt.axhline(y=0,linestyle='--',color='gray')
# plt.axhline(y=-1.96/np.sqrt(len(y)),linestyle='--',color='gray')
# plt.axhline(y=1.96/np.sqrt(len(y)),linestyle='--',color='gray')
# plt.title('Autocorrelation Function')
# plt.xlabel('number of lags')
# plt.ylabel('correlation')
# plt.tight_layout()



## ARIMA

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

def model_arima(data,order):   
    mod_arima = ARIMA(data,order=order,freq='W').fit()  
    print(order, 'AIC ', round(mod_arima.aic,2), '---','BIC ',round(mod_arima.bic,2) )

import itertools
order_list = []
AIC_list = []
# def ARIMA_Loop():
#     
if (d_indicator == 0):
    endog = train['Claims']
    d = [0]
if (d_indicator == 1):
    endog = train['Log Claims']
    d = [0]
if (d_indicator == 2):
    endog = train['Log Claims']
    d = [1]       

# p = range(max(pacf_val-1,0), pacf_val+2) #use in case of larger dataset
# q = range(max(acf_val-1,0), acf_val+2)

p = q = range(0,5)

pdq = list(itertools.product(p,d,q))

best_aic = np.inf
best_bic = np.inf
best_pdq = None
best_results = None

AIC = None

import warnings 
warnings.filterwarnings('ignore')

for order in pdq:
    try:         
        mod_arima = ARIMA(endog, order = order, freq='W').fit() 
        if mod_arima.aic < best_aic:
            best_aic = mod_arima.aic
            best_pdq = order
            print(order,'AIC ',np.round(best_aic,2))
    except:
        continue

# ARIMA_Loop_Output = ARIMA_Loop()

model_arima_final = ARIMA(endog,order=best_pdq,freq="W")
model_arima_final = model_arima_final.fit()
order_Arima_final = best_pdq

print(model_arima_final.summary())

# # endog = train['Log Claims']
# for order in ARIMA_Loop_Output['Order']:
#     try:
#         model_arima_final = ARIMA(endog,order=order,freq="W")
#         model_arima_final = model_arima_final.fit()
#         order_Arima_final = order
#         break
#     except:
#         print("Oops!  That was no valid number.  Try again...")        
# print(model_arima_final.summary())


matplotlib.rcParams['figure.figsize']=[5,3]
model_arima_final.resid.plot(kind='hist',logx=False)
plt.show()

## Prediction on Train

def Prediction_Train(model,trainingData):
    pred = model.predict(typ='levels')
    
    if (d_indicator == 0):
        pred = np.round(pred,2)

        
    else:
        pred = np.exp(np.round(pred,2))

    # Below code is to plot train Vs prediction
    import matplotlib
    matplotlib.rcParams['figure.figsize']=[18,3]

    plt.plot(trainingData)
    plt.plot(pred)
    plt.grid()
    
    actual = trainingData.reset_index(drop=True)
    pred = pred.reset_index(drop=True)
    
    MAPE = np.round(np.mean(((np.abs(actual-pred))/pred)*100), 2)
    print('Mean Absolute Percentage Error :', MAPE)
    return pred, MAPE

pred_arima_train = Prediction_Train(model_arima_final, train['Claims'])

## Prediction on Test

def Prediction_Test(model, testData):
    pred = model.predict(start=startdate_test, end=enddate_test, typ = 'levels')
    
    if (d_indicator == 0):
        pred = np.round(pred,2)

    else:
        pred = np.exp(np.round(pred,2))

    # Below code is to plot train Vs prediction
    import matplotlib
    matplotlib.rcParams['figure.figsize']=[18,3]
    
    plt.plot(testData)
    plt.plot(pred)
    plt.grid()
    
    actual = testData.reset_index(drop=True)
    pred = pred.reset_index(drop=True)
    
    MAPE = np.round(np.mean(((np.abs(actual-pred))/pred)*100), 2)
    print('Mean Absolute Percentage Error :', MAPE)
    return pred,MAPE

pred_arima = Prediction_Test(model_arima_final, test['Claims'])

## Holt's Winter
http://www.statsmodels.org/dev/generated/statsmodels.tsa.holtwinters.ExponentialSmoothing.html#statsmodels.tsa.holtwinters.ExponentialSmoothing

from statsmodels.tsa.holtwinters import ExponentialSmoothing

HoltsWinter = ExponentialSmoothing(np.asarray(train['Claims']), damped = True,seasonal_periods=52 ,trend='add', seasonal='add').fit()
# pred_holts = Prediction_Test(HoltsWinter, test['Claims'])


def Prediction_Train(model,testData):
    pred = model.predict(0)
    pred = np.round(pred,2)

    # Below code is to plot train Vs prediction
    import matplotlib
    matplotlib.rcParams['figure.figsize']=[18,3]
    
    plt.plot(np.asarray(train['Claims']))
    plt.plot(pred)
    plt.grid()
    
    actual = testData.reset_index(drop=True)
    pred = pred
#     print(actual)
#     print(pred)
    MAPE = np.round(np.mean(((np.abs(actual-pred))/pred)*100), 2)
    print('Mean Absolute Percentage Error :', MAPE)
    return pred, MAPE


pred_holts_train = Prediction_Train(HoltsWinter, train['Claims'])

def Prediction_Test(model, testData):
    pred = HoltsWinter.forecast(len(test))
    pred = np.round(pred,2)

    # Below code is to plot train Vs prediction
    import matplotlib
    matplotlib.rcParams['figure.figsize']=[18,3]
    
    plt.plot(np.asarray(test['Claims']))
    plt.plot(pred)
    plt.grid()
    
    actual = testData.reset_index(drop=True)
    pred = pred
    
    MAPE = np.round(np.mean(((np.abs(actual-pred))/pred)*100), 2)
    print('Mean Absolute Percentage Error :', MAPE)
    return pred,MAPE


# create a set of exponential smoothing configs to try

models = list()

# define config lists
d_params = [True, False]
p_params = [13, 26, 52]
t_params = ['add', 'mul']
s_params = ['add', 'mul']
r_params = [True, False]
# create config instances
for t in t_params:
    for d in d_params:
        for s in s_params:
            for p in p_params:
                    for r in r_params:
                        cfg = [d,p,t,s,r]
#                         print(cfg)
                        models.append(cfg)


best_pred_val = 100
testData = test['Claims']
warnings.filterwarnings('ignore')
for config in models:
    
    damped, period, trend, seasonal, remove_bias = config

    HoltsWinter_Model = ExponentialSmoothing(np.asarray(train['Claims']), 
                                   damped = damped ,seasonal_periods=period ,trend=trend, 
                                       seasonal=seasonal).fit(optimized=True, remove_bias=remove_bias)

    pred = HoltsWinter_Model.forecast(len(test))
    pred = np.round(pred,2)

    actual = testData.reset_index(drop=True)
    pred = pred

    MAPE_Val = np.round(np.mean(((np.abs(actual-pred))/pred)*100), 2)

    if (MAPE_Val < best_pred_val):
        print(config)
        best_pred_val = MAPE_Val
        best_config = config
        HoltsWinter = HoltsWinter_Model
        pred_holts = Prediction_Test(HoltsWinter, test['Claims'])



## Prediction on Test

def Prediction_Test(model, testData):
    pred = HoltsWinter.forecast(len(test))
    pred = np.round(pred,2)

    # Below code is to plot train Vs prediction
    import matplotlib
    matplotlib.rcParams['figure.figsize']=[18,3]
    
    plt.plot(np.asarray(test['Claims']))
    plt.plot(pred)
    plt.grid()
    
    actual = testData.reset_index(drop=True)
    pred = pred
    
    MAPE = np.round(np.mean(((np.abs(actual-pred))/pred)*100), 2)
    print('Mean Absolute Percentage Error :', MAPE)
    return pred,MAPE

pred_holts = Prediction_Test(HoltsWinter,test['Claims'])

## SARIMA

def model_sarimax(data,order,seasonalOrder):   
    mod_sarimax = SARIMAX(data,order=order, seasonal_order=seasonalOrder).fit()  
    print(order, seasonality,'AIC ', round(mod_sarimax.aic,2), '---','BIC ',round(mod_sarimax.bic,2) )

import itertools
order_list = []
AIC_list = []
order_list_PDQ = []
# def SARIMA_Loop():   
if (d_indicator == 0):
    endog = train['Claims']
    d = [0]
if (d_indicator == 1):
    endog = train['Claims']
    d = [1]
if (d_indicator == 2):
    endog = train['Log Claims']
    d = [1]  

if (D_indicator == 0):
    D = [0]
if (D_indicator == 1):
    D = [1]

# p = range(max(pacf_val-1,0), pacf_val+2)
# q = range(max(acf_val-1,0), acf_val+2)
p = q = range(0,5)
P=Q=range(0,2)
S=[52]

pdq = list(itertools.product(p,d,q))
PDQ = list(itertools.product(P,D,Q,S))

best_aic = np.inf
best_bic = np.inf
best_pdq = None
best_PDQ = None
best_results = None

import warnings 
warnings.filterwarnings('ignore')

for order in pdq:
        for seasonality in PDQ:
            try:         
                mod_sarima = SARIMAX(endog = endog, order=order, seasonal_order=seasonality).fit() 
                if mod_sarima.aic < best_aic:
                    best_aic = mod_sarima.aic
                    best_bic = mod_sarima.bic
                    best_pdq = order
                    best_PDQ = seasonality
                    print('order=',order,', seasonal_order=',seasonality,'AIC ',np.round(best_aic,2),'BIC ',np.round(best_bic,2))
            except:
                continue


model_sarima_final = SARIMAX(endog,order=best_pdq, seasonal_order= best_PDQ, 
                              enforce_stationarity=True, enforce_invertibility=True).fit() 

# model_sarima_final = SARIMAX(endog,order=row['Order'], seasonal_order=row['Order_PDQ'], 
#                               enforce_stationarity=True, enforce_invertibility=True).fit() 
order_Sarima_final = best_pdq
Seasonal_order_Sarima_final = best_PDQ


# for index, row in SARIMA_Loop_Output.iterrows():
#     try:
#         model_sarima_final = SARIMAX(endog,order=row['Order'], seasonal_order=row['Order_PDQ'], 
#                               enforce_stationarity=True, enforce_invertibility=True).fit() 
#         order_Sarima_final = row['Order']
#         Seasonal_order_Sarima_final = row['Order_PDQ']
        
#         break
#     except:
#         print("Oops!  That was no valid number.  Try again...")   
# print(model_sarima_final.summary())

# model_sarima_final.plot_diagnostics(figsize=(16, 8))
# plt.show()

## Prediction on Train

def Prediction_Train(model,trainingData):
    pred = model.predict(typ='levels')
    
    if (d_indicator == 2):
        pred = np.exp(np.round(pred,2))
    else:
        pred = np.round(pred,2)

    # Below code is to plot train Vs prediction
    import matplotlib
    matplotlib.rcParams['figure.figsize']=[18,3]

    plt.plot(trainingData)
    plt.plot(pred)
    plt.grid()
    
    actual = trainingData.reset_index(drop=True)
    pred = pred.reset_index(drop=True)
    
    MAPE = np.round(np.mean(((np.abs(actual-pred))/actual)*100), 2)
    print('Mean Absolute Percentage Error :', MAPE)
    return pred, MAPE

pred_sarima_train = Prediction_Train(model_sarima_final, train['Claims'])
#D = 1 - 16.41
#D = 0 - 13.7

## Prediction on Test

def Prediction_Test(model, testData):
    pred = model.predict(start=startdate_test, end=enddate_test, typ = 'levels')

    if (d_indicator == 2):
        pred = np.exp(np.round(pred,2))
    else:
        pred = np.round(pred,2)

    # Below code is to plot train Vs prediction
    import matplotlib
    matplotlib.rcParams['figure.figsize']=[18,3]
    
    plt.plot(testData)
    plt.plot(pred)
    plt.grid()
    
    actual = testData.reset_index(drop=True)
    pred = pred.reset_index(drop=True)
    
    MAPE = np.round(np.mean(((np.abs(actual-pred))/actual)*100), 2)
    print('Mean Absolute Percentage Error :', MAPE)
    return pred,MAPE

pred_sarima = Prediction_Test(model_sarima_final, test['Claims'])

## ARIMAX / SARIMAX

train_policies = pd.read_excel(r'Data\Processed Data\Policy_Weekly.xlsx', sheet_name='Train')
train_policies = train_policies[["Date","Average"]]
train_policies.columns = ["Week","#OpenedPolicies"]

test_policies = pd.read_excel(r'Data\Processed Data\Policy_Weekly.xlsx', sheet_name='Test')
test_policies = test_policies[["Date","Average"]]
test_policies.columns = ["Week","#OpenedPolicies"]

policies = train_policies.append(test_policies)

train_policies = policies.loc[(policies['Week'] >= startdate_train) & (policies['Week'] <= enddate_train )][["Week","#OpenedPolicies"]]
train_policies = train_policies.set_index('Week')



test_policies = policies.loc[(policies['Week'] >= startdate_test) & (policies['Week'] <= enddate_test)][["Week","#OpenedPolicies"]]
test_policies = test_policies.set_index('Week')

test_policies.head()

def model_sarimax(data, exog, order,seasonalOrder):   
    mod_sarimax = SARIMAX(data, exog, order=order, seasonal_order=seasonalOrder).fit()  
    print(order, seasonality,'AIC ', round(mod_sarimax.aic,2), '---','BIC ',round(mod_sarimax.bic,2) )

import itertools
order_list = []
AIC_list = []
order_list_PDQ = []
# def SARIMAX_Loop(Exog):   
if (d_indicator == 0):
    endog = train['Claims']
    d = [0]
if (d_indicator == 1):
    endog = train['Claims']
    d = [1]
if (d_indicator == 2):
    endog = train['Log Claims']
    d = [1] 

if (D_indicator == 0):
    D = [0]
if (D_indicator == 1):
    D = [1]

# p = range(max(pacf_val-1,0), pacf_val+2)
# q = range(max(acf_val-1,0), acf_val+2)
p=q=range(0,5)
P=Q=range(0,2)
S=[52]

pdq = list(itertools.product(p,d,q))
PDQ = list(itertools.product(P,D,Q,S))

best_aic = np.inf
best_bic = np.inf
best_pdq = None
best_PDQ = None
best_results = None

exog = train_policies['#OpenedPolicies']

import warnings 
warnings.filterwarnings('ignore')

            
for order in pdq:
        for seasonality in PDQ:
            try:         
                mod_sarima = SARIMAX(endog = endog, exog = exog, order=order, seasonal_order=seasonality).fit() 
                if mod_sarima.aic < best_aic:
                    best_aic = mod_sarima.aic
                    best_bic = mod_sarima.bic
                    best_pdq = order
                    best_PDQ = seasonality
                    print('order=',order,', seasonal_order=',seasonality,'AIC ',np.round(best_aic,2),'BIC ',np.round(best_bic,2))
            except:
                continue


exog = train_policies['#OpenedPolicies']

model_sarimax_final = SARIMAX(endog,exog,order=best_pdq, seasonal_order=best_PDQ, 
                              enforce_stationarity=True, enforce_invertibility=True).fit() 
order_Sarimax_final = best_pdq
Seasonal_order_Sarimax_final = best_PDQ
# SARIMAX_Loop_Output = SARIMAX_Loop(exog)

## Prediction on Train

def Prediction_Train(model,trainingData):
    pred = model.predict(typ='levels')
    
    if (d_indicator == 2):
        pred = np.exp(np.round(pred,2))
    else:
        pred = np.round(pred,2)

    # Below code is to plot train Vs prediction
    import matplotlib
    matplotlib.rcParams['figure.figsize']=[18,3]

    plt.plot(trainingData)
    plt.plot(pred)
    plt.grid()
    
    actual = trainingData.reset_index(drop=True)
    pred = pred.reset_index(drop=True)
    
    MAPE = np.round(np.mean(((np.abs(actual-pred))/actual)*100), 2)
    print('Mean Absolute Percentage Error :', MAPE)
    return pred, MAPE

pred_sarimax_train = Prediction_Train(model_sarimax_final, train['Claims'])
## D = 1 policy 12.82
## D = 0 policy 10.15


## Prediction on Test

def Prediction_Test(model,testData):
    pred = model.predict(start= startdate_test, end= enddate_test, exog=test_policies)
    
    if (d_indicator == 2):
        pred = np.exp(np.round(pred,2))
    else:
        pred = np.round(pred,2)

    # Below code is to plot train Vs prediction
    import matplotlib
    matplotlib.rcParams['figure.figsize']=[18,3]
    
    plt.plot(testData)
    plt.plot(pred)
    plt.grid()
    
    actual = testData.reset_index(drop=True)
    pred = pred.reset_index(drop=True)
    
    MAPE = np.round(np.mean(((np.abs(actual-pred))/actual)*100), 2)
    print('Mean Absolute Percentage Error :', MAPE)
    return pred,MAPE

pred_sarimax = Prediction_Test(model_sarimax_final, test['Claims'])

## Ensemble

Final_Train = {'Train':train['Claims'].reset_index(drop=True), 
             'ARIMA':pd.Series(pred_arima_train)[0], 
               'SARIMA':pd.Series(pred_sarima_train)[0], 
             'SARIMAX':pd.Series(pred_sarimax_train)[0], 
             'Holts':pd.Series(pred_holts_train)[0]}
ensemble_train = pd.DataFrame(Final_Train)
# ensemble_Final_train['Average'] = np.round(ensemble_Final_train.loc[0:,ensemble_Final_train.columns!='Train'].mean(axis=1),0)


ensemble_train.tail(3)

Final_Test = {'Test':test['Claims'].reset_index(drop=True), 
             'ARIMA':pd.Series(pred_arima)[0][0:len(test)], 
              'SARIMA':pd.Series(pred_sarima)[0][0:len(test)], 
             'SARIMAX':pd.Series(pred_sarimax)[0][0:len(test)], 
             'Holts':pd.Series(pred_holts)[0][0:len(test)]
             }
ensemble_test = pd.DataFrame(Final_Test)
# ensemble_Final_test['Average'] = np.round(ensemble_Final_test.loc[0:,ensemble_Final_test.columns!='Test'].mean(axis=1),0)

ensemble_test.head(50)

## Building Model on ensembling_train - Using Xgboost

#### Xgboost - Hyperparameter Optimization - RandomizedSearchCV

X_train = ensemble_train.iloc[1:,1:8]
y_train =ensemble_train.iloc[1:,0:1]
X_test = ensemble_test.iloc[0:,1:8]
y_test =ensemble_train.iloc[0:,0:1]

from sklearn.model_selection import RandomizedSearchCV
import xgboost as xg

xgb_randomCV = xg.XGBRegressor()

param_randomCV = {
 "learning_rate"    : np.array(list(range(1,60,1)))/100,
 "max_depth"        : np.array(list(range(3,20,1))),
 "min_child_weight" : np.array(list(range(3,20,1))),
 "gamma"            : np.array(list(range(10,60,10)))/100,
 "colsample_bytree" : np.array(list(range(10,60,5)))/100 , 
 "n_estimators"     : np.array(list(range(4,200,2))), 
 "num_parallel_tree" : [1,2] 
}
   
                 
clf_randomCV = RandomizedSearchCV(estimator=xgb_randomCV, param_distributions=param_randomCV, cv=5, scoring='neg_mean_absolute_error', random_state=10)
best_clf_randomCV = clf_randomCV.fit(X_train, y_train)

print(best_clf_randomCV.best_estimator_)
print('-----------------------------------------------------------------')
print(best_clf_randomCV.best_params_)
print('-----------------------------------------------------------------')
print(best_clf_randomCV.best_score_)

import xgboost as xg

def XGB_Func(learning_rate,max_depth,min_child_weight, n_estimators, num_parallel_tree,colsample_bytree, gamma):
    xgb = xg.XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                 colsample_bynode=1, colsample_bytree=colsample_bytree, gamma=gamma, gpu_id=-1,
                 importance_type='gain', interaction_constraints='',
                 learning_rate=learning_rate, max_delta_step=0, max_depth=max_depth,
                 min_child_weight=min_child_weight, monotone_constraints='()',
                 n_estimators=n_estimators, n_jobs=2, num_parallel_tree=num_parallel_tree,
                 objective='reg:squarederror', random_state=0, reg_alpha=0,
                 reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method='exact',
                 validate_parameters=1, verbosity=None) 
    return xgb.fit(X_train, y_train) 

## Prediction on Train

def Prediction_Train(model, trainingData, y_train):
    pred = model.predict(trainingData)
    pred = np.round(pred,2)

    # Below code is to plot train Vs prediction
    import matplotlib
    matplotlib.rcParams['figure.figsize']=[18,3]

    plt.plot(y_train)
    plt.plot(pred)
    plt.grid()
    
    actual = y_train
    pred = pred
    
    MAPE = np.round(np.mean(((np.abs(actual-pred))/actual)*100), 2)
    print('Mean Absolute Percentage Error :', MAPE)
    return pred, MAPE



## Prediction on Test

def Prediction_Test(model, testData, y_test):
    pred = model.predict(testData)
    pred = np.round(pred,2)

    # Below code is to plot train Vs prediction
    import matplotlib
    matplotlib.rcParams['figure.figsize']=[18,3]
    
    plt.plot(y_test)
    plt.plot(pred)
    plt.grid()
    
    actual = y_test
    pred = pred
    
    MAPE = np.round(np.mean(((np.abs(actual-pred))/actual)*100), 2)
    print('Mean Absolute Percentage Error :', MAPE)
    return pred,MAPE


gamma = best_clf_randomCV.best_params_['gamma']
colsample_bytree = best_clf_randomCV.best_params_['colsample_bytree']
learning_rate = best_clf_randomCV.best_params_['learning_rate']
max_depth = best_clf_randomCV.best_params_['max_depth']
min_child_weight = best_clf_randomCV.best_params_['min_child_weight']
n_estimators = best_clf_randomCV.best_params_['n_estimators']
num_parallel_tree = best_clf_randomCV.best_params_['num_parallel_tree']

xgb = XGB_Func(learning_rate,max_depth,min_child_weight, n_estimators, num_parallel_tree, colsample_bytree, gamma)

pred_xgboost_train = Prediction_Train(xgb, X_train, ensemble_train['Train'][1:])
pred_xgboost = Prediction_Test(xgb, X_test, ensemble_test['Test'])


col_Len = len(ensemble_test.columns)

## Define pre existing Models
##### NOTE: order of columns in Single_perf_Name should be same as in ensemble test
Single_perf_Name = list(['ARIMA', 'Holts','SARIMA', 'SARIMAX', 'Xgboost'])
Single_perf_Val = list([pred_arima[1], pred_holts[1] ,pred_sarima[1], pred_sarimax[1], pred_xgboost[1]])

## Initialize MAPE
min_mape = 100

#############################Loop for calculating MAPE with each combination

## Loop 1 to select each of the model type as starting value
for outer in range(1,col_Len+1):
    
    ## Loop 2 to select other models in combination with above selected models
    for i in range(outer,col_Len+1):
        
        ## Initialize base model dataset and their corresponding names
        base_df = ensemble_test.iloc[0:,outer:i+1]
        mape_dict_test = Single_perf_Name[outer-1:i]
        
        ## Loop 3 to select combinations of other model's dataset iteratively with above base model
        for j in range(i+1,col_Len):
            inner_df = ensemble_test.iloc[0:,j:j+1]
            
            ## Test dataset for iteration
            concat_df = pd.concat([base_df, inner_df], axis =1)
            
            ## Name and average of test dataset column for iteration
            list_Name = Single_perf_Name[j-1:j]
            concat_df['Average'] = np.round(concat_df.loc[0:].mean(axis=1),0)
            
            #MAPE
            mape_average = np.round(np.mean(((np.abs(ensemble_test['Test']-concat_df['Average']))/ensemble_test['Test'])*100), 2)
            
            # Check if MAPE is minimum
            if( mape_average < min_mape) :
                min_mape = mape_average
                combo  = mape_dict_test + list_Name

## Print combination with minimum MAPE value               
print("Min MAPE for Average of", combo, " = ", min_mape)

Single_perf_Name.append(combo)
Single_perf_Val.append(min_mape)

print(Single_perf_Name)
print(Single_perf_Val)

min_model_loop_final = pd.DataFrame()
min_model_loop_final["name"] = Single_perf_Name
min_model_loop_final["val"] = Single_perf_Val

loop_check = min_model_loop_final[min_model_loop_final["val"] == min(min_model_loop_final["val"])]["name"]
loop_check_ind = min_model_loop_final[min_model_loop_final["val"] == min(min_model_loop_final["val"])]["name"].index
print(loop_check)
if (loop_check_ind == len(Single_perf_Name)-1):
    min_model_loop = list(loop_check)[0]
else:
    min_model_loop = loop_check
    
for i in min_model_loop:
    print(i)

## Final Result

train = df.loc[(df['Date'] >= StartDate_Train_Final) & (df['Date'] <= EndDate_Train_Final )]
train = train.set_index('Date')
train["Log Claims"] = np.log(train['Claims'])

train_policies = pd.read_excel(r'Data\Processed Data\Policy_Weekly.xlsx', sheet_name='Train')
train_policies = train_policies[["Date","Average"]]
train_policies.columns = ["Week","#OpenedPolicies"]

test_policies = pd.read_excel(r'Data\Processed Data\Policy_Weekly.xlsx', sheet_name='Test')
test_policies = test_policies[["Date","Average"]]
test_policies.columns = ["Week","#OpenedPolicies"]

policies = train_policies.append(test_policies)

train_policies = policies.loc[(policies['Week'] >= StartDate_Train_Final) & (policies['Week'] <= EndDate_Train_Final )][["Week","#OpenedPolicies"]]
train_policies = train_policies.set_index('Week')

test_policies = policies.loc[(policies['Week'] >= StartDate_Test_Final) & (policies['Week'] <= enddate_test_Final)][["Week","#OpenedPolicies"]]
test_policies = test_policies.set_index('Week')

test_policies.head()

rng = pd.date_range(StartDate_Test_Final, periods=50, freq='W')
test = pd.DataFrame(rng)
test.columns = ['Date']
test['Index'] = range(1, 1+(len(test)))
test['Index'] = test['Index'].astype(float)
# test = pd.DataFrame({ 'Date': rng, 'Val': range(0, len(df))})

test = test.loc[(test['Date'] >= StartDate_Test_Final) & (test['Date'] <= enddate_test_Final)]
test = test.set_index('Date')

test.tail()

def Arima_final():
    if (d_indicator == 0):
        endog = train['Claims']
    if (d_indicator == 1):
        endog = train['Claims']
    if (d_indicator == 2):
        endog = train['Log Claims']
    model_arima_final = ARIMA(endog,order=order_Arima_final,freq="W")
    model_arima_final = model_arima_final.fit()
    
    pred_train = model_arima_final.predict(typ='levels')
    pred = model_arima_final.predict(start= StartDate_Test_Final, end = enddate_test_Final, typ = 'levels')
    if (d_indicator == 2):
        pred_train = np.exp(np.round(pred_train,2))
        pred = np.exp(np.round(pred,2))
    else:
        pred_train = np.round(pred_train,2)
        pred = np.round(pred,2)
        
    return pred_train, pred

def Holts_final():
    damped, period, trend, seasonal,  remove_bias = best_config

    HoltsWinter_Model = ExponentialSmoothing(np.asarray(train['Claims']), 
                                   damped = damped ,seasonal_periods=period ,trend=trend, 
                                       seasonal=seasonal).fit(optimized=True, remove_bias=remove_bias)

    pred_train = HoltsWinter_Model.predict(0)
    pred_train = np.round(pred_train,2)
    pred = HoltsWinter_Model.forecast(len(test))
    pred = np.round(pred,2)
    
    return pred_train, pred

def Sarima_final():
    if (d_indicator == 0):
        endog = train['Claims']
    if (d_indicator == 1):
        endog = train['Claims']
    if (d_indicator == 2):
        endog = train['Log Claims']
    model_sarima_final = SARIMAX(endog,order = order_Sarima_final, seasonal_order = Seasonal_order_Sarima_final, 
                              enforce_stationarity=True, enforce_invertibility=True).fit() 
    
    pred_train = model_sarima_final.predict(typ='levels')
    pred = model_sarima_final.predict(start= StartDate_Test_Final, end = enddate_test_Final, typ = 'levels')
    
    if (d_indicator == 2):
        pred_train = np.exp(np.round(pred_train,2))
        pred = np.exp(np.round(pred,2))
    else:
        pred_train = np.round(pred_train,2)
        pred = np.round(pred,2)
        
    return pred_train, pred

def Sarimax_final():
    exog = train_policies['#OpenedPolicies']
    if (d_indicator == 0):
        endog = train['Claims']
    if (d_indicator == 1):
        endog = train['Claims']
    if (d_indicator == 2):
        endog = train['Log Claims']
    model_sarimax_final = SARIMAX(endog,exog,order = order_Sarimax_final, seasonal_order = Seasonal_order_Sarimax_final, 
                              enforce_stationarity=True, enforce_invertibility=True).fit() 
      
    pred_train = model_sarimax_final.predict(typ='levels')
    pred = model_sarimax_final.predict(start= StartDate_Test_Final, end = enddate_test_Final, exog=test_policies,typ='levels')
    
    if (d_indicator == 2):
        pred_train = np.exp(np.round(pred_train,2))
        pred = np.exp(np.round(pred,2))
    else:
        pred_train = np.round(pred_train,2)
        pred = np.round(pred,2)
        
    return pred_train, pred

def Xgboost_final(X_train, y_train, X_test):
    xgb_randomCV = xg.XGBRegressor()

    clf_randomCV = RandomizedSearchCV(estimator=xgb_randomCV, param_distributions=param_randomCV, cv=5, scoring='neg_mean_absolute_error', random_state=10)
    best_clf_randomCV = clf_randomCV.fit(X_train, y_train)


    gamma = best_clf_randomCV.best_params_['gamma']
    colsample_bytree = best_clf_randomCV.best_params_['colsample_bytree']
    learning_rate = best_clf_randomCV.best_params_['learning_rate']
    max_depth = best_clf_randomCV.best_params_['max_depth']
    min_child_weight = best_clf_randomCV.best_params_['min_child_weight']
    n_estimators = best_clf_randomCV.best_params_['n_estimators']
    num_parallel_tree = best_clf_randomCV.best_params_['num_parallel_tree']

    xgb = XGB_Func(learning_rate,max_depth,min_child_weight, n_estimators, num_parallel_tree, colsample_bytree, gamma)
    
    pred_train = xgb.predict(X_train)
    pred_train = np.round(pred_train,2)

    pred = xgb.predict(X_test)
    pred = np.round(pred,2)
    
    return pred_train, pred

# min_model_loop = ['Holts']

pred_arima_train_F = []
pred_holts_train_F = []
pred_sarima_train_F = []
pred_sarimax_train_F = []
pred_xgboost_train_F = []

pred_arima_F = []
pred_holts_F = []
pred_sarima_F = []
pred_sarimax_F = []
pred_xgboost_F = []
pred_arima_train_FX = []
pred_arima_FX = []
pred_holts_train_FX = []
pred_holts_FX = []
pred_sarima_train_FX = []
pred_sarima_FX = []
pred_sarimax_train_FX = []
pred_sarimax_FX = []

for i in min_model_loop:
    if(i == 'ARIMA'):
        output_model = Arima_final()
        pred_arima_train_F = output_model[0]
        pred_arima_F = output_model[1]

    if(i == 'Holts'):
        output_model = Holts_final()
        pred_holts_train_F = output_model[0]
        pred_holts_F = output_model[1]

    if(i == 'SARIMA'):
        output_model = Sarima_final()
        pred_sarima_train_F = output_model[0]
        pred_sarima_F = output_model[1]

    if(i == 'SARIMAX'):
        output_model = Sarimax_final()
        pred_sarimax_train_F = output_model[0]
        pred_sarimax_F = output_model[1]

    if(i == 'Xgboost'):
        output_model = Arima_final()
        pred_arima_train_FX = output_model[0]
        pred_arima_FX = output_model[1]
        output_model = Holts_final()
        pred_holts_train_FX = output_model[0]
        pred_holts_FX = output_model[1]
        output_model = Sarima_final()
        pred_sarima_train_FX = output_model[0]
        pred_sarima_FX = output_model[1]
        output_model = Sarimax_final()
        pred_sarimax_train_FX = output_model[0]
        pred_sarimax_FX = output_model[1]

        Final_Train = {'Train':train['Claims'].reset_index(drop=True), 
                     'ARIMA':pd.Series(pred_arima_train_FX)[0], 
                       'SARIMA':pd.Series(pred_sarima_train_FX)[0], 
                     'SARIMAX':pd.Series(pred_sarimax_train_FX)[0], 
                     'Holts':pd.Series(pred_holts_train_FX)[0]}
        ensemble_train = pd.DataFrame(Final_Train)

        Final_Test = {'Test':test['Index'].reset_index(drop=True), 
                     'ARIMA':pd.Series(pred_arima_FX), 
                      'SARIMA':pd.Series(pred_sarima_FX), 
                     'SARIMAX':pd.Series(pred_sarimax_FX), 
                     'Holts':pd.Series(pred_holts_FX)}
        ensemble_test = pd.DataFrame(Final_Test)

        X_train = ensemble_train.iloc[1:,1:8]
        y_train =ensemble_train.iloc[1:,0:1]
        X_test = ensemble_train.iloc[0:,1:8]

        output_model = Xgboost_final(X_train, y_train, X_test)
        pred_xgboost_train_F = output_model[0]
        pred_xgboost_F = output_model[1]

# pred_arima_F = []
# pred_holts_F = []
# pred_sarima_F = []
# pred_sarimax_F = []
# pred_xgboost_F = []
# pred_arima_train_FX = []
# pred_arima_FX = []
# pred_holts_train_FX = []
# pred_holts_FX = []
# pred_sarima_train_FX = []
# pred_sarima_FX = []
# pred_sarimax_train_FX = []
# pred_sarimax_FX = []

Final_Train_Output = {'Date':train.reset_index()['Date'].reset_index(drop=True),
               'Train':train['Claims'].reset_index(drop=True), 
             'ARIMA':pd.Series(list(pred_arima_train_F)), 
                      'SARIMA':pd.Series(list(pred_sarima_train_F)), 
                     'SARIMAX':pd.Series(list(pred_sarimax_train_F)), 
                     'Holts':pd.Series(list(pred_holts_train_F)), 
                     'XGBOOST':pd.Series(list(pred_xgboost_train_F))}
ensemble_train_Final_Train_Output = pd.DataFrame(Final_Train_Output)
ensemble_train_Final_Train_Output['Average'] = np.round(ensemble_train_Final_Train_Output.mean(axis=1),0)

ensemble_train_Final_Train_Output.tail()

Final_Test_Output = {'Date':test.reset_index()['Date'].reset_index(drop=True), 
             'ARIMA':pd.Series(list(pred_arima_F)), 
                      'SARIMA':pd.Series(list(pred_sarima_F)), 
                     'SARIMAX':pd.Series(list(pred_sarimax_F)), 
                     'Holts':pd.Series(list(pred_holts_F)), 
                     'XGBOOST':pd.Series(list(pred_xgboost_F))}
ensemble_Final_test_Output = pd.DataFrame(Final_Test_Output)
ensemble_Final_test_Output['Average'] = np.round(ensemble_Final_test_Output.mean(axis=1),0)

ensemble_Final_test_Output.tail(90)

# datetime.now() - timestartsnow_Overall
##datetime.timedelta(seconds=968, microseconds=962040)

# os.chdir(r'Segment_ASC_All\Outputs')
writer = pd.ExcelWriter('Output\ISC PH\ISC PH Weekly.xlsx', engine='xlsxwriter')

ensemble_train_Final_Train_Output.to_excel(writer, sheet_name='Train', index=None)
ensemble_Final_test_Output.to_excel(writer, sheet_name='Test', index=None)
writer.save()

# Weekly to daily imputation

pd.to_datetime(enddate_test) - timedelta(weeks = 13) #13 weeks ago data


final_test_output = ensemble_Final_test_Output
final_test_output = final_test_output[["Date","Average"]]
final_test_output['Year Week'] = pd.to_datetime(final_test_output['Date']).dt.strftime('%Y%W')

final_test_output.head()

df_holiday = pd.read_excel(r'Data\Raw Data\Holiday_List.xlsx', sheet_name= 'Holiday List')
df_holiday

df_Input_Daily_With_Holiday = df_Input_Daily.merge(df_holiday[["Date", "Event"]], on = 'Date', how = 'left')
df_Input_Daily_With_Holiday['Week'] = pd.to_datetime(df_Input_Daily_With_Holiday['Date']).dt.week
df_Input_Daily_With_Holiday['Day'] = pd.to_datetime(df_Input_Daily_With_Holiday['Date']).dt.dayofweek

df_Input_Daily_With_Holiday.head(50)

weekly_event_Count = df_Input_Daily_With_Holiday.groupby('Week').Event.nunique().reset_index()
weekly_event_Count = weekly_event_Count[weekly_event_Count['Event']== 1]
weekly_event_Count

df_Input_Daily_With_Holiday = df_Input_Daily_With_Holiday[(df_Input_Daily_With_Holiday['Week'].isin(weekly_event_Count.Week)) 
                                  & (df_Input_Daily_With_Holiday['Day'] != 5) 
                                  & (df_Input_Daily_With_Holiday['Day'] != 6)]
df_Input_Daily_With_Holiday

Recent_weekly_call = df_Input_Daily_With_Holiday.groupby('Week')['Claims'].sum().reset_index()
Recent_weekly_call = Recent_weekly_call.rename(columns={"Claims":"WeekClaims"})
# Recent_weekly_call

df_Input_Daily_WO_Holiday = df_Input_Daily_With_Holiday[["Date","Claims","Day","Week"]]
df_Input_Daily_WO_Holiday

df_Input_Daily_WO_Holiday= df_Input_Daily_WO_Holiday.merge(Recent_weekly_call, on = 'Week')
df_Input_Daily_WO_Holiday

df_Input_Daily_WO_Holiday['CallsDist'] = np.round(df_Input_Daily_WO_Holiday['Claims']
                                                  / df_Input_Daily_WO_Holiday['WeekClaims'],2)

df_Input_Daily_WO_Holiday['Year Week'] = pd.to_datetime(df_Input_Daily_WO_Holiday['Date']).dt.strftime('%Y%W')

df_Input_Daily_WO_Holiday.head()

df_daily_avg = df_Input_Daily_WO_Holiday.groupby('Day').agg({"CallsDist":"mean"}).reset_index()

df_daily_avg

rng = pd.date_range(startdate_test, periods=400, freq='B')
test = pd.DataFrame(rng)
test.columns = ['Date']
test['Index'] = range(1, 1+(len(test)))
test['Index'] = test['Index'].astype(float)
# test = pd.DataFrame({ 'Date': rng, 'Val': range(0, len(df))})

test = test.loc[(test['Date'] >= StartDate_Test_Final) & (test['Date'] <= enddate_test_Final)]
test = test.reset_index()
test['Day'] = test['Date'].dt.dayofweek
pred_range = test[["Date","Day"]]
# pred_range.head()

weekly_forecasted = pred_range.merge(df_daily_avg, on = 'Day')
weekly_forecasted['Year Week'] = pd.to_datetime(weekly_forecasted['Date']).dt.strftime('%Y%W')
weekly_forecasted = weekly_forecasted.sort_values(by='Date')
weekly_forecasted = weekly_forecasted.merge(final_test_output[["Average","Year Week"]], on = 'Year Week')
weekly_forecasted

weekly_forecasted['ClaimsForecasted'] = np.round(weekly_forecasted.CallsDist* weekly_forecasted.Average,0)
weekly_forecasted = weekly_forecasted[["Date","Day","Year Week","ClaimsForecasted"]]
weekly_forecasted.head()


weekly_forecasted = weekly_forecasted.merge(df_holiday[["Date", "Event"]], on = 'Date', how = 'left')
weekly_forecasted

# weekly_forecasted.set_index('Date', inplace = True)
weekly_forecasted['Claims_Hol'] = np.where(weekly_forecasted['Event'] == 1, 0, 
                                                 weekly_forecasted['ClaimsForecasted'])

weekly_forecasted['Lag1'] = np.round(weekly_forecasted['Event'].shift(1),0)
weekly_forecasted['Lag2'] = np.round(weekly_forecasted['Event'].shift(2),0)

weekly_forecasted['CallsLag1'] = np.where(weekly_forecasted['Lag1'] == 1, 1.10*weekly_forecasted['Claims_Hol']
                                           ,weekly_forecasted['Claims_Hol'])
weekly_forecasted['CallsLag2'] = np.round(np.where(weekly_forecasted['Lag2'] == 1, 1.05*weekly_forecasted['CallsLag1']
                                           ,weekly_forecasted['CallsLag1']),0)

daily_forecasted_final = weekly_forecasted[["Date","CallsLag2"]].rename(columns={"CallsLag2":"DailyCallForecast"})

# weekly_forecasted.tail(40)

# daily_forecasted_final.head(50)

test['Date'] = pd.to_datetime(test['Date'])

### Writing Final File

# os.chdir(r'2. Segment Core Inquiry\Outputs')
writer = pd.ExcelWriter('Output\ISC PH\ISC PH Daily.xlsx', engine='xlsxwriter')

# Final_Train_File.to_excel(writer, sheet_name='Train', index=None)
daily_forecasted_final.to_excel(writer, sheet_name='Test', index=None)
writer.save()

print('end')

datetime.now() - timestartsnow_Overall

