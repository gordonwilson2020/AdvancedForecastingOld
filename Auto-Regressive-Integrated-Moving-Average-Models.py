##############################################################################
# Advanced Forecasting Models with Python                                    #
# Auto Regressive Integrated Moving Average Models                           #
# (c) Diego Fernandez Garcia 2015-2018                                       #
# www.exfinsis.com                                                           #
##############################################################################

# 1. Packages Importing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.tsa.stattools as ts
import statsmodels.graphics.tsaplots as tsp
import statsmodels.tsa.statespace.sarimax as sarima
import statsmodels.regression.linear_model as rg
import statsmodels.tools.eval_measures as fa

##########################################

# 2. Advanced Forecasting Models Data

# 2.1. Data Reading
spy = pd.read_csv('Data//Advanced-Forecasting-Models-Data.txt', index_col='Date', parse_dates=True)
spy = spy.asfreq('B')
spy = spy.fillna(method='ffill')
print('')
print('== Data Ranges Length ==')
print('')
print('Full Range Days: ', len(spy))
print('Full Range Months: ', np.round(len(spy)/22, 4))
print('')

# 2.2. Training Range Delimiting
spyt = spy[:'2013-12-31']
print('Training Range Days: ', len(spyt))
print('Training Range Months: ', np.round(len(spyt)/22, 4))
print('')

# 2.3. Testing Range Delimiting
spyf = spy['2014-01-02':]
print('Testing Range Days: ', len(spyf))
print('Testing Range Months: ', np.round(len(spyf)/22, 4))
print('')

# 2.4. Training and Testing Ranges Chart
fig1, ax = plt.subplots()
ax.plot(spyt, label='spyt')
ax.plot(spyf, label='spyf')
plt.legend(loc='upper left')
plt.title('SPY 2007-2015')
plt.ylabel('Price')
plt.xlabel('Date')
plt.show()

##########################################

# 3. Auto Regressive Integrated Moving Average Models

# 3.1. First Order Stationary Trend

# 3.1.1. Unit Root Tests
adfspyt = ts.adfuller(spyt.iloc[:, 0].values)
print('')
print('== Unit Root Test (spyt) ==')
print('')
print('Augmented Dickey-Fuller ADF Test P-Value: ', adfspyt[1])
print('')

# 3.1.2. Time Series Level Differencing
dspyt = spyt-spyt.shift(1)
dspyt = dspyt.fillna(method='bfill')

fig3, ax = plt.subplots()
ax.plot(spyt, label='spyt')
plt.legend(loc='upper left')
plt.title('SPY 2007-2013')
plt.ylabel('Price')
plt.xlabel('Date')
plt.show()

fig4, ax = plt.subplots()
ax.plot(dspyt, label='dspyt')
plt.legend(loc='upper left')
plt.title('SPY 2007-2013')
plt.ylabel('Price Differences')
plt.xlabel('Date')
plt.show()

adfdspyt = ts.adfuller(dspyt.iloc[:, 0].values)
print('')
print('== Unit Root Test (dspyt) ==')
print('')
print('Augmented Dickey-Fuller Test ADF P-Value: ', adfdspyt[1])
print('')

# 3.1.3. ARIMA Models Specification

# 3.1.3.1. Auto-correlation Function ACF
tsp.plot_acf(dspyt.iloc[:, 0].values, lags=22, alpha=0.05)
plt.title('Autocorrelation Function ACF (dspyt)')
plt.show()

# 3.1.3.2. Partial Auto-correlation Function PACF
tsp.plot_pacf(dspyt.iloc[:, 0].values, lags=22, alpha=0.05)
plt.title('Partial Autocorrelation Function ACF (dspyt)')
plt.show()

##########################################

# 3.2. ARIMA(0,1,0)

# 3.2.1. Multi-Steps Forecast
rwdt1 = sarima.SARIMAX(spyt, order=(0, 1, 0), trend='c').fit(disp=-1)
print('')
print('== ARIMA(0,1,0) Model (spyt) ==')
print('')
print(rwdt1.summary())
print('')
rwdf1 = rwdt1.forecast(steps=len(spyf))
rwdf1 = pd.DataFrame(rwdf1).set_index(spyf.index)

fig5, ax = plt.subplots()
ax.plot(spyt, label='spyt')
ax.plot(spyf, label='spyf')
ax.plot(rwdf1, label='rwdf1')
plt.legend(loc='upper left')
plt.title('ARIMA(0,1,0) Model 1')
plt.ylabel('Price')
plt.xlabel('Date')
plt.show()

# 3.2.2. One-Step Forecast without Re-Estimation
rwdf2 = sarima.SARIMAX(spy.tail(len(spyf)+1), order=(0, 1, 0), trend='c').smooth(params=rwdt1.params)
rwdf2 = rwdf2.fittedvalues.tail(len(spyf))
rwdf2 = pd.DataFrame(rwdf2).set_index(spyf.index)

fig6, ax = plt.subplots()
ax.plot(spyt, label='spyt')
ax.plot(spyf, label='spyf')
ax.plot(rwdf2, label='rwdf2', linestyle=':')
plt.legend(loc='upper left')
plt.title('ARIMA(0,1,0) Model 2')
plt.ylabel('Price')
plt.xlabel('Date')
plt.show()

##########################################

# 3.3. ARIMA(1,1,0)

# 3.3.1. Multi-Steps Forecast
dart1 = sarima.SARIMAX(spyt, order=(1, 1, 0), trend='c').fit(disp=-1)
print('')
print('== ARIMA(1,1,0) Model (spyt) ==')
print('')
print(dart1.summary())
print('')
darf1 = dart1.forecast(steps=len(spyf))
darf1 = pd.DataFrame(darf1).set_index(spyf.index)

fig7, ax = plt.subplots()
ax.plot(spyt, label='spyt')
ax.plot(spyf, label='spyf')
ax.plot(darf1, label='darf1')
plt.legend(loc='upper left')
plt.title('ARIMA(1,1,0) Model 1')
plt.ylabel('Price')
plt.xlabel('Date')
plt.show()

# 3.3.2. One-Step Forecast without Re-Estimation
darf2 = sarima.SARIMAX(spy.tail(len(spyf)+1), order=(1, 1, 0), trend='c').smooth(params=dart1.params)
darf2 = darf2.fittedvalues.tail(len(spyf))
darf2 = pd.DataFrame(darf2).set_index(spyf.index)

fig8, ax = plt.subplots()
ax.plot(spyt, label='spyt')
ax.plot(spyf, label='spyf')
ax.plot(darf2, label='darf2', linestyle=':')
plt.legend(loc='upper left')
plt.title('ARIMA(1,1,0) Model 2')
plt.ylabel('Price')
plt.xlabel('Date')
plt.show()

##########################################

# 3.4. First Order Seasonality Stationary

# 3.4.1. Deterministic Seasonality Test
sdt = pd.get_dummies(spyt.index.day, drop_first=True)
sdt = sdt.set_index(spyt.index)
dstspyt = rg.OLS(endog=spyt, exog=sdt).fit()
print('')
print('== Deterministic Seasonality Test (spyt) ==')
print('')
print(dstspyt.summary())
print('')

# 3.4.2. Time Series Seasonal Differencing
sdspyt = spyt-spyt.shift(22)
sdspyt = sdspyt.fillna(method='bfill')

fig9, ax = plt.subplots()
ax.plot(spyt, label='spyt')
plt.legend(loc='upper left')
plt.title('SPY 2007-2013')
plt.ylabel('Price')
plt.xlabel('Date')
plt.show()

fig10, ax = plt.subplots()
ax.plot(dspyt, label='dspyt')
plt.legend(loc='upper left')
plt.title('SPY 2007-2013')
plt.ylabel('Price Differences')
plt.xlabel('Date')
plt.show()

fig11, ax = plt.subplots()
ax.plot(sdspyt, label='sdspyt')
plt.legend(loc='upper left')
plt.title('SPY 2007-2013')
plt.ylabel('Price Seasonal Differences')
plt.xlabel('Date')
plt.show()

dstsdspyt = rg.OLS(endog=sdspyt, exog=sdt).fit()
print('')
print('== Deterministic Seasonality Test (sdspyt) ==')
print('')
print(dstsdspyt.summary())
print('')

# 3.4.3. SARIMA Models Specification

# 3.4.3.1. Auto-correlation Function ACF
tsp.plot_acf(sdspyt.iloc[:, 0].values, lags=12, alpha=0.05)
plt.title('Autocorrelation Function ACF (sdspyt)')
plt.show()

# 3.4.3.2. Partial Auto-correlation Function PACF
tsp.plot_pacf(sdspyt.iloc[:, 0].values, lags=12, alpha=0.05)
plt.title('Partial Autocorrelation Function ACF (sdspyt)')
plt.show()

##########################################

# 3.5. SARIMA(0,0,0)x(0,1,0)[22]

# 3.5.1. Multi-Steps Forecast
srwdt1 = sarima.SARIMAX(spyt, order=(0, 0, 0), seasonal_order=(0, 1, 0, 22), trend='c').fit(disp=-1)
print('')
print('== SARIMA(0,0,0)x(0,1,0)[22] Model (spyt) ==')
print('')
print(srwdt1.summary())
print('')
srwdf1 = srwdt1.forecast(steps=len(spyf))
srwdf1 = pd.DataFrame(srwdf1).set_index(spyf.index)

fig12, ax = plt.subplots()
ax.plot(spyt, label='spyt')
ax.plot(spyf, label='spyf')
ax.plot(srwdf1, label='srwdf1')
plt.legend(loc='upper left')
plt.title('SARIMA(0,0,0)x(0,1,0)[22] Model 1')
plt.ylabel('Price')
plt.xlabel('Date')
plt.show()

# 3.5.2. One-Step Forecast without Re-Estimation
srwdf2 = sarima.SARIMAX(spy.tail(len(spyf)+22), order=(0, 0, 0), seasonal_order=(0, 1, 0, 22),
                        trend='c').smooth(params=srwdt1.params)
srwdf2 = srwdf2.fittedvalues.tail(len(spyf))
srwdf2 = pd.DataFrame(srwdf2).set_index(spyf.index)

fig13, ax = plt.subplots()
ax.plot(spyt, label='spyt')
ax.plot(spyf, label='spyf')
ax.plot(srwdf2, label='srwdf2', linestyle=':')
plt.legend(loc='upper left')
plt.title('SARIMA(0,0,0)x(0,1,0)[22] Model 2')
plt.ylabel('Price')
plt.xlabel('Date')
plt.show()

##########################################

# 3.6. SARIMA(1,0,0)x(0,1,0)[22]

# 3.6.1. Multi-Steps Forecast
sdart1 = sarima.SARIMAX(spyt, order=(1, 0, 0), seasonal_order=(0, 1, 0, 22), trend='c').fit(disp=-1)
print('')
print('== SARIMA(1,0,0)x(0,1,0)[22] Model (spyt) ==')
print('')
print(sdart1.summary())
print('')
sdarf1 = sdart1.forecast(steps=len(spyf))
sdarf1 = pd.DataFrame(sdarf1).set_index(spyf.index)

fig14, ax = plt.subplots()
ax.plot(spyt, label='spyt')
ax.plot(spyf, label='spyf')
ax.plot(sdarf1, label='sdarf1')
plt.legend(loc='upper left')
plt.title('SARIMA(1,0,0)x(0,1,0)[22] Model 1')
plt.ylabel('Price')
plt.xlabel('Date')
plt.show()

# 3.6.2. One-Step Forecast without Re-Estimation
sdarf2 = sarima.SARIMAX(spy.tail(len(spyf)+22), order=(1, 0, 0), seasonal_order=(0, 1, 0, 22),
                        trend='c').smooth(sdart1.params)
sdarf2 = sdarf2.fittedvalues.tail(len(spyf))
sdarf2 = pd.DataFrame(sdarf2).set_index(spyf.index)

fig15, ax = plt.subplots()
ax.plot(spyt, label='spyt')
ax.plot(spyf, label='spyf')
ax.plot(sdarf2, label='sdarf2', linestyle=':')
plt.legend(loc='upper left')
plt.title('SARIMA(1,0,0)x(0,1,0)[22] Model 2')
plt.ylabel('Price')
plt.xlabel('Date')
plt.show()

#########################################

# 3.7. ARIMA Model Selection

rwdaict1 = rwdt1.aic
rwdbict1 = rwdt1.bic
daraict1 = dart1.aic
darbict1 = dart1.bic
srwdaict1 = srwdt1.aic
srwdbict1 = srwdt1.bic
sdaraict1 = sdart1.aic
sdarbict1 = sdart1.bic

msdata = [{'0': '', '1': 'AIC', '2': 'BIC'},
        {'0': 'ARIMA(0,1,0) Model 1', '1': np.round(rwdaict1, 4), '2': np.round(rwdbict1, 4)},
        {'0': 'ARIMA(1,1,0) Model 1', '1': np.round(daraict1, 4), '2': np.round(darbict1, 4)},
        {'0': 'SARIMA(0,0,0)x(0,1,0)[22] Model 1',  '1': np.round(srwdaict1, 4), '2': np.round(srwdbict1, 4)},
        {'0': 'SARIMA(1,0,0)x(0,1,0)[22] Model 1', '1': np.round(sdaraict1, 4), '2': np.round(sdarbict1, 4)},
          ]
mstable = pd.DataFrame(msdata)
print('')
print('== ARIMA Model Selection ==')
print('')
print(mstable)
print('')

##########################################

# 3.8. ARIMA Models Forecasting Accuracy

# 3.8.1. Multi-Steps Forecast
rwdmae1 = fa.meanabs(rwdf1, spyf)
rwdrmse1 = fa.rmse(rwdf1, spyf)
darmae1 = fa.meanabs(darf1, spyf)
darrmse1 = fa.rmse(darf1, spyf)
srwdmae1 = fa.meanabs(srwdf1, spyf)
srwdrmse1 = fa.rmse(srwdf1, spyf)
sdarmae1 = fa.meanabs(sdarf1, spyf)
sdarrmse1 = fa.rmse(sdarf1, spyf)

fadata1 = [{'0': '', '1': 'MAE', '2': 'RMSE'},
        {'0': 'ARIMA(0,1,0) Model 1', '1': np.round(rwdmae1, 4), '2': np.round(rwdrmse1, 4)},
        {'0': 'ARIMA(1,1,0) Model 1', '1': np.round(darmae1, 4), '2': np.round(darrmse1, 4)},
        {'0': 'SARIMA(0,0,0)x(0,1,0)[22] Model 1', '1': np.round(srwdmae1, 4), '2': np.round(srwdrmse1, 4)},
        {'0': 'SARIMA(1,0,0)x(0,1,0)[22] Model 1', '1': np.round(sdarmae1, 4), '2': np.round(sdarrmse1, 4)},
           ]
fatable1 = pd.DataFrame(fadata1)
print('')
print('== Multi-Steps Forecasting Accuracy ==')
print('')
print(fatable1)
print('')

# 3.8.2. One-Step Forecast without Re-Estimation
rwdmae2 = fa.meanabs(rwdf2, spyf)
rwdrmse2 = fa.rmse(rwdf2, spyf)
darmae2 = fa.meanabs(darf2, spyf)
darrmse2 = fa.rmse(darf2, spyf)
srwdmae2 = fa.meanabs(srwdf2, spyf)
srwdrmse2 = fa.rmse(srwdf2, spyf)
sdarmae2 = fa.meanabs(sdarf2, spyf)
sdarrmse2 = fa.rmse(sdarf2, spyf)

fadata2 = [{'0': '', '1': 'MAE', '2': 'RMSE'},
        {'0': 'ARIMA(0,1,0) Model 2', '1': np.round(rwdmae2, 4), '2': np.round(rwdrmse2, 4)},
        {'0': 'ARIMA(1,1,0) Model 2', '1': np.round(darmae2, 4), '2': np.round(darrmse2, 4)},
        {'0': 'SARIMA(0,0,0)x(0,1,0)[22] Model 2', '1': np.round(srwdmae2, 4), '2': np.round(srwdrmse2, 4)},
        {'0': 'SARIMA(1,0,0)x(0,1,0)[22] Model 2', '1': np.round(sdarmae2, 4), '2': np.round(sdarrmse2, 4)},
           ]
fatable2 = pd.DataFrame(fadata2)
print('')
print('== One-Step without Re-Estimation Forecasting Accuracy ==')
print('')
print(fatable2)
print('')