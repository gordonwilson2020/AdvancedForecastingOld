##############################################################################
# Advanced Forecasting Models with Python                                    #
# Non-Gaussian General Auto Regressive Conditional Heteroscedasticity Models #
# (c) Diego Fernandez Garcia 2015-2018                                       #
# www.exfinsis.com                                                           #
##############################################################################

# 1. Packages Importing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.graphics.tsaplots as tsp
import statsmodels.tsa.statespace.sarimax as sarima
import statsmodels.stats.diagnostic as st
import statsmodels.stats.stattools as jb
import arch.univariate as arch
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

# 3.2. ARIMA(0,1,0) Model

# 3.2.1. Multi-Steps Forecast
rwdt1 = sarima.SARIMAX(spyt, order=(0, 1, 0), trend='c').fit(disp=-1)
print('')
print('== ARIMA(0,1,0) Model (spyt) ==')
print('')
print(rwdt1.summary())
print('')
rwdf1 = rwdt1.forecast(steps=len(spyf))
rwdf1 = pd.DataFrame(rwdf1).set_index(spyf.index)

fig2, ax = plt.subplots()
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

fig3, ax = plt.subplots()
ax.plot(spyt, label='spyt')
ax.plot(spyf, label='spyf')
ax.plot(rwdf2, label='rwdf2', linestyle=':')
plt.legend(loc='upper left')
plt.title('ARIMA(0,1,0) Model 2')
plt.ylabel('Price')
plt.xlabel('Date')
plt.show()

##########################################

# 4. General Auto Regressive Conditional Heteroscedasticity Models

# 4.1.3. Time Series Level Differencing
dspy = spy-spy.shift(1)
dspy = dspy.fillna(method='bfill')
dspyt = dspy[:'2013-12-31']
dspyf = dspy['2014-01-02':]

# 4.6. ARIMA(0,1,0)-GJR-GARCH(1,1)

# 4.6.1. Multi-Steps Forecast
rwdtgarcht1 = arch.ConstantMean(y=dspyt, volatility=arch.GARCH(p=1, o=1, q=1),
                                distribution=arch.Normal()).fit()
print('')
print('== ARIMA(0,1,0)-GJR-GARCH(1,1) Model (dspyt) ==')
print('')
print(rwdtgarcht1.summary())
print('')
rwdtgarchf1 = pd.DataFrame(pd.concat([spyt.tail(1)]*len(spyf))).set_index(spyf.index)
rwdtgarchf1['SPY.CondDrift'] = pd.DataFrame(pd.concat(
    [rwdtgarcht1.forecast().mean.tail(1)]*len(spyf))).set_index(spyf.index).cumsum()
rwdtgarchf1 = rwdtgarchf1.sum(axis=1)
rwdtgarchf1 = pd.DataFrame(rwdtgarchf1).set_index(spyf.index)

fig6, ax = plt.subplots()
ax.plot(spyt, label='spyt')
ax.plot(spyf, label='spyf')
ax.plot(rwdtgarchf1, label='rwdtgarchf1')
plt.legend(loc='upper left')
plt.title('ARIMA(0,1,0)-GJR-GARCH(1,1) Model 1')
plt.ylabel('Price')
plt.xlabel('Date')
plt.show()

# 4.6.2. One-Step Forecast without Re-Estimation
rwdtgarchf2 = spy.shift(1)['2014-01-02':]
rwdtgarchf2['SPY.CondMean'] = pd.DataFrame(pd.concat(
    [rwdtgarcht1.forecast().mean.tail(1)]*len(spyf))).set_index(spyf.index)
rwdtgarchf2 = rwdtgarchf2.sum(axis=1)
rwdtgarchf2 = pd.DataFrame(rwdtgarchf2).set_index(spyf.index)

fig7, ax = plt.subplots()
ax.plot(spyt, label='spyt')
ax.plot(spyf, label='spyf')
ax.plot(rwdtgarchf2, label='rwdtgarchf2', linestyle=':')
plt.legend(loc='upper left')
plt.title('ARIMA(0,1,0)-GJR-GARCH(1,1) Model 2')
plt.ylabel('Price')
plt.xlabel('Date')
plt.show()

##########################################

# 5. Non-Gaussian General Auto Regressive Conditional Heteroscedasticity Models

# 5.1. Normality Tests

# 5.1.1. Jarque-Bera Normality Test
rwdtgarcht1sres = rwdtgarcht1.resid.dropna()/rwdtgarcht1.conditional_volatility
jbrwdtgarcht1 = jb.jarque_bera(rwdtgarcht1sres)
print('')
print('== Jarque-Bera Normality Tests (rwdtgarcht1sres) ==')
print('')
print('Jarque-Bera Test Chi-Squared P-Value (rwdtgarcht1sres): ', np.round(jbrwdtgarcht1[1], 4))
print('')

##########################################

# 5.2. ARIMA(0,1,0)-GARCH-t(1,1)

# 5.2.1. Multi-Steps Forecast
trwdgarcht1 = arch.ConstantMean(y=dspyt, volatility=arch.GARCH(p=1, o=0, q=1, power=2.0),
                                distribution=arch.StudentsT()).fit()
print('')
print('== ARIMA(0,1,0)-GARCH-t(1,1) Model (dspyt) ==')
print('')
print(trwdgarcht1.summary())
print('')
trwdgarchf1 = pd.DataFrame(pd.concat([spyt.tail(1)]*len(spyf))).set_index(spyf.index)
trwdgarchf1['SPY.CondDrift'] = pd.DataFrame(pd.concat(
    [trwdgarcht1.forecast().mean.tail(1)]*len(spyf))).set_index(spyf.index).cumsum()
trwdgarchf1 = trwdgarchf1.sum(axis=1)
trwdgarchf1 = pd.DataFrame(trwdgarchf1).set_index(spyf.index)

fig8, ax = plt.subplots()
ax.plot(spyt, label='spyt')
ax.plot(spyf, label='spyf')
ax.plot(trwdgarchf1, label='trwdgarchf1')
plt.legend(loc='upper left')
plt.title('ARIMA(0,1,0)-GARCH-t(1,1) Model 1')
plt.ylabel('Price')
plt.xlabel('Date')
plt.show()

# 5.2.2. One-Step Forecast without Re-Estimation
trwdgarchf2 = spy.shift(1)['2014-01-02':]
trwdgarchf2['SPY.CondMean'] = pd.DataFrame(pd.concat(
    [trwdgarcht1.forecast().mean.tail(1)]*len(spyf))).set_index(spyf.index)
trwdgarchf2 = trwdgarchf2.sum(axis=1)
trwdgarchf2 = pd.DataFrame(trwdgarchf2).set_index(spyf.index)

fig9, ax = plt.subplots()
ax.plot(spyt, label='spyt')
ax.plot(spyf, label='spyf')
ax.plot(trwdgarchf2, label='trwdgarchf2', linestyle=':')
plt.legend(loc='upper left')
plt.title('ARIMA(0,1,0)-GARCH-t(1,1) Model 2')
plt.ylabel('Price')
plt.xlabel('Date')
plt.show()

##########################################

# 5.3. ARIMA(1,1,0)-GARCH-t(1,1)

# 5.3.1. Multi-Steps Forecast
tdargarcht1 = arch.ARX(y=dspyt, volatility=arch.GARCH(p=1, o=0, q=1, power=2.0), lags=[1], constant=True,
                       distribution=arch.StudentsT()).fit()
print('')
print('== ARIMA(1,1,0)-GARCH-t(1,1) Model (dspyt) ==')
print('')
print(tdargarcht1.summary())
print('')
tdargarchf1 = pd.DataFrame(pd.concat([tdargarcht1.params[1]*dspyt.tail(1)]*len(spyf))).set_index(spyf.index)
tdargarchf1['SPY.LastAdj'] = pd.DataFrame(pd.concat([spyt.tail(1)]*len(spyf))).set_index(spyf.index)
tdargarchf1['SPY.CondDrift'] = pd.DataFrame(pd.concat(
    [tdargarcht1.forecast().mean.tail(1)]*len(spyf))).set_index(spyf.index).cumsum()
tdargarchf1 = tdargarchf1.sum(axis=1)
tdargarchf1 = pd.DataFrame(tdargarchf1).set_index(spyf.index)

fig10, ax = plt.subplots()
ax.plot(spyt, label='spyt')
ax.plot(spyf, label='spyf')
ax.plot(tdargarchf1, label='tdargarchf1')
plt.legend(loc='upper left')
plt.title('ARIMA(1,1,0)-GARCH-t(1,1) Model 1')
plt.ylabel('Price')
plt.xlabel('Date')
plt.show()

# 5.3.2. One-Step Forecast without Re-Estimation
tdargarchf2 = dspy.shift(1)['2014-01-02':]
tdargarchf2 = tdargarchf2*tdargarcht1.params[1]
tdargarchf2['SPY.Adjusted(-1)'] = spy.shift(1)['2014-01-02':]
tdargarchf2['SPY.CondMean'] = pd.DataFrame(pd.concat(
    [tdargarcht1.forecast().mean.tail(1)]*len(spyf))).set_index(spyf.index)
tdargarchf2 = tdargarchf2.sum(axis=1)
tdargarchf2 = pd.DataFrame(tdargarchf2).set_index(spyf.index)

fig11, ax = plt.subplots()
ax.plot(spyt, label='spyt')
ax.plot(spyf, label='spyf')
ax.plot(tdargarchf2, label='tdargarchf2', linestyle=':')
plt.legend(loc='upper left')
plt.title('ARIMA(1,1,0)-GARCH-t(1,1) Model 2')
plt.ylabel('Price')
plt.xlabel('Date')
plt.show()

##########################################

# 5.4. ARIMA(0,1,0)-EGARCH-t(1,1)

# 5.4.1. Multi-Steps Forecast
trwdegarcht1 = arch.ConstantMean(y=dspyt, volatility=arch.EGARCH(p=1, o=1, q=1),
                                distribution=arch.StudentsT()).fit()
print('')
print('== ARIMA(0,1,0)-EGARCH-t(1,1) Model (dspyt) ==')
print('')
print(trwdegarcht1.summary())
print('')
trwdegarchf1 = pd.DataFrame(pd.concat([spyt.tail(1)]*len(spyf))).set_index(spyf.index)
trwdegarchf1['SPY.CondDrift'] = pd.DataFrame(pd.concat(
    [trwdegarcht1.forecast().mean.tail(1)]*len(spyf))).set_index(spyf.index).cumsum()
trwdegarchf1 = trwdegarchf1.sum(axis=1)
trwdegarchf1 = pd.DataFrame(trwdegarchf1).set_index(spyf.index)

fig12, ax = plt.subplots()
ax.plot(spyt, label='spyt')
ax.plot(spyf, label='spyf')
ax.plot(trwdegarchf1, label='trwdegarchf1')
plt.legend(loc='upper left')
plt.title('ARIMA(0,1,0)-EGARCH-t(1,1) Model 1')
plt.ylabel('Price')
plt.xlabel('Date')
plt.show()

# 5.4.2. One-Step Forecast without Re-Estimation
trwdegarchf2 = spy.shift(1)['2014-01-02':]
trwdegarchf2['SPY.CondMean'] = pd.DataFrame(pd.concat(
    [trwdegarcht1.forecast().mean.tail(1)]*len(spyf))).set_index(spyf.index)
trwdegarchf2 = trwdegarchf2.sum(axis=1)
trwdegarchf2 = pd.DataFrame(trwdegarchf2).set_index(spyf.index)

fig13, ax = plt.subplots()
ax.plot(spyt, label='spyt')
ax.plot(spyf, label='spyf')
ax.plot(trwdegarchf2, label='trwdegarchf2', linestyle=':')
plt.legend(loc='upper left')
plt.title('ARIMA(0,1,0)-EGARCH-t(1,1) Model 2')
plt.ylabel('Price')
plt.xlabel('Date')
plt.show()

##########################################

# 5.5. ARIMA(1,1,0)-EGARCH-t(1,1)

# 5.5.1. Multi-Steps Forecast
tdaregarcht1 = arch.ARX(y=dspyt, volatility=arch.EGARCH(p=1, o=1, q=1), lags=[1], constant=True,
                       distribution=arch.StudentsT()).fit()
print('')
print('== ARIMA(1,1,0)-EGARCH-t(1,1) Model (dspyt) ==')
print('')
print(tdaregarcht1.summary())
print('')
tdaregarchf1 = pd.DataFrame(pd.concat([tdaregarcht1.params[1]*dspyt.tail(1)]*len(spyf))).set_index(spyf.index)
tdaregarchf1['SPY.LastAdj'] = pd.DataFrame(pd.concat([spyt.tail(1)]*len(spyf))).set_index(spyf.index)
tdaregarchf1['SPY.CondDrift'] = pd.DataFrame(pd.concat(
    [tdaregarcht1.forecast().mean.tail(1)]*len(spyf))).set_index(spyf.index).cumsum()
tdaregarchf1 = tdaregarchf1.sum(axis=1)
tdaregarchf1 = pd.DataFrame(tdaregarchf1).set_index(spyf.index)

fig14, ax = plt.subplots()
ax.plot(spyt, label='spyt')
ax.plot(spyf, label='spyf')
ax.plot(tdaregarchf1, label='tdaregarchf1')
plt.legend(loc='upper left')
plt.title('ARIMA(1,1,0)-EGARCH-t(1,1) Model 1')
plt.ylabel('Price')
plt.xlabel('Date')
plt.show()

# 5.5.2. One-Step Forecast without Re-Estimation
tdaregarchf2 = dspy.shift(1)['2014-01-02':]
tdaregarchf2 = tdaregarchf2*tdaregarcht1.params[1]
tdaregarchf2['SPY.Adjusted(-1)'] = spy.shift(1)['2014-01-02':]
tdaregarchf2['SPY.CondMean'] = pd.DataFrame(pd.concat(
    [tdaregarcht1.forecast().mean.tail(1)]*len(spyf))).set_index(spyf.index)
tdaregarchf2 = tdaregarchf2.sum(axis=1)
tdaregarchf2 = pd.DataFrame(tdaregarchf2).set_index(spyf.index)

fig15, ax = plt.subplots()
ax.plot(spyt, label='spyt')
ax.plot(spyf, label='spyf')
ax.plot(tdaregarchf2, label='tdaregarchf2', linestyle=':')
plt.legend(loc='upper left')
plt.title('ARIMA(1,1,0)-EGARCH-t(1,1) Model 2')
plt.ylabel('Price')
plt.xlabel('Date')
plt.show()

##########################################

# 5.6. ARIMA(0,1,0)-GJR-GARCH-t(1,1)

# 5.6.1. Multi-Steps Forecast
trwdtgarcht1 = arch.ConstantMean(y=dspyt, volatility=arch.GARCH(p=1, o=1, q=1),
                                distribution=arch.StudentsT()).fit()
print('')
print('== ARIMA(0,1,0)-GJR-GARCH-t(1,1) Model (dspyt) ==')
print('')
print(trwdtgarcht1.summary())
print('')
trwdtgarchf1 = pd.DataFrame(pd.concat([spyt.tail(1)]*len(spyf))).set_index(spyf.index)
trwdtgarchf1['SPY.CondDrift'] = pd.DataFrame(pd.concat(
    [trwdtgarcht1.forecast().mean.tail(1)]*len(spyf))).set_index(spyf.index).cumsum()
trwdtgarchf1 = trwdtgarchf1.sum(axis=1)
trwdtgarchf1 = pd.DataFrame(trwdtgarchf1).set_index(spyf.index)

fig16, ax = plt.subplots()
ax.plot(spyt, label='spyt')
ax.plot(spyf, label='spyf')
ax.plot(trwdtgarchf1, label='trwdtgarchf1')
plt.legend(loc='upper left')
plt.title('ARIMA(0,1,0)-GJR-GARCH-t(1,1) Model 1')
plt.ylabel('Price')
plt.xlabel('Date')
plt.show()

# 5.6.2. One-Step Forecast without Re-Estimation
trwdtgarchf2 = spy.shift(1)['2014-01-02':]
trwdtgarchf2['SPY.CondMean'] = pd.DataFrame(pd.concat(
    [trwdtgarcht1.forecast().mean.tail(1)]*len(spyf))).set_index(spyf.index)
trwdtgarchf2 = trwdtgarchf2.sum(axis=1)
trwdtgarchf2 = pd.DataFrame(trwdtgarchf2).set_index(spyf.index)

fig17, ax = plt.subplots()
ax.plot(spyt, label='spyt')
ax.plot(spyf, label='spyf')
ax.plot(trwdtgarchf2, label='trwdtgarchf2', linestyle=':')
plt.legend(loc='upper left')
plt.title('ARIMA(0,1,0)-GJR-GARCH-t(1,1) Model 2')
plt.ylabel('Price')
plt.xlabel('Date')
plt.show()

##########################################

# 5.7. ARIMA(1,1,0)-GJR-GARCH-t(1,1)

# 5.7.1. Multi-Steps Forecast
tdartgarcht1 = arch.ARX(y=dspyt, volatility=arch.GARCH(p=1, o=1, q=1), lags=[1], constant=True,
                       distribution=arch.StudentsT()).fit()
print('')
print('== ARIMA(1,1,0)-GJR-GARCH-t(1,1) Model (dspyt) ==')
print('')
print(tdartgarcht1.summary())
print('')
tdartgarchf1 = pd.DataFrame(pd.concat([tdartgarcht1.params[1]*dspyt.tail(1)]*len(spyf))).set_index(spyf.index)
tdartgarchf1['SPY.LastAdj'] = pd.DataFrame(pd.concat([spyt.tail(1)]*len(spyf))).set_index(spyf.index)
tdartgarchf1['SPY.CondDrift'] = pd.DataFrame(pd.concat(
    [tdartgarcht1.forecast().mean.tail(1)]*len(spyf))).set_index(spyf.index).cumsum()
tdartgarchf1 = tdartgarchf1.sum(axis=1)
tdartgarchf1 = pd.DataFrame(tdartgarchf1).set_index(spyf.index)

fig18, ax = plt.subplots()
ax.plot(spyt, label='spyt')
ax.plot(spyf, label='spyf')
ax.plot(tdartgarchf1, label='tdartgarchf1')
plt.legend(loc='upper left')
plt.title('ARIMA(1,1,0)-GJR-GARCH-t(1,1) Model 1')
plt.ylabel('Price')
plt.xlabel('Date')
plt.show()

# 5.7.2. One-Step Forecast without Re-Estimation
tdartgarchf2 = dspy.shift(1)['2014-01-02':]
tdartgarchf2 = tdartgarchf2*tdartgarcht1.params[1]
tdartgarchf2['SPY.Adjusted(-1)'] = spy.shift(1)['2014-01-02':]
tdartgarchf2['SPY.CondMean'] = pd.DataFrame(pd.concat(
    [tdartgarcht1.forecast().mean.tail(1)]*len(spyf))).set_index(spyf.index)
tdartgarchf2 = tdartgarchf2.sum(axis=1)
tdartgarchf2 = pd.DataFrame(tdartgarchf2).set_index(spyf.index)

fig19, ax = plt.subplots()
ax.plot(spyt, label='spyt')
ax.plot(spyf, label='spyf')
ax.plot(tdartgarchf2, label='tdartgarchf2', linestyle=':')
plt.legend(loc='upper left')
plt.title('ARIMA(1,1,0)-GJR-GARCH-t(1,1) Model 2')
plt.ylabel('Price')
plt.xlabel('Date')
plt.show()

#########################################

# 5.8. GARCH-t Model Selection

rwdtgarchaict1 = rwdtgarcht1.aic
rwdtgarchbict1 = rwdtgarcht1.bic
trwdgarchaict1 = trwdgarcht1.aic
trwdgarchbict1 = trwdgarcht1.bic
tdargarchaict1 = tdargarcht1.aic
tdargarchbict1 = tdargarcht1.bic
trwdegarchaict1 = trwdegarcht1.aic
trwdegarchbict1 = trwdegarcht1.bic
tdaregarchaict1 = tdaregarcht1.aic
tdaregarchbict1 = tdaregarcht1.bic
trwdtgarchaict1 = trwdtgarcht1.aic
trwdtgarchbict1 = trwdtgarcht1.bic
tdartgarchaict1 = tdartgarcht1.aic
tdartgarchbict1 = tdartgarcht1.bic

msdata = [{'0': '', '1': 'AIC', '2': 'BIC'},
        {'0': 'ARIMA(0,1,0)-GJR-GARCH(1,1) Model 1', '1': np.round(rwdtgarchaict1, 4),
         '2': np.round(rwdtgarchbict1, 4)},
        {'0': 'ARIMA(0,1,0)-GARCH-t(1,1) Model 1', '1': np.round(trwdgarchaict1, 4),
         '2': np.round(trwdgarchbict1, 4)},
        {'0': 'ARIMA(1,1,0)-GARCH-t(1,1) Model 1', '1': np.round(tdargarchaict1, 4),
         '2': np.round(tdargarchbict1, 4)},
        {'0': 'ARIMA(0,1,0)-EGARCH-t(1,1) Model 1', '1': np.round(trwdegarchaict1, 4),
         '2': np.round(trwdegarchbict1, 4)},
        {'0': 'ARIMA(1,1,0)-EGARCH-t(1,1) Model 1', '1': np.round(tdaregarchaict1, 4),
         '2': np.round(tdaregarchbict1, 4)},
        {'0': 'ARIMA(0,1,0)-GJR-GARCH-t(1,1) Model 1', '1': np.round(trwdtgarchaict1, 4),
         '2': np.round(trwdtgarchbict1, 4)},
        {'0': 'ARIMA(1,1,0)-GJR-GARCH-t(1,1) Model 1', '1': np.round(tdartgarchaict1, 4),
         '2': np.round(tdartgarchbict1, 4)},
          ]
mstable = pd.DataFrame(msdata)
print('')
print('== GARCH-t Model Selection ==')
print('')
print(mstable)
print('')

#########################################

# 5.9. GARCH-t Models Forecasting Accuracy

# 5.9.1. Multi-Steps Forecast
rwdmae1 = fa.meanabs(rwdf1, spyf)
rwdrmse1 = fa.rmse(rwdf1, spyf)
rwdtgarchmae1 = fa.meanabs(rwdtgarchf1, spyf)
rwdtgarchrmse1 = fa.rmse(rwdtgarchf1, spyf)
trwdgarchmae1 = fa.meanabs(trwdgarchf1, spyf)
trwdgarchrmse1 = fa.rmse(trwdgarchf1, spyf)
tdargarchmae1 = fa.meanabs(tdargarchf1, spyf)
tdargarchrmse1 = fa.rmse(tdargarchf1, spyf)
trwdegarchmae1 = fa.meanabs(trwdegarchf1, spyf)
trwdegarchrmse1 = fa.rmse(trwdegarchf1, spyf)
tdaregarchmae1 = fa.meanabs(tdaregarchf1, spyf)
tdaregarchrmse1 = fa.rmse(tdaregarchf1, spyf)
trwdtgarchmae1 = fa.meanabs(trwdtgarchf1, spyf)
trwdtgarchrmse1 = fa.rmse(trwdtgarchf1, spyf)
tdartgarchmae1 = fa.meanabs(tdartgarchf1, spyf)
tdartgarchrmse1 = fa.rmse(tdartgarchf1, spyf)

fadata1 = [{'0': '', '1': 'MAE', '2': 'RMSE'},
        {'0': 'ARIMA(0,1,0) Model 1', '1': np.round(rwdmae1, 4),
         '2': np.round(rwdrmse1, 4)},
        {'0': 'ARIMA(0,1,0)-GJR-GARCH(1,1) Model 1', '1': np.round(rwdtgarchmae1, 4),
         '2': np.round(rwdtgarchrmse1, 4)},
        {'0': 'ARIMA(0,1,0)-GARCH-t(1,1) Model 1', '1': np.round(trwdgarchmae1, 4),
         '2': np.round(trwdgarchrmse1, 4)},
        {'0': 'ARIMA(1,1,0)-GARCH-t(1,1) Model 1', '1': np.round(tdargarchmae1, 4),
         '2': np.round(tdargarchrmse1, 4)},
        {'0': 'ARIMA(0,1,0)-EGARCH-t(1,1) Model 1', '1': np.round(trwdegarchmae1, 4),
         '2': np.round(trwdegarchrmse1, 4)},
        {'0': 'ARIMA(1,1,0)-EGARCH-t(1,1) Model 1', '1': np.round(tdaregarchmae1, 4),
         '2': np.round(tdaregarchrmse1, 4)},
        {'0': 'ARIMA(0,1,0)-GJR-GARCH-t(1,1) Model 1', '1': np.round(trwdtgarchmae1, 4),
         '2': np.round(trwdtgarchrmse1, 4)},
        {'0': 'ARIMA(1,1,0)-GJR-GARCH-t(1,1) Model 1', '1': np.round(tdartgarchmae1, 4),
         '2': np.round(tdartgarchrmse1, 4)},
           ]
fatable1 = pd.DataFrame(fadata1)
print('')
print('== Multi-Steps Forecasting Accuracy ==')
print('')
print(fatable1)
print('')

# 5.9.2. One-Step Forecast without Re-Estimation
rwdmae2 = fa.meanabs(rwdf2, spyf)
rwdrmse2 = fa.rmse(rwdf2, spyf)
rwdtgarchmae2 = fa.meanabs(rwdtgarchf2, spyf)
rwdtgarchrmse2 = fa.rmse(rwdtgarchf2, spyf)
trwdgarchmae2 = fa.meanabs(trwdgarchf2, spyf)
trwdgarchrmse2 = fa.rmse(trwdgarchf2, spyf)
tdargarchmae2 = fa.meanabs(tdargarchf2, spyf)
tdargarchrmse2 = fa.rmse(tdargarchf2, spyf)
trwdegarchmae2 = fa.meanabs(trwdegarchf2, spyf)
trwdegarchrmse2 = fa.rmse(trwdegarchf2, spyf)
tdaregarchmae2 = fa.meanabs(tdaregarchf2, spyf)
tdaregarchrmse2 = fa.rmse(tdaregarchf2, spyf)
trwdtgarchmae2 = fa.meanabs(trwdtgarchf2, spyf)
trwdtgarchrmse2 = fa.rmse(trwdtgarchf2, spyf)
tdartgarchmae2 = fa.meanabs(tdartgarchf2, spyf)
tdartgarchrmse2 = fa.rmse(tdartgarchf2, spyf)

fadata2 = [{'0': '', '1': 'MAE', '2': 'RMSE'},
        {'0': 'ARIMA(0,1,0) Model 2', '1': np.round(rwdmae2, 4),
         '2': np.round(rwdrmse2, 4)},
        {'0': 'ARIMA(0,1,0)-GJR-GARCH(1,1) Model 2', '1': np.round(rwdtgarchmae2, 4),
         '2': np.round(rwdtgarchrmse2, 4)},
        {'0': 'ARIMA(0,1,0)-GARCH-t(1,1) Model 2', '1': np.round(trwdgarchmae2, 4),
         '2': np.round(trwdgarchrmse2, 4)},
        {'0': 'ARIMA(1,1,0)-GARCH-t(1,1) Model 2', '1': np.round(tdargarchmae2, 4),
         '2': np.round(tdargarchrmse2, 4)},
        {'0': 'ARIMA(0,1,0)-EGARCH-t(1,1) Model 2', '1': np.round(trwdegarchmae2, 4),
         '2': np.round(trwdegarchrmse2, 4)},
        {'0': 'ARIMA(1,1,0)-EGARCH-t(1,1) Model 2', '1': np.round(tdaregarchmae2, 4),
         '2': np.round(tdaregarchrmse2, 4)},
        {'0': 'ARIMA(0,1,0)-GJR-GARCH-t(1,1) Model 2', '1': np.round(trwdtgarchmae2, 4),
         '2': np.round(trwdtgarchrmse2, 4)},
        {'0': 'ARIMA(1,1,0)-GJR-GARCH-t(1,1) Model 2', '1': np.round(tdartgarchmae2, 4),
         '2': np.round(tdartgarchrmse2, 4)},
           ]
fatable2 = pd.DataFrame(fadata2)
print('')
print('== One-Step without Re-Estimation Forecasting Accuracy ==')
print('')
print(fatable2)
print('')

#########################################

# 5.10. Residuals White Noise

trwdegarcht1sres = trwdegarcht1.resid.dropna()/trwdegarcht1.conditional_volatility

# 5.10.1. Residuals No Auto-correlation

# 5.10.1.1. Auto-correlation Functions ACF
tsp.plot_acf(trwdegarcht1sres, lags=22, alpha=0.05)
plt.title('Autocorrelation Function ACF (trwdegarcht1sres)')
plt.show()

tsp.plot_acf(rwdtgarcht1sres, lags=22, alpha=0.05)
plt.title('Autocorrelation Function ACF (rwdtgarcht1sres)')
plt.show()

# 5.10.1.2. Ljung-Box Auto-correlation Tests
lbtrwdegarcht1sres = st.acorr_ljungbox(trwdegarcht1sres, lags=22)
lbrwdtgarcht1sres = st.acorr_ljungbox(rwdtgarcht1sres, lags=22)
print('')
print('== Ljung-Box Auto-Correlation Tests (trwdegarcht1sres, rwdtgarcht1sres) ==')
print('')
print('Ljung-Box Q Test Statistic P-Value (trwdegarcht1sres): ', np.round(lbtrwdegarcht1sres[1][21], 4))
print('Ljung-Box Q Test Statistic P-Value (rwdtgarcht1sres): ', np.round(lbrwdtgarcht1sres[1][21], 4))
print('')

# 5.10.2. Residuals Homoscedasticity

# 5.10.2.1. Auto Regressive Conditional Heteroscedasticity Tests
archtrwdegarcht1sres = st.het_arch(trwdegarcht1sres)
archrwdtgarcht1sres = st.het_arch(rwdtgarcht1sres)
print('')
print('== Auto Regressive Conditional Heteroscedasticity Tests (trwdegarcht1sres, rwdtgarcht1sres) ==')
print('')
print('ARCH Test Lagrange Multiplier P-Value (trwdegarcht1sres): ', np.round(archtrwdegarcht1sres[1], 4))
print('ARCH Test Lagrange Multiplier P-Value (rwdtgarcht1sres): ', np.round(archrwdtgarcht1sres[1], 4))
print('')

# 5.10.3. Residuals Normality

# 5.10.3.1. Jarque-Bera Normality Tests
jbtrwdegarcht1sres = jb.jarque_bera(trwdegarcht1sres)
jbrwdtgarcht1sres = jb.jarque_bera(rwdtgarcht1sres)
print('')
print('== Jarque-Bera Normality Tests (trwdegarcht1sres, rwdtgarcht1sres) ==')
print('')
print('Jarque-Bera Test Chi-Squared P-Value (trwdegarcht1sres): ', np.round(jbtrwdegarcht1sres[1], 4))
print('Jarque-Bera Test Chi-Squared P-Value (rwdtgarcht1sres): ', np.round(jbrwdtgarcht1sres[1], 4))
print('')