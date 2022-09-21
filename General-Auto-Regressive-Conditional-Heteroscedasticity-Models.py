##############################################################################
# Advanced Forecasting Models with Python                                    #
# General Auto Regressive Conditional Heteroscedasticity Models              #
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

# 4. General Auto Regressive Conditional Heteroscedasticity Models

# 4.1. Second Order Stationarity

# 4.1.1. Auto Regressive Conditional Heteroscedasticity Test
rwdt1res = rwdt1.resid.tail(len(spyt)-1)
archrwdt1 = st.het_arch(rwdt1res)
print('')
print('== Auto Regressive Conditional Heteroscedasticity Test (arwdt1) ==')
print('')
print('ARCH Test Lagrange Multiplier P-Value: ', np.round(archrwdt1[1], 4))
print('')

# 4.1.2. GARCH Models Specification

# 4.1.2.1. Auto-correlation Function ACF
tsp.plot_acf(rwdt1res**2, lags=22, alpha=0.05)
plt.title('Autocorrelation Function ACF (rwdt1.resid^2)')
plt.show()

# 4.1.2.2. Partial Auto-correlation Function PACF
tsp.plot_pacf(rwdt1res**2, lags=22, alpha=0.05)
plt.title('Partial Autocorrelation Function ACF (rwdt1.resid^2)')
plt.show()

# 4.1.3. Time Series Level Differencing
dspy = spy-spy.shift(1)
dspy = dspy.fillna(method='bfill')
dspyt = dspy[:'2013-12-31']
dspyf = dspy['2014-01-02':]

##########################################

# 4.2. ARIMA(0,1,0)-GARCH(1,1)

# 4.2.1. Multi-Steps Forecast
rwdgarcht1 = arch.ConstantMean(y=dspyt, volatility=arch.GARCH(p=1, o=0, q=1, power=2.0),
                                distribution=arch.Normal()).fit()
print('')
print('== ARIMA(0,1,0)-GARCH(1,1) Model (dspyt) ==')
print('')
print(rwdgarcht1.summary())
print('')
rwdgarchf1 = pd.DataFrame(pd.concat([spyt.tail(1)]*len(spyf))).set_index(spyf.index)
rwdgarchf1['SPY.CondDrift'] = pd.DataFrame(pd.concat(
    [rwdgarcht1.forecast().mean.tail(1)]*len(spyf))).set_index(spyf.index).cumsum()
rwdgarchf1 = rwdgarchf1.sum(axis=1)
rwdgarchf1 = pd.DataFrame(rwdgarchf1).set_index(spyf.index)

fig7, ax = plt.subplots()
ax.plot(spyt, label='spyt')
ax.plot(spyf, label='spyf')
ax.plot(rwdgarchf1, label='rwdgarchf1')
plt.legend(loc='upper left')
plt.title('ARIMA(0,1,0)-GARCH(1,1) Model 1')
plt.ylabel('Price')
plt.xlabel('Date')
plt.show()

# 4.2.2. One-Step Forecast without Re-Estimation
rwdgarchf2 = spy.shift(1)['2014-01-02':]
rwdgarchf2['SPY.CondMean'] = pd.DataFrame(pd.concat(
    [rwdgarcht1.forecast().mean.tail(1)]*len(spyf))).set_index(spyf.index)
rwdgarchf2 = rwdgarchf2.sum(axis=1)
rwdgarchf2 = pd.DataFrame(rwdgarchf2).set_index(spyf.index)

fig8, ax = plt.subplots()
ax.plot(spyt, label='spyt')
ax.plot(spyf, label='spyf')
ax.plot(rwdgarchf2, label='rwdgarchf2', linestyle=':')
plt.legend(loc='upper left')
plt.title('ARIMA(0,1,0)-GARCH(1,1) Model 2')
plt.ylabel('Price')
plt.xlabel('Date')
plt.show()

##########################################

# 4.3. ARIMA(1,1,0)-GARCH(1,1)

# 4.3.1. Multi-Steps Forecast
dargarcht1 = arch.ARX(y=dspyt, volatility=arch.GARCH(p=1, o=0, q=1, power=2.0), lags=[1], constant=True,
                       distribution=arch.Normal()).fit()
print('')
print('== ARIMA(1,1,0)-GARCH(1,1) Model (dspyt) ==')
print('')
print(dargarcht1.summary())
print('')
dargarchf1 = pd.DataFrame(pd.concat([dargarcht1.params[1]*dspyt.tail(1)]*len(spyf))).set_index(spyf.index)
dargarchf1['SPY.LastAdj'] = pd.DataFrame(pd.concat([spyt.tail(1)]*len(spyf))).set_index(spyf.index)
dargarchf1['SPY.CondDrift'] = pd.DataFrame(pd.concat(
    [dargarcht1.forecast().mean.tail(1)]*len(spyf))).set_index(spyf.index).cumsum()
dargarchf1 = dargarchf1.sum(axis=1)
dargarchf1 = pd.DataFrame(dargarchf1).set_index(spyf.index)

fig9, ax = plt.subplots()
ax.plot(spyt, label='spyt')
ax.plot(spyf, label='spyf')
ax.plot(dargarchf1, label='dargarchf1')
plt.legend(loc='upper left')
plt.title('ARIMA(1,1,0)-GARCH(1,1) Model 1')
plt.ylabel('Price')
plt.xlabel('Date')
plt.show()

# 4.3.2. One-Step Forecast without Re-Estimation
dargarchf2 = dspy.shift(1)['2014-01-02':]
dargarchf2 = dargarchf2*dargarcht1.params[1]
dargarchf2['SPY.Adjusted(-1)'] = spy.shift(1)['2014-01-02':]
dargarchf2['SPY.CondMean'] = pd.DataFrame(pd.concat(
    [dargarcht1.forecast().mean.tail(1)]*len(spyf))).set_index(spyf.index)
dargarchf2 = dargarchf2.sum(axis=1)
dargarchf2 = pd.DataFrame(dargarchf2).set_index(spyf.index)

fig10, ax = plt.subplots()
ax.plot(spyt, label='spyt')
ax.plot(spyf, label='spyf')
ax.plot(dargarchf2, label='dargarchf2', linestyle=':')
plt.legend(loc='upper left')
plt.title('ARIMA(1,1,0)-GARCH(1,1) Model 2')
plt.ylabel('Price')
plt.xlabel('Date')
plt.show()

##########################################

# 4.4. ARIMA(0,1,0)-EGARCH(1,1)

# 4.4.1. Multi-Steps Forecast
rwdegarcht1 = arch.ConstantMean(y=dspyt, volatility=arch.EGARCH(p=1, o=1, q=1),
                                distribution=arch.Normal()).fit()
print('')
print('== ARIMA(0,1,0)-EGARCH(1,1) Model (dspyt) ==')
print('')
print(rwdegarcht1.summary())
print('')
rwdegarchf1 = pd.DataFrame(pd.concat([spyt.tail(1)]*len(spyf))).set_index(spyf.index)
rwdegarchf1['SPY.CondDrift'] = pd.DataFrame(pd.concat(
    [rwdegarcht1.forecast().mean.tail(1)]*len(spyf))).set_index(spyf.index).cumsum()
rwdegarchf1 = rwdegarchf1.sum(axis=1)
rwdegarchf1 = pd.DataFrame(rwdegarchf1).set_index(spyf.index)

fig11, ax = plt.subplots()
ax.plot(spyt, label='spyt')
ax.plot(spyf, label='spyf')
ax.plot(rwdegarchf1, label='rwdegarchf1')
plt.legend(loc='upper left')
plt.title('ARIMA(0,1,0)-EGARCH(1,1) Model 1')
plt.ylabel('Price')
plt.xlabel('Date')
plt.show()

# 4.4.2. One-Step Forecast without Re-Estimation
rwdegarchf2 = spy.shift(1)['2014-01-02':]
rwdegarchf2['SPY.CondMean'] = pd.DataFrame(pd.concat(
    [rwdegarcht1.forecast().mean.tail(1)]*len(spyf))).set_index(spyf.index)
rwdegarchf2 = rwdegarchf2.sum(axis=1)
rwdegarchf2 = pd.DataFrame(rwdegarchf2).set_index(spyf.index)

fig12, ax = plt.subplots()
ax.plot(spyt, label='spyt')
ax.plot(spyf, label='spyf')
ax.plot(rwdegarchf2, label='rwdegarchf2', linestyle=':')
plt.legend(loc='upper left')
plt.title('ARIMA(0,1,0)-EGARCH(1,1) Model 2')
plt.ylabel('Price')
plt.xlabel('Date')
plt.show()

##########################################

# 4.5. ARIMA(1,1,0)-EGARCH(1,1)

# 4.5.1. Multi-Steps Forecast
daregarcht1 = arch.ARX(y=dspyt, volatility=arch.EGARCH(p=1, o=1, q=1), lags=[1], constant=True,
                       distribution=arch.Normal()).fit()
print('')
print('== ARIMA(1,1,0)-EGARCH(1,1) Model (dspyt) ==')
print('')
print(daregarcht1.summary())
print('')
daregarchf1 = pd.DataFrame(pd.concat([daregarcht1.params[1]*dspyt.tail(1)]*len(spyf))).set_index(spyf.index)
daregarchf1['SPY.LastAdj'] = pd.DataFrame(pd.concat([spyt.tail(1)]*len(spyf))).set_index(spyf.index)
daregarchf1['SPY.CondDrift'] = pd.DataFrame(pd.concat(
    [daregarcht1.forecast().mean.tail(1)]*len(spyf))).set_index(spyf.index).cumsum()
daregarchf1 = daregarchf1.sum(axis=1)
daregarchf1 = pd.DataFrame(daregarchf1).set_index(spyf.index)

fig13, ax = plt.subplots()
ax.plot(spyt, label='spyt')
ax.plot(spyf, label='spyf')
ax.plot(daregarchf1, label='daregarchf1')
plt.legend(loc='upper left')
plt.title('ARIMA(1,1,0)-EGARCH(1,1) Model 1')
plt.ylabel('Price')
plt.xlabel('Date')
plt.show()

# 4.5.2. One-Step Forecast without Re-Estimation
daregarchf2 = dspy.shift(1)['2014-01-02':]
daregarchf2 = daregarchf2*daregarcht1.params[1]
daregarchf2['SPY.Adjusted(-1)'] = spy.shift(1)['2014-01-02':]
daregarchf2['SPY.CondMean'] = pd.DataFrame(pd.concat(
    [daregarcht1.forecast().mean.tail(1)]*len(spyf))).set_index(spyf.index)
daregarchf2 = daregarchf2.sum(axis=1)
daregarchf2 = pd.DataFrame(daregarchf2).set_index(spyf.index)

fig14, ax = plt.subplots()
ax.plot(spyt, label='spyt')
ax.plot(spyf, label='spyf')
ax.plot(daregarchf2, label='daregarchf2', linestyle=':')
plt.legend(loc='upper left')
plt.title('ARIMA(1,1,0)-EGARCH(1,1) Model 2')
plt.ylabel('Price')
plt.xlabel('Date')
plt.show()

##########################################

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

fig15, ax = plt.subplots()
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

fig16, ax = plt.subplots()
ax.plot(spyt, label='spyt')
ax.plot(spyf, label='spyf')
ax.plot(rwdtgarchf2, label='rwdtgarchf2', linestyle=':')
plt.legend(loc='upper left')
plt.title('ARIMA(0,1,0)-GJR-GARCH(1,1) Model 2')
plt.ylabel('Price')
plt.xlabel('Date')
plt.show()

##########################################

# 4.7. ARIMA(1,1,0)-GJR-GARCH(1,1)

# 4.7.1. Multi-Steps Forecast
dartgarcht1 = arch.ARX(y=dspyt, volatility=arch.GARCH(p=1, o=1, q=1), lags=[1], constant=True,
                       distribution=arch.Normal()).fit()
print('')
print('== ARIMA(1,1,0)-GJR-GARCH(1,1) Model (dspyt) ==')
print('')
print(dartgarcht1.summary())
print('')
dartgarchf1 = pd.DataFrame(pd.concat([dartgarcht1.params[1]*dspyt.tail(1)]*len(spyf))).set_index(spyf.index)
dartgarchf1['SPY.LastAdj'] = pd.DataFrame(pd.concat([spyt.tail(1)]*len(spyf))).set_index(spyf.index)
dartgarchf1['SPY.CondDrift'] = pd.DataFrame(pd.concat(
    [dartgarcht1.forecast().mean.tail(1)]*len(spyf))).set_index(spyf.index).cumsum()
dartgarchf1 = dartgarchf1.sum(axis=1)
dartgarchf1 = pd.DataFrame(dartgarchf1).set_index(spyf.index)

fig17, ax = plt.subplots()
ax.plot(spyt, label='spyt')
ax.plot(spyf, label='spyf')
ax.plot(dartgarchf1, label='dartgarchf1')
plt.legend(loc='upper left')
plt.title('ARIMA(1,1,0)-GJR-GARCH(1,1) Model 1')
plt.ylabel('Price')
plt.xlabel('Date')
plt.show()

# 4.7.2. One-Step Forecast without Re-Estimation
dartgarchf2 = dspy.shift(1)['2014-01-02':]
dartgarchf2 = dartgarchf2*dartgarcht1.params[1]
dartgarchf2['SPY.Adjusted(-1)'] = spy.shift(1)['2014-01-02':]
dartgarchf2['SPY.CondMean'] = pd.DataFrame(pd.concat(
    [dartgarcht1.forecast().mean.tail(1)]*len(spyf))).set_index(spyf.index)
dartgarchf2 = dartgarchf2.sum(axis=1)
dartgarchf2 = pd.DataFrame(dartgarchf2).set_index(spyf.index)

fig18, ax = plt.subplots()
ax.plot(spyt, label='spyt')
ax.plot(spyf, label='spyf')
ax.plot(dartgarchf2, label='dartgarchf2', linestyle=':')
plt.legend(loc='upper left')
plt.title('ARIMA(1,1,0)-GJR-GARCH(1,1) Model 2')
plt.ylabel('Price')
plt.xlabel('Date')
plt.show()

#########################################

# 4.8. GARCH Model Selection

rwdgarchaict1 = rwdgarcht1.aic
rwdgarchbict1 = rwdgarcht1.bic
dargarchaict1 = dargarcht1.aic
dargarchbict1 = dargarcht1.bic
rwdegarchaict1 = rwdegarcht1.aic
rwdegarchbict1 = rwdegarcht1.bic
daregarchaict1 = daregarcht1.aic
daregarchbict1 = daregarcht1.bic
rwdtgarchaict1 = rwdtgarcht1.aic
rwdtgarchbict1 = rwdtgarcht1.bic
dartgarchaict1 = dartgarcht1.aic
dartgarchbict1 = dartgarcht1.bic

msdata = [{'0': '', '1': 'AIC', '2': 'BIC'},
        {'0': 'ARIMA(0,1,0)-GARCH(1,1) Model 1', '1': np.round(rwdgarchaict1, 4),
         '2': np.round(rwdgarchbict1, 4)},
        {'0': 'ARIMA(1,1,0)-GARCH(1,1) Model 1', '1': np.round(dargarchaict1, 4),
         '2': np.round(dargarchbict1, 4)},
        {'0': 'ARIMA(0,1,0)-EGARCH(1,1) Model 1', '1': np.round(rwdegarchaict1, 4),
         '2': np.round(rwdegarchbict1, 4)},
        {'0': 'ARIMA(1,1,0)-EGARCH(1,1) Model 1', '1': np.round(daregarchaict1, 4),
         '2': np.round(daregarchbict1, 4)},
        {'0': 'ARIMA(0,1,0)-GJR-GARCH(1,1) Model 1', '1': np.round(rwdtgarchaict1, 4),
         '2': np.round(rwdtgarchbict1, 4)},
        {'0': 'ARIMA(1,1,0)-GJR-GARCH(1,1) Model 1', '1': np.round(dartgarchaict1, 4),
         '2': np.round(dartgarchbict1, 4)},
          ]
mstable = pd.DataFrame(msdata)
print('')
print('== GARCH Model Selection ==')
print('')
print(mstable)
print('')

#########################################

# 4.9. GARCH Models Forecasting Accuracy

# 4.9.1. Multi-Steps Forecast
rwdmae1 = fa.meanabs(rwdf1, spyf)
rwdrmse1 = fa.rmse(rwdf1, spyf)
rwdgarchmae1 = fa.meanabs(rwdgarchf1, spyf)
rwdgarchrmse1 = fa.rmse(rwdgarchf1, spyf)
dargarchmae1 = fa.meanabs(dargarchf1, spyf)
dargarchrmse1 = fa.rmse(dargarchf1, spyf)
rwdegarchmae1 = fa.meanabs(rwdegarchf1, spyf)
rwdegarchrmse1 = fa.rmse(rwdegarchf1, spyf)
daregarchmae1 = fa.meanabs(daregarchf1, spyf)
daregarchrmse1 = fa.rmse(daregarchf1, spyf)
rwdtgarchmae1 = fa.meanabs(rwdtgarchf1, spyf)
rwdtgarchrmse1 = fa.rmse(rwdtgarchf1, spyf)
dartgarchmae1 = fa.meanabs(dartgarchf1, spyf)
dartgarchrmse1 = fa.rmse(dartgarchf1, spyf)

fadata1 = [{'0': '', '1': 'MAE', '2': 'RMSE'},
        {'0': 'ARIMA(0,1,0) Model 1', '1': np.round(rwdmae1, 4),
         '2': np.round(rwdrmse1, 4)},
        {'0': 'ARIMA(0,1,0)-GARCH(1,1) Model 1', '1': np.round(rwdgarchmae1, 4),
         '2': np.round(rwdgarchrmse1, 4)},
        {'0': 'ARIMA(1,1,0)-GARCH(1,1) Model 1', '1': np.round(dargarchmae1, 4),
         '2': np.round(dargarchrmse1, 4)},
        {'0': 'ARIMA(0,1,0)-EGARCH(1,1) Model 1', '1': np.round(rwdegarchmae1, 4),
         '2': np.round(rwdegarchrmse1, 4)},
        {'0': 'ARIMA(1,1,0)-EGARCH(1,1) Model 1', '1': np.round(daregarchmae1, 4),
         '2': np.round(daregarchrmse1, 4)},
        {'0': 'ARIMA(0,1,0)-GJR-GARCH(1,1) Model 1', '1': np.round(rwdtgarchmae1, 4),
         '2': np.round(rwdtgarchrmse1, 4)},
        {'0': 'ARIMA(1,1,0)-GJR-GARCH(1,1) Model 1', '1': np.round(dartgarchmae1, 4),
         '2': np.round(dartgarchrmse1, 4)},
           ]
fatable1 = pd.DataFrame(fadata1)
print('')
print('== Multi-Steps Forecasting Accuracy ==')
print('')
print(fatable1)
print('')

# 4.9.2. One-Step Forecast without Re-Estimation
rwdmae2 = fa.meanabs(rwdf2, spyf)
rwdrmse2 = fa.rmse(rwdf2, spyf)
rwdgarchmae2 = fa.meanabs(rwdgarchf2, spyf)
rwdgarchrmse2 = fa.rmse(rwdgarchf2, spyf)
dargarchmae2 = fa.meanabs(dargarchf2, spyf)
dargarchrmse2 = fa.rmse(dargarchf2, spyf)
rwdegarchmae2 = fa.meanabs(rwdegarchf2, spyf)
rwdegarchrmse2 = fa.rmse(rwdegarchf2, spyf)
daregarchmae2 = fa.meanabs(daregarchf2, spyf)
daregarchrmse2 = fa.rmse(daregarchf2, spyf)
rwdtgarchmae2 = fa.meanabs(rwdtgarchf2, spyf)
rwdtgarchrmse2 = fa.rmse(rwdtgarchf2, spyf)
dartgarchmae2 = fa.meanabs(dartgarchf2, spyf)
dartgarchrmse2 = fa.rmse(dartgarchf2, spyf)

fadata2 = [{'0': '', '1': 'MAE', '2': 'RMSE'},
        {'0': 'ARIMA(0,1,0) Model 2', '1': np.round(rwdmae2, 4),
         '2': np.round(rwdrmse2, 4)},
        {'0': 'ARIMA(0,1,0)-GARCH(1,1) Model 2', '1': np.round(rwdgarchmae2, 4),
         '2': np.round(rwdgarchrmse2, 4)},
        {'0': 'ARIMA(1,1,0)-GARCH(1,1) Model 2', '1': np.round(dargarchmae2, 4),
         '2': np.round(dargarchrmse2, 4)},
        {'0': 'ARIMA(0,1,0)-EGARCH(1,1) Model 2', '1': np.round(rwdegarchmae2, 4),
         '2': np.round(rwdegarchrmse2, 4)},
        {'0': 'ARIMA(1,1,0)-EGARCH(1,1) Model 2', '1': np.round(daregarchmae2, 4),
         '2': np.round(daregarchrmse2, 4)},
        {'0': 'ARIMA(0,1,0)-GJR-GARCH(1,1) Model 2', '1': np.round(rwdtgarchmae2, 4),
         '2': np.round(rwdtgarchrmse2, 4)},
        {'0': 'ARIMA(1,1,0)-GJR-GARCH(1,1) Model 2', '1': np.round(dartgarchmae2, 4),
         '2': np.round(dartgarchrmse2, 4)},
           ]
fatable2 = pd.DataFrame(fadata2)
print('')
print('== One-Step without Re-Estimation Forecasting Accuracy ==')
print('')
print(fatable2)
print('')
