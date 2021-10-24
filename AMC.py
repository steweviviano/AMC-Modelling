# we are writing the backhand of the streamlet platform

import numpy as np
import streamlit as st
import yfinance as yf
import pandas as pd
import datetime as dt
from pandas_datareader import data as pdr
import scipy as sc
from scipy.optimize import minimize
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, DayLocator
pd.options.mode.chained_assignment = None

st.sidebar.write(" PERCENTO.CAPITAL ")
option = st.sidebar.selectbox("which dashboard?", ('stocks', 'ETF', 'SP500 Strategy', 'Portfolio'))

if option == 'stocks':
    st.title('Asset Management Challenge')
    tickers = (['RDS-B','BP','MPC','TTE','OGZPY','OJSCY','XOM','CVX','OAOFY', 'CCJ', 'GS','EVR', 'JPM', 'C', 'MS', 'MC', 'SNY', 
                'GILD', 'ALV.DE', 'ALLY', 'F'])

    dropdown = st.multiselect('Pick your assets',tickers)
    start = st.date_input('Start', value = pd.to_datetime('2021-01-01'))
    end = st.date_input('End', value = pd.to_datetime('today'))

    def relativeret(df):
        rel = df.pct_change()
        cumret = (1+rel).cumprod() -1
        cumret = cumret.fillna(0) # the cumulative return the first day would be NaN so we start from zero
        return cumret




    if len(dropdown) > 0:
        #df = yf.download(dropdown,start,end)['Adj Close']
        df = relativeret(yf.download(dropdown,start,end)['Adj Close'])

        st.line_chart(df)



    
if option == 'ETF':
    ETFs = (['IWDA.L', 'EMXC','CGW','IUMS.L','XAAG.F','XDWF.L','MOBI.L','D6RQ.DE','SUJP.SW'])
    dropdown_etf = st.multiselect('Pick your ETF',ETFs)
    start_etf = st.date_input('Start', value = pd.to_datetime('2021-01-01'))
    end_etf= st.date_input('End', value = pd.to_datetime('today'))


    def relativeretetf(df_etf):
        reletf = df_etf.pct_change()
        cumretetf = (1+reletf).cumprod() -1
        cumretetf = cumretetf.fillna(0) # the cumulative return the first day would be NaN so we start from zero
        return cumretetf

    if len(dropdown_etf) > 0:
    #df = yf.download(dropdown,start,end)['Adj Close']
        df_etf = relativeretetf(yf.download(dropdown_etf,start_etf,end_etf)['Adj Close'])
        st.line_chart(df_etf)
if option == 'Portfolio':
    def portfolioPerformance(weights, meanReturns, covMatrix):
        returns = np.sum(meanReturns*weights)
        std = np.sqrt(
                np.dot(weights.T,np.dot(covMatrix, weights))
            )
        return returns, std

    def negativeSR(weights, meanReturns, covMatrix, riskFreeRate = 0):
        pReturns, pStd = portfolioPerformance(weights, meanReturns, covMatrix)
        return - (pReturns - riskFreeRate)/pStd

    def maxSR(meanReturns, covMatrix, riskFreeRate = 0, constraintSet=(0,1)):
        numAssets = len(meanReturns)
        args = (meanReturns, covMatrix, riskFreeRate)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bound = constraintSet
        bounds = ((0.01,0.05), (0.01,0.05), (0.01,0.05), (0.01,0.05), (0.01,0.05), (0.01,0.05), (0.01,0.05), (0.01,0.05), (0.01,0.05), (0.01,0.05), (0.01,0.05), (0.01,0.05), (0.01,0.05), (0.01,0.05), (0.01,0.05), (0.01,0.05), (0.01,0.05), (0.4, 0.5) , (0.01,0.05), (0.01,0.05), (0.01,0.05),(0.01,0.05), (0.01,0.05), (0.01,0.05), (0.01,0.05),(0.01,0.05), (0.01,0.05), (0.01,0.05), (0.01,0.05), (0.01,0.05), (0.01,0.05), (0.01,0.05))
        result = minimize(negativeSR, numAssets*[1./numAssets], args=args,
                            method='SLSQP', bounds=bounds, constraints=constraints)
        return result

    def portfolioVariance(weights, meanReturns, covMatrix):
        return portfolioPerformance(weights, meanReturns, covMatrix)[1]

    def minimizeVariance(meanReturns, covMatrix, constraintSet=(0,1)):
        numAssets = len(meanReturns)
        args = (meanReturns, covMatrix)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bound = constraintSet
        bounds = ((0.01,0.05), (0.01,0.05), (0.01,0.05), (0.01,0.05), (0.01,0.05), (0.01,0.05), (0.01,0.05), (0.01,0.05), (0.01,0.05), (0.01,0.05), (0.01,0.05), (0.01,0.05), (0.01,0.05), (0.01,0.05), (0.01,0.05), (0.01,0.05), (0.01,0.05), (0.4, 0.5) , (0.01,0.05), (0.01,0.05), (0.01,0.05),(0.01,0.05), (0.01,0.05), (0.01,0.05), (0.01,0.05),(0.01,0.05), (0.01,0.05), (0.01,0.05), (0.01,0.05), (0.01,0.05), (0.01,0.05), (0.01,0.05))
        result = minimize(portfolioVariance, numAssets*[1./numAssets], args=args,
                            method='SLSQP', bounds=bounds, constraints=constraints)
        return result
    def portfolioReturn(weights, meanReturns, covMatrix):
        return portfolioPerformance(weights, meanReturns, covMatrix)[0]

    def efficientOpt(meanReturns, covMatrix, returnTarget, constraintSet=(0,1)):
        numAssets = len(meanReturns)
        args = (meanReturns, covMatrix)

        constraints = ({'type':'eq', 'fun': lambda x: portfolioReturn(x, meanReturns, covMatrix) - returnTarget},
                        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bound = ((0.01,0.05), (0.01,0.05), (0.01,0.05), (0.01,0.05), (0.01,0.05), (0.01,0.05), (0.01,0.05), (0.01,0.05), (0.01,0.05), (0.01,0.05), (0.01,0.05), (0.01,0.05), (0.01,0.05), (0.01,0.05), (0.01,0.05), (0.01,0.05), (0.01,0.05), (0.4, 0.5) , (0.01,0.05), (0.01,0.05), (0.01,0.05),(0.01,0.05), (0.01,0.05), (0.01,0.05), (0.01,0.05),(0.01,0.05), (0.01,0.05), (0.01,0.05), (0.01,0.05), (0.01,0.05), (0.01,0.05), (0.01,0.05))
        bounds = tuple(bound for asset in range(numAssets))
        effOpt = minimize(portfolioVariance, numAssets*[1./numAssets], args=args, method = 'SLSQP', bounds=bound, constraints=constraints)
        return effOpt

    def calculatedResults(meanReturns, covMatrix, riskFreeRate=0, constraintSet=(0,1)):
         # Max Sharpe Ratio Portfolio
        maxSR_Portfolio = maxSR(meanReturns, covMatrix)
        maxSR_returns, maxSR_std = portfolioPerformance(maxSR_Portfolio['x'], meanReturns, covMatrix)
        #maxSR_returns, maxSR_std = round(maxSR_returns*100,2), round(maxSR_std*100,2)
        maxSR_allocation = pd.DataFrame(maxSR_Portfolio['x'], index=meanReturns.index, columns=['allocation'])
        #maxSR_allocation.allocation = [round(i*100,0) for i in maxSR_allocation.allocation]
        
        # Min Volatility Portfolio
        minVol_Portfolio = minimizeVariance(meanReturns, covMatrix)
        minVol_returns, minVol_std = portfolioPerformance(minVol_Portfolio['x'], meanReturns, covMatrix)
        #minVol_returns, minVol_std = round(minVol_returns*100,2), round(minVol_std*100,2)
        minVol_allocation = pd.DataFrame(minVol_Portfolio['x'], index=meanReturns.index, columns=['allocation'])
        #minVol_allocation.allocation = [round(i*100,0) for i in minVol_allocation.allocation]

        # Efficient Frontier
        efficientList = []
        targetReturns = np.linspace(minVol_returns, maxSR_returns, 20)
        for target in targetReturns:
            efficientList.append(efficientOpt(meanReturns, covMatrix, target)['fun'])

        return maxSR_returns, maxSR_std, maxSR_allocation, minVol_returns, minVol_std, minVol_allocation, efficientList,targetReturns
    def EF_graph(meanReturns, covMatrix, riskFreeRate=0, constraintSet=(0,1)):
        maxSR_returns, maxSR_std, maxSR_allocation, minVol_returns, minVol_std, minVol_allocation, efficientList, targetReturns = calculatedResults(meanReturns, covMatrix, riskFreeRate, constraintSet)

        #Max SR
        MaxSharpeRatio = go.Scatter(
            name='Maximium Sharpe Ratio',
            mode='markers',
            x=[maxSR_std],
            y=[maxSR_returns],
            marker=dict(color='red',size=14,line=dict(width=3, color='black'))
        )

        #Min Vol
        MinVol = go.Scatter(
            name='Mininium Volatility',
            mode='markers',
            x=[minVol_std],
            y=[minVol_returns],
            marker=dict(color='green',size=14,line=dict(width=3, color='black'))
        )

        #Efficient Frontier
        EF_curve = go.Scatter(
            name='Efficient Frontier',
            mode='lines',
            #x=[round(ef_std, 2) for ef_std in efficientList],
            y=targetReturns,
            x=efficientList,
            
            #y=[round(target, 2) for target in targetReturns],
            line=dict(color='black', width=4, dash='dashdot')
        )

        data = [MaxSharpeRatio, MinVol, EF_curve]

        layout = go.Layout(
            title = 'Portfolio Optimisation with the Efficient Frontier',
            yaxis = dict(title='Annualised Return (%)'),
            xaxis = dict(title='Annualised Volatility (%)'),
            showlegend = True,
            legend = dict(
                x = 0.75, y = 0, traceorder='normal',
                bgcolor='#E2E2E2',
                bordercolor='black',
                borderwidth=2),
            width=800,
            height=600)
        
        fig = go.Figure(data=data, layout=layout)
        return fig.show()
    
    portfolio_info = (['maximum sharpe ratio', 'minimum variance', 'equally wheighted', 'benchmark'])
    st.title('Portfolio Strategies')
    dropdown_pf = st.multiselect('Pick your portfolio strategy',portfolio_info)
    
    start_pf = st.date_input('Start', value = pd.to_datetime('2021-01-01'))
    end_pf = st.date_input('End', value = pd.to_datetime('today'))
    st.title('Portfolio')
    tickers_pf = ( ['RDS-B','BP','MPC','TTE','OGZPY','XOM','CVX','OAOFY','NVTK.IL','ATAD.IL','CCj','LIN','CLF','AA','FCX','ADM','BHP','SBSW','GS','EVR','JPM','C','MS','MC','F','BRK-B','IWDA.L', 'EMXC','CGW','IUMS.L','XDWF.L','SUJP.SW' ])



        #df = yf.download(dropdown,start,end)['Adj Close']
    df_pf = yf.download(tickers_pf,start_pf,end_pf)['Adj Close']
    df_pf = df_pf.resample('M').last()
    #stockData = stockData.dropna()
    df_pf = df_pf.pct_change()[1:]
    covMatrix = df_pf.cov()
    arr=[]
    maxsr_return=[]
    maxsr_sd=[]
    minvar_return=[]
    minvar_sd=[]
    SR_maxsr = []
    SR_minvar = []
    TrackingError_M = []

    for ind in df_pf.index:
        arr.append(np.array(df_pf.loc[ind]))
    for i in range(len(arr)):
        maxsr_i=portfolioPerformance(maxSR(arr[i],covMatrix)['x'],arr[i],covMatrix)
        minvar_i=portfolioPerformance(minimizeVariance(arr[i],covMatrix)['x'],arr[i],covMatrix)
        maxsr_return.append(maxsr_i[0])
        maxsr_sd.append(maxsr_i[1])
        minvar_return.append(minvar_i[0])
        minvar_sd.append(minvar_i[1])
        SR_maxsr.append(maxsr_return[i]/maxsr_sd[i])
        SR_minvar.append(minvar_return[i]/minvar_sd[i])
        #TrackingError_M.append(maxsr_return[i]-stockData['IWDA.L'][i])
    df_pf['Returns MaxSR']=maxsr_return
    df_pf['Returns Equally Weighted']=df_pf.mean(axis=1)
    df_pf['Returns MaxSR']=maxsr_return
    df_pf['StDev MaxSR'] = maxsr_sd
    df_pf['Sharpe Ratio MaxSR'] = SR_maxsr
    df_pf['Returns MinVar']=minvar_return
    df_pf['StDev MinVar'] = minvar_sd
    df_pf['Sharpe Ratio MinVar'] = SR_minvar
    #df_pf['Monthly Tracking Error'] = TrackingError_M
    QWE = df_pf['Returns MaxSR'].cumsum()
    ASD = df_pf['Returns Equally Weighted'].cumsum()
    ZXC = df_pf['Returns MinVar'].cumsum()
    if len(dropdown_pf) > 0:
        portfolio=pd.DataFrame()
        for i in dropdown_pf:
            if(i=='maximum sharpe ratio'):
                portfolio['maximum sharpe ratio']=QWE
            elif(i=='minimum variance'):
                portfolio['minimum variance']=ASD
            elif(i=='equally wheighted'):
                portfolio['equally wheighted']=ZXC
            elif(i=='benchmark'):
                portfolio['benchmark']=df_pf['IWDA.L'].cumsum()

        st.line_chart(portfolio)

if option == 'SP500 Strategy':
    tickers_bt = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    tickers_bt = tickers_bt.Symbol.to_list()
    tickers_bt = [i.replace('.','-') for i in tickers_bt]
    
    
    def RSIcalc(asset):
       # for i in len(asset):
       #     asset[i]=str(asset[2:-2])
      #  st.write(asset)
        df = yf.download(str(asset[2:-2]),str(start_bt))
        st.write(df)
        df['MA200'] = df['Adj Close'].rolling(window=200).mean() # here we create the 200 days MA
        df['price change'] = df['Adj Close'].pct_change()
        df['Upmove'] = df['price change'].apply(lambda x: x if x > 0 else 0)
        df['Downmove'] = df['price change'].apply(lambda x: abs(x) if x <0 else 0)
        df['avg Up'] = df['Upmove'].ewm(span=19).mean()
        df['avg Down'] = df['Downmove'].ewm(span=19).mean()
        df = df.dropna()
        df['RS'] = df['avg Up']/df['avg Down']
        df['RSI'] = df['RS'].apply(lambda x: 100-(100/(x+1)))
        df.loc[(df['Adj Close'] > df['MA200']) & (df['RSI'] < 30), 'Buy']= 'Yes'
        df.loc[(df['Adj Close'] < df['MA200']) & (df['RSI'] > 30), 'Buy']= 'No'
        df['ticker'] = asset
        return df

    def getSignals(df):
        Buying_dates = []
        Selling_dates = []
        
        for i in range(len(df) - 11):  #we have to subtract the lenght of j + 1
            if "Yes" in str(df['Buy'].iloc[i]):
                Buying_dates.append(df.iloc[i+1].name)
                Selling_dates.append(df.iloc[i+21].name)
               # for j in range(1,11):
                #    if df['RSI'].iloc[i + j] > 40:
                #        Selling_dates.append(df.iloc[i+j+1].name)
                #        break
                 #   elif j == 10: #if the RSI doesn't cross 40 we are selling after 10 days
                #        Selling_dates.append(df.iloc[i+j+1].name)
                        
            return Buying_dates,Selling_dates
        
    dropdown_bt = st.multiselect('Pick your assets',tickers_bt)
    start_bt = st.date_input('Start', value = pd.to_datetime('2020-01-01'))
    end_bt = st.date_input('End', value = pd.to_datetime('today'))

    if len(dropdown_bt)>0:
        if len(dropdown_bt)>1:
            st.write("Please select only 1 stock at the same time")
        else:
            frame=RSIcalc(str(dropdown_bt))
            buy, sell = getSignals(frame)

            buy_signals = frame.loc[frame['Buy'] == 'Yes']
            #Selling_dates.append(df.iloc[i + j + 1].name)
        
            st.write(buy_signals)
            st.line_chart(frame['Adj Close'])
