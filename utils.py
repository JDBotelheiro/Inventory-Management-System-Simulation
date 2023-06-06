# utils.py
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import datetime

from math import sqrt
from model import *

def load_data():
    supplier_info = pd.read_parquet('./data/supplier_info.parquet')
    sales_data = pd.read_parquet('./data/sales_data.parquet')
    inventory_level = pd.read_parquet('./data/inventory_level.parquet')
    return supplier_info, sales_data, inventory_level

def calculate_eoq(D, K, h, L, Z, std_dev):
    min_cost = None
    best_Q = None
    for Q in range(1, int(D)):  # search possible values of Q
        safety_stock = Z * std_dev * sqrt(L)  # Calculate safety stock
        cost = K * (D / Q) + h * (Q / 2) + safety_stock * h  # calculate total cost considering safety stock
        if min_cost is None or cost < min_cost:  # update min cost and best Q
            min_cost = cost
            best_Q = Q
    return best_Q, safety_stock

def calculate_safety_stock(z_value, std_dev, lead_time):
    return z_value * std_dev * sqrt(lead_time)

def remote_css(url):
    st.markdown(f'<link href="{url}" rel="stylesheet">', unsafe_allow_html=True)

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

@st.cache_resource
def sales_forecasting(product_sales):
    # Prepare data for sales forecasting
    product_sales['ordinal_datetime'] = product_sales['date'].map(datetime.datetime.toordinal).values.reshape(-1, 1)
    product_sales['month'] = product_sales['date'].dt.month
    product_sales['week'] = product_sales['date'].dt.isocalendar().week.astype('int64')
    product_sales['sales_lag_1y'] = product_sales['sales'].shift(365)
    product_sales = product_sales.dropna(subset=['sales_lag_1y'])
    y = product_sales['sales'].values
    X = product_sales[['ordinal_datetime', 'month', 'week', 'sales_lag_1y']]
    # Train a linear regression model
    model, X_cal, y_cal = train_model(X, y)
    
    # Use the model to predict future sales
    future_dates = pd.DataFrame(pd.date_range(start=max(product_sales['date']), periods=30, freq='D')[1:], columns=['date'])
    # Preprocess future dates
    future_dates['ordinal_datetime'] = future_dates['date'].map(datetime.datetime.toordinal)
    future_dates['month'] = future_dates['date'].dt.month
    future_dates['week'] = future_dates['date'].dt.isocalendar().week.astype('int64')
    future_dates['sales'] = 0
    combined_data = pd.concat([future_dates[['date', 'ordinal_datetime', 'month', 'week', 'sales']], product_sales[['date', 'ordinal_datetime', 'month', 'week', 'sales']]], axis=0, ignore_index=True)
    combined_data = combined_data.sort_values(by="date")
    combined_data['sales_lag_1y'] = combined_data['sales'].shift(365)
    future_dates = combined_data[combined_data['date'].isin(future_dates['date'])]
    # Forecast sales
    future_dates['y_forecast'] = forecast(model, future_dates)
    
    # Use calibration st to apply the model to calculate the residuals
    y_hat_cal = model.predict(X_cal)
    X_cal["y_cal"] = y_hat_cal
    X_cal["residuals_cal"] = y_cal - y_hat_cal
    # Calculate conformal intervals
    future_dates = calculate_intervals(model, X_cal, y_cal, future_dates)

    # Filter data for one year
    one_year_data = product_sales.loc[product_sales['date'] >= (product_sales['date'].max() - pd.DateOffset(years=1))]

    # Plot sales data and sales forecast
    fig = go.Figure()

    # Historical Sales
    fig.add_trace(go.Scatter(
        x=one_year_data['date'],
        y=one_year_data['sales'],
        mode='lines',
        name='Historical Sales',
        line=dict(color='#FF6D6D') 
    ))

    # Confidence Interval
    fig.add_trace(go.Scatter(
        x=future_dates['date'],
        y=future_dates['y_forecast-hi-80'],
        mode='lines',
        name='Upper Bound (80% Confidence Interval)',
        line=dict(color='#AAAAAA')  # Set gray color for upper bound
        #fill='tonexty'  # Fill area between upper bound and lower bound
    ))

    fig.add_trace(go.Scatter(
        x=future_dates['date'],
        y=future_dates['y_forecast-lo-80'],
        mode='lines',
        name='Lower Bound (80% Confidence Interval)',
        line=dict(color='#AAAAAA'),  # Set gray color for lower bound
        fill='tonexty'  # Fill area between lower bound and x-axis
    ))

    # Forecasted Sales
    fig.add_trace(go.Scatter(
        x=future_dates['date'],
        y=future_dates['y_forecast'],
        mode='lines',
        name='Forecasted Sales',
        line=dict(color='#666666')  # Set Tesla blue color
    ))

    fig.update_layout(
        title='Sales Data and Forecast',
        xaxis_title='Date',
        yaxis_title='Sales',
        #height=800,  # Increase the height of the plot
        #width=700,  # Increase the width of the plot
        legend_title_font=dict(size=9),  # Decrease the size of the legend title
        legend_font=dict(size=9),  # Decrease the size of the legend text
        title_font=dict(size=14),  # Decrease the size of the title
        xaxis_title_font=dict(size=12),  # Decrease the size of the x-axis title
        yaxis_title_font=dict(size=12),  # Decrease the size of the y-axis title
    )

    return fig, future_dates


@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')