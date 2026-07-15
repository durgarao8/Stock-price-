import pandas as pd
import numpy as np
import streamlit as slt 

import joblib
import os

slt.set_page_config(
    page_title="NIFTY-50 Stock Prediction",
    page_icon="📈",
    layout="wide"
)

slt.title("📈 NIFTY-50 Stock Prediction Dashboard")

slt.write(
    "Predict tomorrow's closing price using Machine Learning."
)

df = pd.read_csv("dataset/NIFTY50_all.csv.gz", compression="gzip")
companies=sorted(df["Symbol"].unique())
selected_company=slt.sidebar.selectbox("Select company",companies)

model=joblib.load(f"models/{selected_company}.pkl")

#scaler=joblib.load(f"scalers/{selected_company}_scaler.pkl")

company_df=df[df["Symbol"]==selected_company]

slt.subheader(selected_company)
slt.dataframe(company_df.tail())

slt.line_chart(company_df.set_index("Date")["Close"])

slt.bar_chart(company_df.set_index("Date")["Volume"])

slt.subheader("Enter Today's Market Values")

prev_close=slt.number_input(f"Enter the Previous Closing Price f {selected_company} ")
open_price=slt.number_input(f"Enter Input Price of {selected_company}")
high=slt.number_input(f"Enter High Price {selected_company}")
low=slt.number_input(f"Enter the Low Price {selected_company}")
close = slt.number_input("Today's Close")
vwap=slt.number_input(f"Enter VWAP {selected_company}")
volume=slt.number_input(f"Enter Volume in that Day {selected_company}",min_value=0,value=4850000,step=1000,format="%d")
trades=slt.number_input(f"Enter Number of Trades {selected_company}",min_value=0,value=98500,step=100,format="%d")
deliverable=slt.number_input(f"Enter Deliverbale volume {selected_company}",min_value=0,value=2420000,step=1000,format="%d")
percent=slt.number_input(f"Enter Deliverable volume  Percantage {selected_company}")

if slt.button("Predict Tomorrow Closing Price"):

    price_change = close - open_price

    high_low_difference = high - low

    if open_price != 0:
        daily_return = ((close - open_price) / open_price) * 100
    else:
        daily_return = 0
    slt.subheader("📊 Calculated Features")

    slt.write(f"**Price Change:** {price_change:.2f}")

    slt.write(f"**High-Low Difference:** {high_low_difference:.2f}")

    slt.write(f"**Daily Return:** {daily_return:.2f}%")

    feature_names = [
    "Prev Close",
    "Open",
    "High",
    "Low",
    "VWAP",
    "Volume",
    "Trades",
    "Deliverable Volume",
    "%Deliverble",
    "Price_Change",
    "High_Low_Difference",
    "Daily_Return"
    ]

    features = pd.DataFrame(
    [[
        prev_close,
        open_price,
        high,
        low,
        vwap,
        volume,
        trades,
        deliverable,
        percent,
        price_change,
        high_low_difference,
        daily_return
    ]],
    columns=feature_names
    )
    prediction = model.predict(features)

    slt.success(
        f"Predicted Tomorrow Closing Price : ₹{prediction[0]:.2f}"
    )
