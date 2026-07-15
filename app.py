import pandas as pd
import numpy as np
import streamlit as slt
import plotly.graph_objects as go
import joblib
import os

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
slt.set_page_config(
    page_title="NIFTY-50 Stock Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------------------------------
# CUSTOM CSS — dark, trading-terminal inspired theme
# ----------------------------------------------------------------------------
slt.markdown("""
<style>
    .main { background-color: #0e1117; }

    .app-header {
        padding: 1.4rem 1.8rem;
        border-radius: 14px;
        background: linear-gradient(120deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        margin-bottom: 1.2rem;
        border: 1px solid rgba(255,255,255,0.06);
    }
    .app-header h1 { color: #f5f7fa; font-size: 2rem; margin-bottom: 0.2rem; }
    .app-header p  { color: #9fb3c8; font-size: 0.95rem; margin: 0; }

    .kpi-card {
        background: #161b22;
        border: 1px solid #262d38;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        text-align: left;
    }
    .kpi-label {
        color: #8b98a9;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.25rem;
    }
    .kpi-value { color: #f5f7fa; font-size: 1.5rem; font-weight: 700; }
    .kpi-delta-up   { color: #26a69a; font-size: 0.9rem; font-weight: 600; }
    .kpi-delta-down { color: #ef5350; font-size: 0.9rem; font-weight: 600; }

    .feature-card {
        background: #161b22;
        border: 1px solid #262d38;
        border-radius: 12px;
        padding: 0.9rem 1.1rem;
        text-align: center;
    }
    .feature-label { color: #8b98a9; font-size: 0.75rem; text-transform: uppercase; letter-spacing: .05em; }
    .feature-value { color: #f5f7fa; font-size: 1.2rem; font-weight: 700; margin-top: 4px; }

    .predict-card {
        background: linear-gradient(135deg, #1b2a3d 0%, #16222f 100%);
        border: 1px solid #2c5364;
        border-radius: 16px;
        padding: 1.8rem;
        text-align: center;
        margin-top: 1rem;
    }
    .predict-card h2 { color: #7fdbca; font-size: 2.3rem; margin: 0.3rem 0; }
    .predict-card p  { color: #9fb3c8; margin: 0; }

    section[data-testid="stSidebar"] { background-color: #10151c; }

    div[data-testid="stMetric"] {
        background-color: #161b22;
        border: 1px solid #262d38;
        border-radius: 12px;
        padding: 0.8rem;
    }

    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #161b22;
        border-radius: 8px 8px 0 0;
        padding: 0.5rem 1.2rem;
        color: #9fb3c8;
    }
    .stTabs [aria-selected="true"] { background-color: #203a43; color: #ffffff; }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# DATA LOADING (cached)
# ----------------------------------------------------------------------------
@slt.cache_data
def load_data():
    return pd.read_csv("dataset/NIFTY50_all.csv.gz", compression="gzip", parse_dates=["Date"])

@slt.cache_resource
def load_model(company):
    path = f"models/{company}.pkl"
    if not os.path.exists(path):
        return None
    return joblib.load(path)

try:
    df = load_data()
except FileNotFoundError:
    slt.error("Couldn't find `dataset/NIFTY50_all.csv.gz`. Make sure the dataset folder is deployed alongside app.py.")
    slt.stop()

companies = sorted(df["Symbol"].unique())

# ----------------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------------
with slt.sidebar:
    slt.markdown("## 📈 Controls")
    selected_company = slt.selectbox("Select company", companies)

    company_df = df[df["Symbol"] == selected_company].sort_values("Date")

    min_date, max_date = company_df["Date"].min(), company_df["Date"].max()
    date_range = slt.slider(
        "Chart date range",
        min_value=min_date.to_pydatetime(),
        max_value=max_date.to_pydatetime(),
        value=(max(min_date, max_date - pd.Timedelta(days=365)).to_pydatetime(), max_date.to_pydatetime()),
        format="DD MMM YYYY"
    )

    show_ma = slt.checkbox("Show moving averages (20 / 50 day)", value=True)

    slt.markdown("---")
    slt.caption("Data source: `dataset/NIFTY50_all.csv.gz`")
    slt.caption("Model: `models/<SYMBOL>.pkl`")

chart_df = company_df[(company_df["Date"] >= date_range[0]) & (company_df["Date"] <= date_range[1])]

# ----------------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------------
slt.markdown(f"""
<div class="app-header">
    <h1>📈 NIFTY-50 Stock Prediction Dashboard</h1>
    <p>Predict tomorrow's closing price for <b>{selected_company}</b> using Machine Learning.</p>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# KPI ROW
# ----------------------------------------------------------------------------
latest = company_df.iloc[-1]
prev = company_df.iloc[-2] if len(company_df) > 1 else latest
change = latest["Close"] - prev["Close"]
pct_change = (change / prev["Close"] * 100) if prev["Close"] else 0
period_high = chart_df["High"].max() if not chart_df.empty else latest["High"]
period_low = chart_df["Low"].min() if not chart_df.empty else latest["Low"]

k1, k2, k3, k4 = slt.columns(4)
delta_class = "kpi-delta-up" if change >= 0 else "kpi-delta-down"
arrow = "▲" if change >= 0 else "▼"

with k1:
    slt.markdown(f"""<div class="kpi-card">
        <div class="kpi-label">Last Close ({latest['Date'].date()})</div>
        <div class="kpi-value">₹{latest['Close']:.2f}</div>
        <div class="{delta_class}">{arrow} {change:+.2f} ({pct_change:+.2f}%)</div>
    </div>""", unsafe_allow_html=True)
with k2:
    slt.markdown(f"""<div class="kpi-card">
        <div class="kpi-label">Volume</div>
        <div class="kpi-value">{latest['Volume']:,.0f}</div>
    </div>""", unsafe_allow_html=True)
with k3:
    slt.markdown(f"""<div class="kpi-card">
        <div class="kpi-label">Period High</div>
        <div class="kpi-value">₹{period_high:.2f}</div>
    </div>""", unsafe_allow_html=True)
with k4:
    slt.markdown(f"""<div class="kpi-card">
        <div class="kpi-label">Period Low</div>
        <div class="kpi-value">₹{period_low:.2f}</div>
    </div>""", unsafe_allow_html=True)

slt.markdown("<br>", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# TABS
# ----------------------------------------------------------------------------
tab_overview, tab_predict, tab_data = slt.tabs(["📊 Overview & Charts", "🔮 Predict Tomorrow", "📁 Raw Data"])

# --- TAB 1: OVERVIEW -----------------------------------------------------
with tab_overview:
    slt.subheader(selected_company)
    slt.dataframe(company_df.tail(), use_container_width=True)

    if chart_df.empty:
        slt.info("No data in the selected date range.")
    else:
        slt.markdown("**Closing Price Trend**")
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=chart_df["Date"],
            open=chart_df["Open"], high=chart_df["High"],
            low=chart_df["Low"], close=chart_df["Close"],
            name=selected_company,
            increasing_line_color="#26a69a", decreasing_line_color="#ef5350"
        ))
        if show_ma:
            ma20 = chart_df["Close"].rolling(20).mean()
            ma50 = chart_df["Close"].rolling(50).mean()
            fig.add_trace(go.Scatter(x=chart_df["Date"], y=ma20, name="MA 20",
                                      line=dict(color="#ffb74d", width=1.3)))
            fig.add_trace(go.Scatter(x=chart_df["Date"], y=ma50, name="MA 50",
                                      line=dict(color="#64b5f6", width=1.3)))
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            height=440,
            margin=dict(l=10, r=10, t=20, b=10),
            xaxis_rangeslider_visible=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        slt.plotly_chart(fig, use_container_width=True)

        slt.markdown("**Trading Volume Trend**")
        vol_fig = go.Figure()
        colors = np.where(chart_df["Close"] >= chart_df["Open"], "#26a69a", "#ef5350")
        vol_fig.add_trace(go.Bar(x=chart_df["Date"], y=chart_df["Volume"], marker_color=colors, name="Volume"))
        vol_fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            height=200,
            margin=dict(l=10, r=10, t=10, b=10),
            showlegend=False
        )
        slt.plotly_chart(vol_fig, use_container_width=True)

# --- TAB 2: PREDICTION -----------------------------------------------------
with tab_predict:
    model = load_model(selected_company)

    if model is None:
        slt.warning(f"No trained model found at `models/{selected_company}.pkl`. Prediction is disabled for this company.")
    else:
        slt.markdown("Enter today's market values — defaults are pre-filled from the most recent trading day.")

        c1, c2, c3 = slt.columns(3)
        with c1:
            prev_close = slt.number_input("Previous Close", value=float(latest["Close"]), format="%.2f")
            open_price = slt.number_input("Open", value=float(latest["Open"]), format="%.2f")
            close = slt.number_input("Today's Close", value=float(latest["Close"]), format="%.2f")
            vwap = slt.number_input("VWAP", value=float(latest.get("VWAP", latest["Close"])), format="%.2f")
        with c2:
            high = slt.number_input("High", value=float(latest["High"]), format="%.2f")
            low = slt.number_input("Low", value=float(latest["Low"]), format="%.2f")
            volume = slt.number_input("Volume", min_value=0, value=int(latest.get("Volume", 4850000)), step=1000, format="%d")
        with c3:
            trades = slt.number_input("Number of Trades", min_value=0, value=int(latest.get("Trades", 98500)), step=100, format="%d")
            deliverable = slt.number_input("Deliverable Volume", min_value=0, value=int(latest.get("Deliverable Volume", 2420000)), step=1000, format="%d")
            percent = slt.number_input("% Deliverable", value=float(latest.get("%Deliverble", 0)), format="%.2f")

        slt.markdown("<br>", unsafe_allow_html=True)
        predict_clicked = slt.button("🔮 Predict Tomorrow's Closing Price", use_container_width=True)

        if predict_clicked:
            price_change = close - open_price
            high_low_difference = high - low
            daily_return = ((close - open_price) / open_price) * 100 if open_price != 0 else 0

            slt.markdown("#### 📊 Calculated Features")
            f1, f2, f3 = slt.columns(3)
            with f1:
                slt.markdown(f"""<div class="feature-card">
                    <div class="feature-label">Price Change</div>
                    <div class="feature-value">₹{price_change:+.2f}</div>
                </div>""", unsafe_allow_html=True)
            with f2:
                slt.markdown(f"""<div class="feature-card">
                    <div class="feature-label">High-Low Difference</div>
                    <div class="feature-value">₹{high_low_difference:.2f}</div>
                </div>""", unsafe_allow_html=True)
            with f3:
                slt.markdown(f"""<div class="feature-card">
                    <div class="feature-label">Daily Return</div>
                    <div class="feature-value">{daily_return:+.2f}%</div>
                </div>""", unsafe_allow_html=True)

            feature_names = [
                "Prev Close", "Open", "High", "Low", "VWAP", "Volume",
                "Trades", "Deliverable Volume", "%Deliverble",
                "Price_Change", "High_Low_Difference", "Daily_Return"
            ]
            features = pd.DataFrame([[
                prev_close, open_price, high, low, vwap, volume,
                trades, deliverable, percent,
                price_change, high_low_difference, daily_return
            ]], columns=feature_names)

            try:
                prediction = model.predict(features)[0]
                pred_change = prediction - close
                pred_pct = (pred_change / close * 100) if close else 0
                arrow2 = "▲" if pred_change >= 0 else "▼"

                slt.markdown(f"""
                <div class="predict-card">
                    <p>Predicted Closing Price — {selected_company}</p>
                    <h2>₹{prediction:.2f}</h2>
                    <p>{arrow2} {pred_change:+.2f} ({pred_pct:+.2f}%) vs today's close</p>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                slt.error(f"Prediction failed: {e}")

# --- TAB 3: RAW DATA -----------------------------------------------------
with tab_data:
    slt.subheader(f"{selected_company} — Historical Data")
    slt.dataframe(company_df.sort_values("Date", ascending=False), use_container_width=True, height=460)
    slt.download_button(
        "⬇️ Download as CSV",
        data=company_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{selected_company}_historical.csv",
        mime="text/csv"
    )
