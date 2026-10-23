"""Options pricing dashboard backed by daily Alpha Vantage stock data."""

import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.alpha_vantage import MarketDataError, get_daily_prices
from src.options_pricing import (
    binomial_price,
    black_scholes,
    black_scholes_greeks,
    monte_carlo_price,
    payoff_curve,
)

st.set_page_config(
    page_title="Options Pricing Lab",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(
    """
    <style>
    .stApp { background: #0b1017; }
    [data-testid="stSidebar"] { background: #101823; border-right: 1px solid #202c3a; }
    .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1400px; }
    h1, h2, h3 { letter-spacing: -0.035em; }
    h1 { font-weight: 650; }
    [data-testid="stMetric"] { background: #111a25; border: 1px solid #233142; padding: 16px; border-radius: 10px; }
    [data-testid="stMetricLabel"] { color: #9baabd; }
    .eyebrow { color: #75b9ff; font-size: .78rem; letter-spacing: .16em; text-transform: uppercase; }
    .muted { color: #9baabd; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="eyebrow">Quantitative finance · learning project</div>', unsafe_allow_html=True)
st.title("Options Pricing Lab")
st.markdown("Compare pricing models, explore risk, and see how the underlying stock has moved.")

if "market_data" not in st.session_state:
    st.session_state.market_data = None
    st.session_state.market_symbol = None

with st.sidebar:
    st.header("Market data")
    env_key = os.environ.get("ALPHA_VANTAGE_API_KEY", "")
    with st.form("market_data_form"):
        symbol_entry = st.text_input("Stock symbol", value="AAPL", max_chars=20)
        api_key = st.text_input(
            "Alpha Vantage API key",
            value=env_key,
            type="password",
            help="The key is used for this request and is never written to a file.",
        )
        load_prices = st.form_submit_button("Load daily prices", type="primary", use_container_width=True)

    st.caption("Source: Alpha Vantage daily stock prices. The free daily endpoint returns up to 100 recent days.")
    st.caption("This app prices hypothetical options. It does not fetch listed option quotes.")

if load_prices:
    try:
        with st.spinner(f"Loading {symbol_entry.strip().upper()} from Alpha Vantage…"):
            st.session_state.market_data = get_daily_prices(symbol_entry, api_key)
            st.session_state.market_symbol = symbol_entry.strip().upper()
        st.success(f"Loaded {st.session_state.market_symbol} price history.")
    except MarketDataError as error:
        st.error(str(error))

prices = st.session_state.market_data
active_symbol = st.session_state.market_symbol
if prices is None:
    st.info("Enter your Alpha Vantage key in the sidebar and load a symbol to begin.")
    st.markdown(
        "**What you can explore:** historical price and volatility charts, Black–Scholes and binomial prices, "
        "a Monte Carlo estimate, option Greeks, and expiration payoff scenarios."
    )
    st.stop()

prices = prices.copy()
prices["daily_return"] = prices["close"].pct_change()
prices["log_return"] = np.log(prices["close"] / prices["close"].shift(1))
prices["realized_vol"] = prices["log_return"].rolling(20).std() * np.sqrt(252)
last_bar = prices.iloc[-1]
spot = float(last_bar["close"])
recent_returns = prices["log_return"].dropna()
realized_vol = float(recent_returns.tail(20).std() * np.sqrt(252)) if len(recent_returns) > 1 else 0.25
if not np.isfinite(realized_vol) or realized_vol <= 0:
    realized_vol = 0.25
realized_vol = min(realized_vol, 1.50)
day_change = float(last_bar["close"] / prices.iloc[-2]["close"] - 1) if len(prices) > 1 else 0.0

st.caption(
    f"{active_symbol} · latest daily close {last_bar['date'].date()} · "
    "Alpha Vantage daily close is raw (not adjusted for dividends or splits)."
)

summary = st.columns(4)
summary[0].metric("Latest close", f"${spot:,.2f}")
summary[1].metric("Previous-day move", f"{day_change:+.2%}")
summary[2].metric("20-day realized vol", f"{realized_vol:.1%}")
summary[3].metric("Daily bars", f"{len(prices)}")

market_tab, pricing_tab, notes_tab = st.tabs(["Market history", "Option pricing", "Model notes"])

with market_tab:
    st.subheader(f"{active_symbol} daily history")
    price_fig = go.Figure()
    price_fig.add_trace(
        go.Candlestick(
            x=prices["date"], open=prices["open"], high=prices["high"],
            low=prices["low"], close=prices["close"], name="Daily price",
        )
    )
    price_fig.update_layout(
        height=430, xaxis_title="Date", yaxis_title="Share price ($)",
        xaxis_rangeslider_visible=False, margin={"t": 15, "b": 20},
    )
    st.plotly_chart(price_fig, use_container_width=True)

    vol_fig = go.Figure()
    vol_fig.add_trace(
        go.Scatter(
            x=prices["date"], y=prices["realized_vol"] * 100,
            mode="lines", name="20-day realized volatility",
            line={"color": "#64b5f6", "width": 2},
        )
    )
    vol_fig.update_layout(
        height=280, xaxis_title="Date", yaxis_title="Annualized volatility (%)",
        margin={"t": 15, "b": 20},
    )
    st.plotly_chart(vol_fig, use_container_width=True)
    st.download_button(
        "Download this price history as CSV",
        prices[["date", "open", "high", "low", "close", "volume"]].to_csv(index=False),
        file_name=f"{active_symbol.lower()}_daily_prices.csv",
        mime="text/csv",
    )

with pricing_tab:
    st.subheader("Set up a hypothetical option")
    st.caption("Share price comes from the latest loaded daily close. Volatility starts at recent realized volatility; change it to explore a different assumption.")
    controls = st.columns(4)
    strike_multiple = controls[0].number_input("Strike (% of spot)", min_value=50.0, max_value=150.0, value=100.0, step=1.0) / 100
    days_to_expiry = controls[1].number_input("Days to expiry", min_value=1, max_value=730, value=90, step=1)
    volatility_pct = controls[2].number_input(
        "Volatility assumption (%)", min_value=1.0, max_value=150.0,
        value=float(round(realized_vol * 100, 1)), step=1.0,
    )
    rate_pct = controls[3].number_input("Risk-free rate (%)", min_value=-5.0, max_value=20.0, value=4.0, step=0.25)
    option_type = st.radio("Contract", ["Call", "Put"], horizontal=True, key="option_type")
    contracts = st.number_input("Number of contracts", min_value=1, max_value=1000, value=1, step=1)

    strike = spot * strike_multiple
    time = days_to_expiry / 365.0
    volatility = volatility_pct / 100.0
    rate = rate_pct / 100.0
    kind = option_type.lower()

    european_price = black_scholes(spot, strike, time, rate, volatility, kind)
    tree_european = binomial_price(spot, strike, time, rate, volatility, kind, steps=200)
    tree_american = binomial_price(spot, strike, time, rate, volatility, kind, steps=200, american=True)
    mc_estimate, mc_margin = monte_carlo_price(
        spot, strike, time, rate, volatility, kind, simulations=50_000, seed=11
    )
    greeks = black_scholes_greeks(spot, strike, time, rate, volatility, kind)

    price_cols = st.columns(4)
    price_cols[0].metric("Black–Scholes", f"${european_price:.3f} / share")
    price_cols[1].metric("Binomial · European", f"${tree_european:.3f}", f"{tree_european - european_price:+.3f} vs BS")
    price_cols[2].metric("Binomial · American", f"${tree_american:.3f}")
    price_cols[3].metric("Monte Carlo", f"${mc_estimate:.3f}", f"±${mc_margin:.3f} sampling range")

    st.subheader("Greeks")
    greek_cols = st.columns(4)
    greek_cols[0].metric("Delta", f"{greeks['delta']:.4f}")
    greek_cols[1].metric("Gamma", f"{greeks['gamma']:.5f}")
    greek_cols[2].metric("Theta / year", f"{greeks['theta']:.4f}")
    greek_cols[3].metric("Vega / 1.00 vol", f"{greeks['vega']:.4f}")
    st.caption("Vega is for a full 1.00 volatility change. For a 1 percentage-point move, multiply vega by 0.01.")

    chart_left, chart_right = st.columns(2)
    spot_grid = np.linspace(spot * 0.65, spot * 1.35, 121)
    pnl = payoff_curve(spot_grid, strike, european_price, kind, int(contracts))
    payoff_fig = go.Figure()
    payoff_fig.add_trace(go.Scatter(
        x=spot_grid, y=pnl, mode="lines", name="Long option P/L",
        line={"color": "#64b5f6", "width": 3}, fill="tozeroy",
    ))
    payoff_fig.add_hline(y=0, line_dash="dash", line_color="#7b8795")
    payoff_fig.add_vline(x=strike, line_dash="dot", line_color="#f2b84b", annotation_text="Strike")
    payoff_fig.update_layout(
        title="Profit or loss at expiry", height=360,
        xaxis_title="Share price at expiry ($)", yaxis_title="Profit / loss ($)",
        margin={"t": 50, "b": 20},
    )
    chart_left.plotly_chart(payoff_fig, use_container_width=True)

    value_by_spot = [black_scholes(float(s), strike, time, rate, volatility, kind) for s in spot_grid]
    value_fig = go.Figure()
    value_fig.add_trace(go.Scatter(
        x=spot_grid, y=value_by_spot, mode="lines", name="Model value",
        line={"color": "#91d3a2", "width": 3},
    ))
    value_fig.add_vline(x=spot, line_dash="dot", line_color="#f2b84b", annotation_text="Current spot")
    value_fig.update_layout(
        title="Model value as spot changes", height=360,
        xaxis_title="Share price now ($)", yaxis_title="Option value per share ($)",
        margin={"t": 50, "b": 20},
    )
    chart_right.plotly_chart(value_fig, use_container_width=True)

    breakeven = strike + european_price if kind == "call" else strike - european_price
    summary_text = f"Break-even at expiry: ${breakeven:.2f} per share."
    if kind == "call":
        summary_text += " A long call's maximum loss is the premium paid; upside is theoretically unlimited."
    else:
        summary_text += f" A long put's maximum loss is the premium paid; maximum profit is ${(strike - european_price) * 100 * contracts:,.2f} for this position."
    st.info(summary_text)
    st.caption("Payoff uses the Black–Scholes value as the option premium and assumes one standard contract represents 100 shares.")

with notes_tab:
    st.subheader("How the numbers are made")
    st.markdown("""
    **Market data.** The app requests `TIME_SERIES_DAILY` from Alpha Vantage and uses the latest daily close as spot. The endpoint returns raw, as-traded prices; historical prices are not adjusted for splits or dividends. Recent realized volatility is an estimate from daily log returns, annualized using 252 trading days.

    **Black–Scholes.** Prices a European call or put under geometric Brownian motion with constant volatility and rate, no dividends, and continuous trading assumptions.

    **Binomial tree.** Builds a Cox–Ross–Rubinstein up/down price tree and works backward using risk-neutral probabilities. The American version checks early exercise at each node.

    **Monte Carlo.** Simulates terminal prices, discounts the payoff, and reports an approximate 95% range for simulation sampling error. This range does not capture model error.

    **Greeks.** Delta, gamma, theta, and vega are analytical Black–Scholes sensitivities. Theta is per year; vega is for a 1.00 change in volatility.

    **Important limit.** Alpha Vantage stock prices are not option-chain quotes. The app does not know the market premium or implied volatility for a listed contract. These outputs are hypothetical model values for learning, not investment advice.
    """)
    st.markdown("#### Data source")
    st.markdown("[Alpha Vantage API documentation](https://www.alphavantage.co/documentation/)")
