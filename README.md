# Options Pricing Lab

A small interactive app for studying option pricing and risk with real daily stock prices from Alpha Vantage.

It compares Black–Scholes, a Cox–Ross–Rubinstein binomial tree, and Monte Carlo simulation. It also shows recent stock history, realized volatility, Greeks, and a long option's profit or loss at expiry.

## Run it

You need Python 3.10 or newer and an Alpha Vantage API key.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Open the local address Streamlit prints. Enter your API key in the sidebar, choose a stock symbol, and load its daily prices. The app uses Alpha Vantage's `TIME_SERIES_DAILY` endpoint with `outputsize=compact`, which returns up to 100 recent daily observations. The key is not saved in a file or sent anywhere except Alpha Vantage.

You can also set `ALPHA_VANTAGE_API_KEY` in your local environment before starting the app. Never commit your key.

## What is inside

- Daily OHLCV prices and a 20-day realized volatility estimate.
- Black–Scholes pricing for European calls and puts.
- European and American Cox–Ross–Rubinstein tree prices.
- A reproducible Monte Carlo estimate with a sampling-error range.
- Analytical delta, gamma, theta, and vega.
- Expiration P/L and model value sensitivity charts.
- CSV download of the fetched daily price history.

## The math in brief

The Black–Scholes and Monte Carlo methods assume a lognormal stock-price process with constant volatility and rate. The tree uses risk-neutral up and down moves and backward induction. Daily realized volatility is calculated from log returns and annualized with 252 trading days.

## Data and model limits

The app uses Alpha Vantage daily **raw close** prices. They are not adjusted for splits or dividends, so historical returns can be distorted around corporate actions. The app does not fetch options-chain quotes or implied volatility: the selected volatility is a user assumption that starts from recent realized volatility. Monte Carlo's interval describes simulation sampling error, not uncertainty in the model. None of the displayed values are live option quotes or investment advice.

Alpha Vantage may limit request frequency or access by API key. If a request is limited, wait and retry. See the [official API documentation](https://www.alphavantage.co/documentation/) for endpoint details and access terms.

## Project structure

- `app.py` — Streamlit interface and charts.
- `src/alpha_vantage.py` — input checks and Alpha Vantage price download.
- `src/options_pricing.py` — pricing formulas, Greeks, and payoff calculation.

## Good next steps

1. Add adjusted daily data and a clear dividend-yield input.
2. Add implied-volatility solving from a user-entered market premium.
3. Plot how tree and Monte Carlo prices converge as computation increases.
4. Compare analytical Greeks with finite-difference estimates.
