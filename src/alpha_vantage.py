"""Small client for Alpha Vantage's daily stock-price endpoint."""

from __future__ import annotations

import re

import pandas as pd
import requests

API_URL = "https://www.alphavantage.co/query"
SERIES_KEY = "Time Series (Daily)"
SYMBOL_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9._-]{0,19}$")


class MarketDataError(Exception):
    """A short, user-facing market-data error."""


def get_daily_prices(symbol: str, api_key: str) -> pd.DataFrame:
    """Fetch up to 100 recent daily bars and return them oldest first."""
    symbol = symbol.strip().upper()
    api_key = api_key.strip()
    if not SYMBOL_PATTERN.fullmatch(symbol):
        raise MarketDataError("Enter a stock symbol using letters, numbers, dots, or dashes.")
    if not api_key:
        raise MarketDataError("Enter your Alpha Vantage API key to load market data.")

    try:
        response = requests.get(
            API_URL,
            params={
                "function": "TIME_SERIES_DAILY",
                "symbol": symbol,
                "outputsize": "compact",
                "apikey": api_key,
            },
            timeout=20,
        )
        response.raise_for_status()
        payload = response.json()
    except requests.Timeout as error:
        raise MarketDataError("Alpha Vantage took too long to respond. Try again in a moment.") from error
    except requests.RequestException as error:
        raise MarketDataError("Could not reach Alpha Vantage. Check your connection and try again.") from error
    except ValueError as error:
        raise MarketDataError("Alpha Vantage returned data the app could not read.") from error

    if SERIES_KEY not in payload:
        if "Note" in payload or "Information" in payload:
            message = "Alpha Vantage has temporarily limited this key. Wait a little, then try again."
        elif "Error Message" in payload:
            message = "That symbol was not accepted. Check it and try again."
        else:
            message = "No daily price history came back. Check your key and symbol."
        raise MarketDataError(message)

    rows = []
    for day, values in payload[SERIES_KEY].items():
        try:
            rows.append(
                {
                    "date": pd.to_datetime(day),
                    "open": float(values["1. open"]),
                    "high": float(values["2. high"]),
                    "low": float(values["3. low"]),
                    "close": float(values["4. close"]),
                    "volume": int(values["5. volume"]),
                }
            )
        except (KeyError, TypeError, ValueError) as error:
            raise MarketDataError("Alpha Vantage returned a price row in an unexpected format.") from error

    prices = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    if prices.empty:
        raise MarketDataError("No daily prices were found for that symbol.")
    return prices
