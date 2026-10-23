"""Option pricing methods and risk measures used by the explorer.

All inputs use annualized rates and volatility, with time measured in years.
Prices are per share; multiply by 100 for a standard US equity contract.
"""

from __future__ import annotations

import math

import numpy as np


def _validate_inputs(spot: float, strike: float, time: float, rate: float, volatility: float) -> None:
    if spot <= 0 or strike <= 0:
        raise ValueError("Spot and strike prices must be positive.")
    if time < 0:
        raise ValueError("Time to expiry cannot be negative.")
    if volatility < 0:
        raise ValueError("Volatility cannot be negative.")


def _normal_cdf(value: float) -> float:
    """Standard normal cumulative distribution, using the standard library."""
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def black_scholes(
    spot: float, strike: float, time: float, rate: float, volatility: float, option_type: str
) -> float:
    """Return the European option value under Black–Scholes, with no dividends."""
    _validate_inputs(spot, strike, time, rate, volatility)
    kind = option_type.lower()
    if kind not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'.")
    if time == 0 or volatility == 0:
        discounted_strike = strike * math.exp(-rate * time)
        if kind == "call":
            return max(spot - discounted_strike, 0.0)
        return max(discounted_strike - spot, 0.0)

    root_time = math.sqrt(time)
    d1 = (math.log(spot / strike) + (rate + 0.5 * volatility**2) * time) / (volatility * root_time)
    d2 = d1 - volatility * root_time
    discount = math.exp(-rate * time)
    if kind == "call":
        return spot * _normal_cdf(d1) - strike * discount * _normal_cdf(d2)
    return strike * discount * _normal_cdf(-d2) - spot * _normal_cdf(-d1)


def black_scholes_greeks(
    spot: float, strike: float, time: float, rate: float, volatility: float, option_type: str
) -> dict[str, float]:
    """Return analytical delta, gamma, theta (per year), and vega (per 1.00 vol)."""
    _validate_inputs(spot, strike, time, rate, volatility)
    if time == 0 or volatility == 0:
        # At expiry or zero volatility, the smooth Black–Scholes derivatives do not
        # exist at the payoff kink. Returning NaN is more honest than inventing them.
        return {name: math.nan for name in ("delta", "gamma", "theta", "vega")}

    kind = option_type.lower()
    if kind not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'.")
    root_time = math.sqrt(time)
    d1 = (math.log(spot / strike) + (rate + 0.5 * volatility**2) * time) / (volatility * root_time)
    d2 = d1 - volatility * root_time
    density = math.exp(-0.5 * d1**2) / math.sqrt(2.0 * math.pi)
    discount = math.exp(-rate * time)
    gamma = density / (spot * volatility * root_time)
    vega = spot * density * root_time
    if kind == "call":
        delta = _normal_cdf(d1)
        theta = -(spot * density * volatility) / (2.0 * root_time) - rate * strike * discount * _normal_cdf(d2)
    else:
        delta = _normal_cdf(d1) - 1.0
        theta = -(spot * density * volatility) / (2.0 * root_time) + rate * strike * discount * _normal_cdf(-d2)
    return {"delta": delta, "gamma": gamma, "theta": theta, "vega": vega}


def binomial_price(
    spot: float,
    strike: float,
    time: float,
    rate: float,
    volatility: float,
    option_type: str,
    steps: int = 200,
    american: bool = False,
) -> float:
    """Price an option using a Cox–Ross–Rubinstein tree and backward induction."""
    _validate_inputs(spot, strike, time, rate, volatility)
    kind = option_type.lower()
    if kind not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'.")
    if steps < 1:
        raise ValueError("steps must be at least 1.")
    if time == 0 or volatility == 0:
        return black_scholes(spot, strike, time, rate, volatility, kind)

    step_time = time / steps
    up = math.exp(volatility * math.sqrt(step_time))
    down = 1.0 / up
    growth = math.exp(rate * step_time)
    probability_up = (growth - down) / (up - down)
    if not 0.0 <= probability_up <= 1.0:
        raise ValueError("Tree parameters imply an invalid risk-neutral probability.")

    # At expiry, node j has j up moves and steps-j down moves.
    up_moves = np.arange(steps + 1)
    terminal_spots = spot * up**up_moves * down ** (steps - up_moves)
    if kind == "call":
        values = np.maximum(terminal_spots - strike, 0.0)
    else:
        values = np.maximum(strike - terminal_spots, 0.0)

    discount = math.exp(-rate * step_time)
    for level in range(steps - 1, -1, -1):
        values = discount * (probability_up * values[1:] + (1.0 - probability_up) * values[:-1])
        if american:
            node_moves = np.arange(level + 1)
            node_spots = spot * up**node_moves * down ** (level - node_moves)
            if kind == "call":
                exercise_value = np.maximum(node_spots - strike, 0.0)
            else:
                exercise_value = np.maximum(strike - node_spots, 0.0)
            values = np.maximum(values, exercise_value)
    return float(values[0])


def monte_carlo_price(
    spot: float,
    strike: float,
    time: float,
    rate: float,
    volatility: float,
    option_type: str,
    simulations: int = 100_000,
    seed: int = 7,
) -> tuple[float, float]:
    """Return a European price estimate and its approximate 95% sampling interval."""
    _validate_inputs(spot, strike, time, rate, volatility)
    kind = option_type.lower()
    if kind not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'.")
    if simulations < 2:
        raise ValueError("At least two simulations are required.")
    if time == 0 or volatility == 0:
        exact = black_scholes(spot, strike, time, rate, volatility, kind)
        return exact, 0.0

    random = np.random.default_rng(seed)
    normal_draws = random.standard_normal(simulations)
    terminal_spot = spot * np.exp(
        (rate - 0.5 * volatility**2) * time + volatility * math.sqrt(time) * normal_draws
    )
    if kind == "call":
        payoffs = np.maximum(terminal_spot - strike, 0.0)
    else:
        payoffs = np.maximum(strike - terminal_spot, 0.0)
    discounted = math.exp(-rate * time) * payoffs
    estimate = float(discounted.mean())
    margin = 1.96 * float(discounted.std(ddof=1)) / math.sqrt(simulations)
    return estimate, margin


def payoff_curve(
    spot_prices: np.ndarray, strike: float, premium: float, option_type: str, contracts: int = 1
) -> np.ndarray:
    """Expiration profit/loss for a long option position, in currency units."""
    kind = option_type.lower()
    if kind == "call":
        intrinsic = np.maximum(spot_prices - strike, 0.0)
    elif kind == "put":
        intrinsic = np.maximum(strike - spot_prices, 0.0)
    else:
        raise ValueError("option_type must be 'call' or 'put'.")
    return (intrinsic - premium) * contracts * 100
