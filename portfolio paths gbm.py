#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 02:46:57 2026

@author: dwpetterson
"""

# -*- coding: utf-8 -*-

import yfinance as yf
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("C:\\Users\\dwpet\\Downloads\\portfolio frontier.csv", delimiter=";")

df = df.dropna(subset=["Tickers", "Weight"])
df["Tickers"] = df["Tickers"].astype(str).str.strip()
df["Weight"] = pd.to_numeric(df["Weight"], errors="coerce")
df = df.dropna(subset=["Weight"])

stocks = df["Tickers"].tolist()
weight = df["Weight"].tolist()

initial_price = []
beta = []
alpha = []
specific_volatility = []
valid_weights = []
valid_stocks = []

market = yf.download("OSEBX.OL", period="5y", auto_adjust=True, progress=False)["Close"]

if isinstance(market, pd.Series):
    market = market.to_frame("market")
else:
    market.columns = ["market"]

market_returns = market.pct_change(fill_method=None).dropna()

for idx, ticker in enumerate(stocks):
    stock = yf.download(ticker, period="5y", auto_adjust=True, progress=False)["Close"]

    if stock.empty:
        print("Skipped empty:", ticker)
        continue

    if isinstance(stock, pd.Series):
        stock = stock.to_frame("stock")
    else:
        stock.columns = ["stock"]

    if stock.isna().all().all():
        print("Skipped all NaN:", ticker)
        continue

    stock_returns = stock.pct_change(fill_method=None).dropna()

    data = stock_returns.join(market_returns, how="inner").dropna()

    if len(data) < 100:
        print("Skipped too little data:", ticker)
        continue

    X = sm.add_constant(data["market"])
    model = sm.OLS(data["stock"], X).fit()

    a = model.params["const"]
    b = model.params["market"]
    s = np.std(model.resid)

    if np.isnan(a) or np.isnan(b) or np.isnan(s):
        print("Skipped NaN regression:", ticker)
        continue

    latest_price = stock.dropna().iloc[-1, 0]

    initial_price.append(latest_price)
    alpha.append(a)
    beta.append(b)
    specific_volatility.append(s)
    valid_weights.append(weight[idx])
    valid_stocks.append(ticker)

valid_weights = np.array(valid_weights)

if len(valid_weights) == 0:
    raise ValueError("No valid stocks found.")

valid_weights = valid_weights / valid_weights.sum()

steps = 252
simulations = 1000
portfolio_size = len(valid_stocks)

market_mu = 0.0005
market_sigma = 0.0089
initial_investment = 1000

portfolio_paths = np.zeros((steps, simulations))

for sim in range(simulations):
    stock_paths = np.zeros((steps, portfolio_size))
    market_returns_sim = np.random.normal(market_mu, market_sigma, steps)

    for stock in range(portfolio_size):
        stock_noise = np.random.normal(0, specific_volatility[stock], steps)

        stock_returns = (
            alpha[stock]
            + beta[stock] * market_returns_sim
            + stock_noise
        )

        stock_price = initial_price[stock] * (1 + stock_returns).cumprod()

        stock_paths[:, stock] = stock_price

    portfolio = np.zeros(steps)

    for i in range(portfolio_size):
        shares = (initial_investment * valid_weights[i]) / initial_price[i]
        portfolio += shares * stock_paths[:, i]

    portfolio_paths[:, sim] = portfolio

for sim in range(simulations):
    plt.plot(portfolio_paths[:, sim], alpha=0.05)

plt.plot(
    portfolio_paths.mean(axis=1),
    linewidth=2,
    label="Average Path",
    alpha=0.8,
    color="black"
)

plt.title("1000 Simulated Portfolio Paths")
plt.ylabel("Portfolio Value")
plt.xlabel("Days")
plt.legend()
plt.show()

final_values = portfolio_paths[-1, :]

print("Mean final price:", final_values.mean())
print("Median final price:", np.median(final_values))
print("5th percentile:", np.percentile(final_values, 5))
print("95th percentile:", np.percentile(final_values, 95))
print("-")

portfolio_returns = portfolio_paths[1:] / portfolio_paths[:-1] - 1

volatility = np.std(portfolio_returns) * np.sqrt(steps)
print(f"Annualized Volatility {volatility*100:.4f}% ")

risk_free = 0.02
sharpe = ((np.mean(portfolio_returns) * 252) - risk_free) / volatility

print(f"Sharpe ratio: {sharpe:.2f}")
print(f"Probability of loss: {np.mean(final_values < initial_investment)*100:.2f}%")
print("Valid stocks used:", portfolio_size)
print("Weight sum:", valid_weights.sum())
