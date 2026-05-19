#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 22:42:53 2026

@author: dwpet
"""

import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("C:\\Users\\dwpet\\Downloads\\oslo_.csv",delimiter=";")
df["Symbol"] = df["Symbol"].astype(str)+ df["Oslo"].astype(str)
print(df.head())

stocks = df["Symbol"].dropna().tolist()
    
returns_df = pd.DataFrame()
for ticker in stocks:
    stock = yf.download(ticker, period="5y", auto_adjust=True)["Close"]

    if stock.empty:
        print("Skipped:", ticker)
        continue
    
    returns_df[ticker] = stock.pct_change()

returns_df = returns_df.dropna(axis=1, tresh=252*3)
returns_df = returns_df.fillna(0)

stocks = returns_df.columns.tolist()    
cov_matrix = returns_df.cov() * 252
expected_returns = np.clip(returns_df.mean() * 252, -0.20, 0.25)

num_portfolios = 10000

results = []
all_weights = []
risk_free = 0.02

for _ in range(num_portfolios):

    weights = np.random.random(len(stocks))
    weights /= np.sum(weights)

    portfolio_return = np.dot(weights, expected_returns)

    portfolio_volatility = np.sqrt(
        weights.T @ cov_matrix @ weights
    )

    sharpe = (portfolio_return-risk_free) / portfolio_volatility

    results.append([
        portfolio_return,
        portfolio_volatility,
        sharpe
    ])

    all_weights.append(weights)

results = np.array(results)
all_weights = np.array(all_weights)

plt.scatter(
    results[:,1],
    results[:,0],
    c=results[:,2]
)
top_5_idx = np.argsort(results[:, 2])[-5:][::-1]

for rank, idx in enumerate(top_5_idx, start=1):
    print(f"\nPortfolio #{rank}")
    print(f"Expected return: {results[idx, 0]*100:.2f}%")
    print(f"Volatility: {results[idx, 1]*100:.2f}%")
    print(f"Sharpe ratio: {results[idx, 2]:.2f}")

    weights = all_weights[idx]

    portfolio_details = pd.DataFrame({
        "Stock": stocks,
        "Weight": weights
    })

    portfolio_details = portfolio_details.sort_values(
        by="Weight",
        ascending=False
    )
    portfolio_details["Weight"] = portfolio_details["Weight"] * 100

    print(portfolio_details.head(10))

plt.xlabel("Volatility")
plt.ylabel("Expected Return")
plt.title("Efficient Frontier")
plt.colorbar(label="Sharpe Ratio")

max_sharpe_idx = np.argmax(results[:,2])

best_portfolio = results[max_sharpe_idx]
print(portfolio_details.head(10).to_string(index=False))
print(best_portfolio)
plt.show()