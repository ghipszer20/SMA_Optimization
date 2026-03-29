# SMA_Optimization

Overview:

For a specified stock or cryptocurrency, backtests 200 SMA strategies (prev. 1 - 200 days) on the full tradeable history of the stock/crypto. Creates two graphs, one of the top 10 strategies by return compared to a buy and hold strategy (return vs. time over full history), and another of the best strategy compared to buy and hold. will seperately print the strategies data, including sharpe & sortino ratios, kelly criterion, max drawdown, and CAGR. Takes spread cost and slippage inputs, and accounts for robinhood fees for each trade. 

Dependencies:
pandas, numpy, yfinance, matplotlib.pyplot

Example Input:

Enter ticker symbol: btc-usd
Asset type? (crypto/stock) [default stock]: crypto
Enter total spread cost as decimal [default 0.0086]: .0086
Enter extra slippage per side as decimal [default 0.0]: 0.01

Disclaimer:

This is not financial advice. Please consult with a financial advisor before making investment decisions.

Liscense:

MIT-liscence
