import backtrader as bt
import yfinance as yf

# Step 1: Create a custom strategy class for backtesting
class VIXSNPStrategy(bt.Strategy):
    params = (
        ('vix_pct_threshold', 20),  # 20% change threshold for VIX
    )
    
    def __init__(self):
        # Keep track of VIX and S&P 500 price data
        self.vix = self.datas[0].close  # VIX data
        self.snp = self.datas[1].close  # S&P 500 data
        self.vix_prev = None

    def next(self):
        # Skip if we don't have previous VIX data
        if self.vix_prev is None:
            self.vix_prev = self.vix[0]
            return

        # Calculate VIX percentage change
        vix_pct_change = (self.vix[0] - self.vix_prev) / self.vix_prev * 100

        # Check if VIX has changed by 20% or more
        if abs(vix_pct_change) >= self.params.vix_pct_threshold:
            snp_pct_change = (self.snp[0] - self.snp[-1]) / self.snp[-1] * 100
            print(f"VIX Change: {vix_pct_change:.2f}% | S&P 500 Change: {snp_pct_change:.2f}%")

        # Update previous VIX value
        self.vix_prev = self.vix[0]

# Step 2: Fetch historical data using yfinance
def get_data():
    vix_data = bt.feeds.PandasData(dataname=yf.download('^VIX', start='2000-01-01', end='2023-01-01'))
    snp_data = bt.feeds.PandasData(dataname=yf.download('^GSPC', start='2000-01-01', end='2023-01-01'))
    return vix_data, snp_data

# Step 3: Set up and run the backtest
def run_backtest():
    # Initialize Cerebro engine
    cerebro = bt.Cerebro()
    
    # Add the strategy
    cerebro.addstrategy(VIXSNPStrategy)
    
    # Get data and add to Cerebro
    vix_data, snp_data = get_data()
    cerebro.adddata(vix_data)  # Add VIX data
    cerebro.adddata(snp_data)  # Add S&P 500 data
    
    # Run the backtest
    cerebro.run()

# Execute the backtest
if __name__ == '__main__':
    run_backtest()
