import backtrader as bt
import yfinance as yf
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = 'browser'  # Ensures the plot opens in your default browser


# Step 1: Create a custom strategy class for backtesting and storing results
class VIXSNPStrategy(bt.Strategy):
    params = (
        ('vix_pct_threshold', 20),  # 20% change threshold for VIX
    )
    
    def __init__(self):
        self.vix = self.datas[0].close  # VIX data
        self.snp = self.datas[1].close  # S&P 500 data
        self.vix_prev = None
        
        # List to store results for plotting
        self.vix_changes = []
        self.snp_changes = []
        self.dates = []

    def next(self):
        if self.vix_prev is None:
            self.vix_prev = self.vix[0]
            return

        vix_pct_change = (self.vix[0] - self.vix_prev) / self.vix_prev * 100

        if abs(vix_pct_change) >= self.params.vix_pct_threshold:
            snp_pct_change = (self.snp[0] - self.snp[-1]) / self.snp[-1] * 100
            print(f"VIX Change: {vix_pct_change:.2f}% | S&P 500 Change: {snp_pct_change:.2f}%")

            # Store the changes and date for plotting
            self.vix_changes.append(vix_pct_change)
            self.snp_changes.append(snp_pct_change)
            self.dates.append(self.datas[0].datetime.date(0))

        self.vix_prev = self.vix[0]

    def stop(self):
        # Create a Plotly interactive scatter plot
        fig = go.Figure()

        # Add VIX changes as a scatter plot
        fig.add_trace(go.Scatter(
            x=self.dates,
            y=self.vix_changes,
            mode='markers+lines',
            name='VIX Change (%)',
            marker=dict(color='blue', size=10),
            hovertemplate='%{x}<br>VIX Change: %{y:.2f}%'  # Show VIX change on hover
        ))

        # Add S&P 500 changes as a scatter plot
        fig.add_trace(go.Scatter(
            x=self.dates,
            y=self.snp_changes,
            mode='markers+lines',
            name="S&P 500 Change (%)",
            marker=dict(color='red', size=10),
            hovertemplate='%{x}<br>S&P 500 Change: %{y:.2f}%'  # Show S&P 500 change on hover
        ))

        # Update layout
        fig.update_layout(
            title="VIX vs S&P 500 Percentage Change when VIX changes by 20% or more",
            xaxis_title="Date",
            yaxis_title="Percentage Change",
            hovermode="x",  # Hover over both VIX and S&P 500 on the same date
            legend=dict(x=0.05, y=0.95),
            plot_bgcolor="white",
            font=dict(family="Arial", size=12)
        )

        # Show the interactive plot
        fig.show()

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
