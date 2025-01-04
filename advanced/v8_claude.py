import backtrader as bt
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

class ResistanceBreakoutStrategy(bt.Strategy):
    params = (
        ('lookback', 20),
        ('volume_factor', 1.5),
        ('profit_target', 0.05),
        ('stop_loss', 0.03),
    )

    def __init__(self):
        self.resistance = bt.indicators.Highest(self.data.high, period=self.p.lookback)
        self.volume_ma = bt.indicators.SMA(self.data.volume, period=self.p.lookback)
        self.order = None
        self.entry_price = None
        self.breakouts = []
        self.successes = 0

    def next(self):
        if self.order:
            return

        if not self.position:
            if (self.data.close[0] > self.resistance[0] and
                self.data.volume[0] > self.volume_ma[0] * self.p.volume_factor):
                self.order = self.buy()
                self.entry_price = self.data.close[0]
                self.breakouts.append(self.data.datetime.date(0))
        else:
            if (self.data.close[0] >= self.entry_price * (1 + self.p.profit_target) or
                self.data.close[0] <= self.entry_price * (1 - self.p.stop_loss)):
                self.order = self.sell()
                if self.data.close[0] >= self.entry_price * (1 + self.p.profit_target):
                    self.successes += 1

    def stop(self):
        self.success_rate = (self.successes / len(self.breakouts)) * 100 if self.breakouts else 0

def run_backtest(lookback, volume_factor, profit_target, stop_loss):
    cerebro = bt.Cerebro()

    data = bt.feeds.PandasData(dataname=yf.download('SPY', start='2022-01-01', end='2022-06-30'))
    cerebro.adddata(data)

    cerebro.addstrategy(ResistanceBreakoutStrategy,
                        lookback=lookback,
                        volume_factor=volume_factor,
                        profit_target=profit_target,
                        stop_loss=stop_loss)

    cerebro.run()

    return cerebro.runstrats[0][0].success_rate, cerebro.runstrats[0][0].breakouts

def optimize_strategy():
    best_params = None
    best_success_rate = 0
    runs = 0

    with open('continual_refining.txt', 'w') as refining_file, open('success_and_failure_reason.txt', 'w') as analysis_file:
        while best_success_rate < 60 and runs < 100:
            lookback = np.random.randint(10, 50)
            volume_factor = np.random.uniform(1.1, 2.0)
            profit_target = np.random.uniform(0.02, 0.1)
            stop_loss = np.random.uniform(0.01, 0.05)

            success_rate, breakouts = run_backtest(lookback, volume_factor, profit_target, stop_loss)
            runs += 1

            refining_file.write(f"Run {runs}: Lookback={lookback}, Volume Factor={volume_factor:.2f}, "
                                f"Profit Target={profit_target:.2f}, Stop Loss={stop_loss:.2f}, "
                                f"Success Rate={success_rate:.2f}%\n")

            if success_rate > best_success_rate:
                best_success_rate = success_rate
                best_params = (lookback, volume_factor, profit_target, stop_loss)

            analysis_file.write(f"Run {runs}:\n")
            if success_rate >= 60:
                analysis_file.write("Success:\n")
                analysis_file.write(f"- Achieved target success rate of {success_rate:.2f}%\n")
                analysis_file.write(f"- Parameters: Lookback={lookback}, Volume Factor={volume_factor:.2f}, "
                                    f"Profit Target={profit_target:.2f}, Stop Loss={stop_loss:.2f}\n")
            else:
                analysis_file.write("Failure:\n")
                analysis_file.write(f"- Success rate of {success_rate:.2f}% is below target\n")
                analysis_file.write("- Possible reasons:\n")
                if success_rate < 30:
                    analysis_file.write("  - Parameters may be too restrictive\n")
                elif success_rate < 45:
                    analysis_file.write("  - Strategy may need fine-tuning\n")
                else:
                    analysis_file.write("  - Close to target, minor adjustments needed\n")

            analysis_file.write("\n")

    return best_params, best_success_rate, runs

best_params, best_success_rate, total_runs = optimize_strategy()

print(f"Optimization completed after {total_runs} runs.")
print(f"Best parameters: Lookback={best_params[0]}, Volume Factor={best_params[1]:.2f}, "
      f"Profit Target={best_params[2]:.2f}, Stop Loss={best_params[3]:.2f}")
print(f"Best success rate: {best_success_rate:.2f}%")

# Run the best strategy and plot the results
success_rate, breakouts = run_backtest(*best_params)

# Fetch data for plotting
data = yf.download('SPY', start='2022-01-01', end='2022-06-30')

# Create the plot
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])

# Candlestick chart
fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name='Price'), row=1, col=1)

# Volume chart
fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name='Volume'), row=2, col=1)

# Add breakout points
for breakout in breakouts:
    fig.add_shape(type="line", x0=breakout, y0=0, x1=breakout, y1=1, xref="x", yref="paper",
                  line=dict(color="green", width=2, dash="dash"))

fig.update_layout(title='SPY Price and Volume with Breakout Points',
                  xaxis_title='Date',
                  yaxis_title='Price',
                  xaxis_rangeslider_visible=False)

fig.update_yaxes(title_text="Volume", row=2, col=1)

fig.show()

print(f"Number of breakouts: {len(breakouts)}")
print(f"Success rate: {success_rate:.2f}%")