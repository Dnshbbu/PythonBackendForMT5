import backtrader as bt
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(filename='all_logs.txt', level=logging.INFO, format='%(asctime)s - %(message)s')

# Parameters
TICKER = 'AAPL'
START_DATE = '2019-01-01'
END_DATE = '2022-06-01'

# Fetch data using yfinance
data = yf.download(TICKER, start=START_DATE, end=END_DATE)
data.reset_index(inplace=True)
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Display the first few rows to verify data
print(data.head())

# Define the strategy
class RefinedBreakoutStrategy(bt.Strategy):
    params = (
        ('resistance_period', 10),        # Lookback period for resistance
        ('volume_multiplier', 1.1),       # Dynamic volume multiplier
        ('bull_run_threshold', 0.03),     # 3% price increase to qualify as bull run
        ('bull_run_period', 10),          # Extended bull run period to 10 days
        ('rsi_period', 14),               # RSI calculation period
        ('rsi_threshold', 55),            # Increased RSI threshold for stronger momentum
        ('ma_period', 50),                # Moving Average period for trend confirmation
        ('macd_fast', 12),                # MACD fast period
        ('macd_slow', 26),                # MACD slow period
        ('macd_signal', 9),                # MACD signal period
        ('atr_period', 14),                # ATR calculation period
        ('atr_threshold', 1.5),            # ATR multiplier for volatility filter
    )
    
    def __init__(self):
        # Highest high over the previous 'resistance_period' bars, excluding current bar
        self.resistance = bt.ind.Highest(self.data.high(-1), period=self.p.resistance_period)
        
        # Simple Moving Average (SMA) of volume over the 'resistance_period'
        self.avg_volume = bt.ind.SMA(self.data.volume, period=self.p.resistance_period)
        
        # Relative Strength Index (RSI)
        self.rsi = bt.ind.RSI_Safe(self.data.close, period=self.p.rsi_period)
        
        # Moving Average (MA) for trend confirmation
        self.ma = bt.ind.SMA(self.data.close, period=self.p.ma_period)
        
        # MACD for momentum confirmation
        self.macd = bt.ind.MACD(self.data.close,
                                 period_me1=self.p.macd_fast,
                                 period_me2=self.p.macd_slow,
                                 period_signal=self.p.macd_signal)
        self.macd_cross = bt.ind.CrossOver(self.macd.macd, self.macd.signal)
        
        # Average True Range (ATR) for volatility filter
        self.atr = bt.ind.ATR(self.data, period=self.p.atr_period)
        
        # Initialize lists to store breakout and bull run information
        self.breakouts = []
        self.bull_runs = []
        self.failures = []  # To store failures and reasons
    
    def next(self):
        # Ensure there is enough data to compute indicators
        required_length = max(self.p.resistance_period, self.p.ma_period, self.p.atr_period, self.p.macd_slow) + 1
        if len(self) < required_length:
            return
        
        current_close = self.data.close[0]
        current_volume = self.data.volume[0]
        current_resistance = self.resistance[0]
        current_avg_volume = self.avg_volume[0]
        current_rsi = self.rsi[0]
        current_ma = self.ma[0]
        current_macd = self.macd.macd[0]
        current_macd_signal = self.macd.signal[0]
        current_macd_cross = self.macd_cross[0]
        current_atr = self.atr[0]
        current_date = self.data.datetime.date(0)
        
        # Dynamic Volume Threshold based on ATR
        dynamic_volume_threshold = self.p.volume_multiplier * current_atr
        
        # Check if current close breaks above the resistance level
        if current_close > current_resistance:
            # Check if volume is significantly higher than average
            if current_volume > dynamic_volume_threshold:
                # Check if RSI is above the threshold indicating bullish momentum
                if current_rsi > self.p.rsi_threshold:
                    # Check if price is above the Moving Average indicating an uptrend
                    if current_close > current_ma:
                        # Check if MACD is above signal line indicating bullish momentum
                        if current_macd > current_macd_signal:
                            breakout_date = current_date
                            breakout_price = current_close
                            self.breakouts.append((breakout_date, breakout_price))
                            
                            # Define the target price for the bull run
                            target_price = breakout_price * (1 + self.p.bull_run_threshold)
                            
                            # Initialize bull run variables
                            bull_run_achieved = False
                            bull_run_date = None
                            bull_run_price = None
                            
                            # Iterate through the next 'bull_run_period' bars to check for bull run
                            for i in range(1, self.p.bull_run_period + 1):
                                if len(self.data) > i:
                                    future_close = self.data.close[i]
                                    if future_close >= target_price:
                                        bull_run_date = self.data.datetime.date(i)
                                        bull_run_price = future_close
                                        self.bull_runs.append((breakout_date, breakout_price, bull_run_date, bull_run_price))
                                        bull_run_achieved = True
                                        break
                            

                            # Logging breakout and bull run information
                            if bull_run_achieved:
                                gain = (bull_run_price - breakout_price) / breakout_price * 100
                                self.failures.append({
                                    'Breakout Date': breakout_date,
                                    'Breakout Price': breakout_price,
                                    'Outcome': 'Success',
                                    'Bull Run Date': bull_run_date,
                                    'Bull Run Price': bull_run_price,
                                    'Gain (%)': round(gain, 2),
                                    'Reason': 'Breakout above resistance with high volume, bullish RSI, MACD positive crossover, and uptrend confirmed by MA.'
                                })
                                log_msg = f"Success: Breakout on {breakout_date} at ${breakout_price:.2f} led to bull run on {bull_run_date} at ${bull_run_price:.2f} (+{gain:.2f}%)"
                                print(log_msg)
                                logging.info(log_msg)
                            else:
                                # Determine the reason for failure
                                last_price = self.data.close[-1]
                                loss = (last_price - breakout_price) / breakout_price * 100
                                self.failures.append({
                                    'Breakout Date': breakout_date,
                                    'Breakout Price': breakout_price,
                                    'Outcome': 'Failure',
                                    'Bull Run Date': None,
                                    'Bull Run Price': None,
                                    'Gain (%)': round(loss, 2),
                                    'Reason': 'Price did not reach target within the bull run period.'
                                })
                                log_msg = f"Failure: Breakout on {breakout_date} at ${breakout_price:.2f} did not lead to a bull run within {self.p.bull_run_period} days. Current Price: ${last_price:.2f} ({loss:.2f}%)"
                                print(log_msg)
                                logging.info(log_msg)
                        else:
                            # MACD below signal line
                            breakout_date = current_date
                            breakout_price = current_close
                            self.breakouts.append((breakout_date, breakout_price))
                            
                            # Record failure
                            last_price = self.data.close[-1]
                            loss = (last_price - breakout_price) / breakout_price * 100
                            self.failures.append({
                                'Breakout Date': breakout_date,
                                'Breakout Price': breakout_price,
                                'Outcome': 'Failure',
                                'Bull Run Date': None,
                                'Bull Run Price': None,
                                'Gain (%)': round(loss, 2),
                                'Reason': 'MACD below signal line indicating bearish momentum.'
                            })
                            log_msg = f"Failure: Breakout on {breakout_date} at ${breakout_price:.2f} did not lead to a bull run because MACD is below the signal line. Current Price: ${last_price:.2f} ({loss:.2f}%)"
                            print(log_msg)
                            logging.info(log_msg)
                    else:
                        # Price not above Moving Average
                        breakout_date = current_date
                        breakout_price = current_close
                        self.breakouts.append((breakout_date, breakout_price))
                        
                        # Record failure
                        last_price = self.data.close[-1]
                        loss = (last_price - breakout_price) / breakout_price * 100
                        self.failures.append({
                            'Breakout Date': breakout_date,
                            'Breakout Price': breakout_price,
                            'Outcome': 'Failure',
                            'Bull Run Date': None,
                            'Bull Run Price': None,
                            'Gain (%)': round(loss, 2),
                            'Reason': 'Price below Moving Average indicating a downtrend.'
                        })
                        log_msg = f"Failure: Breakout on {breakout_date} at ${breakout_price:.2f} did not lead to a bull run because price is below the Moving Average. Current Price: ${last_price:.2f} ({loss:.2f}%)"
                        print(log_msg)
                        logging.info(log_msg)
                else:
                    # RSI below threshold
                    breakout_date = current_date
                    breakout_price = current_close
                    self.breakouts.append((breakout_date, breakout_price))
                    
                    # Record failure
                    last_price = self.data.close[-1]
                    loss = (last_price - breakout_price) / breakout_price * 100
                    self.failures.append({
                        'Breakout Date': breakout_date,
                        'Breakout Price': breakout_price,
                        'Outcome': 'Failure',
                        'Bull Run Date': None,
                        'Bull Run Price': None,
                        'Gain (%)': round(loss, 2),
                        'Reason': 'RSI below threshold indicating bearish momentum.'
                    })
                    log_msg = f"Failure: Breakout on {breakout_date} at ${breakout_price:.2f} did not lead to a bull run because RSI is below the threshold. Current Price: ${last_price:.2f} ({loss:.2f}%)"
                    print(log_msg)
                    logging.info(log_msg)
            else:
                # Volume not sufficient
                breakout_date = current_date
                breakout_price = current_close
                self.breakouts.append((breakout_date, breakout_price))
                
                # Record failure
                last_price = self.data.close[-1]
                loss = (last_price - breakout_price) / breakout_price * 100
                self.failures.append({
                    'Breakout Date': breakout_date,
                    'Breakout Price': breakout_price,
                    'Outcome': 'Failure',
                    'Bull Run Date': None,
                    'Bull Run Price': None,
                    'Gain (%)': round(loss, 2),
                    'Reason': 'Volume not above dynamic threshold.'
                })
                log_msg = f"Failure: Breakout on {breakout_date} at ${breakout_price:.2f} did not lead to a bull run due to insufficient volume. Current Price: ${last_price:.2f} ({loss:.2f}%)"
                print(log_msg)
                logging.info(log_msg)
        else:
            # No breakout detected
            return

# Initialize Cerebro engine
cerebro = bt.Cerebro()

# Create a data feed
data_feed = bt.feeds.PandasData(dataname=data)
cerebro.adddata(data_feed)

# Add the strategy to Cerebro
cerebro.addstrategy(RefinedBreakoutStrategy)

# Run the backtest
results = cerebro.run()
strategy = results[0]  # Access the strategy instance from the result

# Log the total number of breakouts, successful bull runs, and failures
logging.info(f"Total Breakouts: {len(strategy.breakouts)}")
logging.info(f"Total Successful Bull Runs: {len(strategy.bull_runs)}")
logging.info(f"Total Failures: {len(strategy.failures)}")

# Prepare the failure log for printing
failure_log = pd.DataFrame(strategy.failures)
print(failure_log)

# Plotting
fig = go.Figure()

# Add closing prices
fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price'))

# Highlight breakout points
for breakout in strategy.breakouts:
    fig.add_trace(go.Scatter(x=[breakout[0]], y=[breakout[1]], mode='markers', marker=dict(color='red', size=10), name='Breakout'))

# Highlight successful bull run points
for bull_run in strategy.bull_runs:
    fig.add_trace(go.Scatter(x=[bull_run[2]], y=[bull_run[3]], mode='markers', marker=dict(color='green', size=10), name='Bull Run'))

# Update layout
fig.update_layout(title='Breakouts and Bull Runs', xaxis_title='Date', yaxis_title='Price', xaxis_rangeslider_visible=True)

# Show the plot
fig.show()
