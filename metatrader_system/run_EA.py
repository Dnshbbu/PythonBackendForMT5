import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta

# Connect to MetaTrader 5
if not mt5.initialize():
    print("initialize() failed")
    mt5.shutdown()

# Set up the strategy tester parameters
symbol = "EURUSD"  # You can change this to your preferred symbol
timeframe = mt5.TIMEFRAME_M15  # Adjust as needed

# Set the date range for testing
end_date = datetime.now()
start_date = end_date - timedelta(days=365)  # Test for the last year

# Load your EA
ea_path = r"C:\Users\StdUser\AppData\Roaming\MetaQuotes\Terminal\D0E8209F77C8CF37AD8BF550E51FF075\MQL5\Experts\Mul_indicators.ex5"  # Make sure to compile your .mq5 to .ex5

# EA input parameters
ea_inputs = {
    "useMA": True,
    "useMACD": True,
    "useRSI": True,
    "useStoch": True,
    "useBB": True,
    "useATR": True,
    "useVolume": True,
    "useFibo": True,
    "useIchimoku": True,
    "useSAR": True,
    "useADX": True,
    "weightMA": 3.0,
    "weightMACD": 3.0,
    "weightRSI": 5.0,
    "weightStoch": 3.0,
    "weightBB": 4.0,
    "weightATR": 3.0,
    "weightVolume": 3.0,
    "weightFibo": 3.0,
    "weightIchimoku": 1.0,
    "weightSAR": 1.0,
    "weightADX": 5.0,
    "MA_Period": 10,
    "RSI_Period": 7,
    "StochK_Period": 14,
    "StochD_Period": 3,
    "MACD_FastEMA": 12,
    "MACD_SlowEMA": 26,
    "MACD_SignalSMA": 9,
    "BB_Period": 20,
    "BB_Deviation": 2.0,
    "ATR_Period": 14,
    "ADX_Period": 14,
    "BuyThreshold": 40.0,
    "SellThreshold": -40.0,
    "RiskPercentage": 1.0,
    "inputMinLot": 0.01,
    "maxLot": 10000000000.0,
    "lotStepInput": 0.01
}

# Run the strategy tester
result = mt5.strategy_tester(
    ea_path,
    symbol,
    timeframe,
    start_date,
    end_date,
    ea_inputs=ea_inputs,
    initial_deposit=10000,  # Adjust as needed
    spread=10,  # Adjust as needed
)

if result is not None:
    # Process the results
    df = pd.DataFrame(result)
    print(df)
    
    # Save results to CSV
    csv_filename = f"Mul_indicators_results_{symbol}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
    df.to_csv(csv_filename, index=False)
    print(f"Results saved to {csv_filename}")
else:
    print("Strategy tester failed. Error code:", mt5.last_error())

# Shutdown connection
mt5.shutdown()