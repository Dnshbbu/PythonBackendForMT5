import MetaTrader5 as mt5

# Connect to the MetaTrader 5 terminal
if not mt5.initialize():
    print("Initialize failed")
    mt5.shutdown()
    exit()

# Set the parameters for the strategy tester
# Replace 'YourExpertAdvisorName.ex5' with your EA filename
ea_name = 'main_EA.ex5'

# Start the strategy tester
result = mt5.tester_start(
    ea_name, 
    "symbol=EURUSD; timeframe=M1; date_start=2023.01.01; date_end=2023.12.31; equity=10000; balance=10000"
)

if result:
    print("Strategy Tester initiated successfully.")
else:
    print("Failed to start Strategy Tester. Error:", mt5.last_error())

# Shut down the connection
mt5.shutdown()
