import MetaTrader5 as mt5
import datetime

mt5.login(login=87065890, server="MetaQuotes-Demo",password="7iCn*uWb")


rates = mt5.copy_rates_from('TSLA.NAS',mt5.TIMEFRAME_D1,datetime.datetime.now(),100)

print(rates)