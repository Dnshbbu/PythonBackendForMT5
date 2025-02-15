//+------------------------------------------------------------------+
//| Expert Advisor with Dynamic Lot Sizing                           |
//+------------------------------------------------------------------+
#property copyright "Your Name"
#property link      "http://www.yourwebsite.com"
#property version   "1.03"
#property strict

#include <Trade\Trade.mqh>

input double weightMA = 5.0;          // Weight for Moving Average
input double weightMACD = 5.0;        // Weight for MACD
input double weightRSI = 4.0;         // Weight for RSI
input double weightStoch = 4.0;       // Weight for Stochastic
input double weightBB = 3.0;          // Weight for Bollinger Bands
input double weightATR = 3.0;         // Weight for ATR
input double weightVolume = 3.0;      // Weight for OBV
input double weightFibo = 3.0;        // Weight for Fibonacci
input double weightIchimoku = 5.0;    // Weight for Ichimoku Cloud
input double weightSAR = 5.0;         // Weight for Parabolic SAR
input double weightADX = 5.0;         // Weight for ADX

// Indicator settings
input int MA_Period = 50;
input int RSI_Period = 14;
input int StochK_Period = 14;
input int StochD_Period = 3;
input int MACD_FastEMA = 12;
input int MACD_SlowEMA = 26;
input int MACD_SignalSMA = 9;
input int BB_Period = 20;
input double BB_Deviation = 2.0;
input int ATR_Period = 14;
input int ADX_Period = 14;

// Thresholds for decision making
input double BuyThreshold = 40.0;    // Strong Buy threshold
input double SellThreshold = -40.0;  // Strong Sell threshold
input double RiskPercentage = 2.0;   // Risk percentage per trade (0-100)

// Indicator handles
int maHandle, macdHandle, rsiHandle, stochHandle, bbHandle, atrHandle, sarHandle, ichimokuHandle, adxHandle, obvHandle;

// Trading object
CTrade trade;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
    // Initialize indicator handles
    maHandle = iMA(_Symbol, PERIOD_CURRENT, MA_Period, 0, MODE_SMA, PRICE_CLOSE);
    macdHandle = iMACD(_Symbol, PERIOD_CURRENT, MACD_FastEMA, MACD_SlowEMA, MACD_SignalSMA, PRICE_CLOSE);
    rsiHandle = iRSI(_Symbol, PERIOD_CURRENT, RSI_Period, PRICE_CLOSE);
    stochHandle = iStochastic(_Symbol, PERIOD_CURRENT, StochK_Period, StochD_Period, 3, MODE_SMA, STO_LOWHIGH);
    bbHandle = iBands(_Symbol, PERIOD_CURRENT, BB_Period, 0, BB_Deviation, PRICE_CLOSE);
    atrHandle = iATR(_Symbol, PERIOD_CURRENT, ATR_Period);
    sarHandle = iSAR(_Symbol, PERIOD_CURRENT, 0.02, 0.2);
    ichimokuHandle = iIchimoku(_Symbol, PERIOD_CURRENT, 9, 26, 52);
    adxHandle = iADX(_Symbol, PERIOD_CURRENT, ADX_Period);
    obvHandle = iOBV(_Symbol, PERIOD_CURRENT, VOLUME_TICK);
    
    return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    // Release indicator handles
    IndicatorRelease(maHandle);
    IndicatorRelease(macdHandle);
    IndicatorRelease(rsiHandle);
    IndicatorRelease(stochHandle);
    IndicatorRelease(bbHandle);
    IndicatorRelease(atrHandle);
    IndicatorRelease(sarHandle);
    IndicatorRelease(ichimokuHandle);
    IndicatorRelease(adxHandle);
    IndicatorRelease(obvHandle);
}

//+------------------------------------------------------------------+
//| Calculate Lot Size Function                                      |
//+------------------------------------------------------------------+
double CalculateLotSize()
{
    double balance = AccountInfoDouble(ACCOUNT_BALANCE);
    double equity = AccountInfoDouble(ACCOUNT_EQUITY);
    double freeMargin = AccountInfoDouble(ACCOUNT_MARGIN_FREE);
    
    double tickSize = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
    double tickValue = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
    double minLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
    double maxLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
    double lotStep = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
    
    double riskAmount = equity * (RiskPercentage / 100);
    double lotSize = NormalizeDouble(riskAmount / (tickValue / tickSize), 2);
    
    // Ensure lot size is within allowed range and step
    lotSize = MathMax(minLot, MathMin(maxLot, lotSize));
    lotSize = NormalizeDouble(lotSize / lotStep, 0) * lotStep;
    
    return lotSize;
}

//+------------------------------------------------------------------+
//| Indicator Calculation Functions                                  |
//+------------------------------------------------------------------+

// Moving Average (MA) Score
double CalculateMovingAverageScore()
{
    double maArray[];
    CopyBuffer(maHandle, 0, 0, 1, maArray);
    double ma = maArray[0];
    double price = SymbolInfoDouble(_Symbol, SYMBOL_LAST);
    
    if (price > ma) return 2 * weightMA;
    else if (price < ma) return -2 * weightMA;
    return 0;  // Neutral
}

// MACD Score
double CalculateMACDScore()
{
    double macdMain[], macdSignal[];
    CopyBuffer(macdHandle, 0, 0, 1, macdMain);
    CopyBuffer(macdHandle, 1, 0, 1, macdSignal);
    
    if (macdMain[0] > macdSignal[0]) return 2 * weightMACD;
    else if (macdMain[0] < macdSignal[0]) return -2 * weightMACD;
    return 0;  // Neutral
}

// RSI Score
double CalculateRSIScore()
{
    double rsiArray[];
    CopyBuffer(rsiHandle, 0, 0, 1, rsiArray);
    double rsi = rsiArray[0];
    
    if (rsi < 30) return 2 * weightRSI;  // Oversold, buy signal
    else if (rsi > 70) return -2 * weightRSI;  // Overbought, sell signal
    return 0;  // Neutral
}

// Stochastic Oscillator Score
double CalculateStochasticScore()
{
    double kArray[], dArray[];
    CopyBuffer(stochHandle, 0, 0, 1, kArray);
    CopyBuffer(stochHandle, 1, 0, 1, dArray);
    double k_value = kArray[0];
    
    if (k_value < 20) return 2 * weightStoch;  // Oversold
    else if (k_value > 80) return -2 * weightStoch;  // Overbought
    return 0;  // Neutral
}

// Bollinger Bands Score
double CalculateBollingerBandsScore()
{
    double upperArray[], lowerArray[];
    CopyBuffer(bbHandle, 1, 0, 1, upperArray);
    CopyBuffer(bbHandle, 2, 0, 1, lowerArray);
    double upper_band = upperArray[0];
    double lower_band = lowerArray[0];
    double price = SymbolInfoDouble(_Symbol, SYMBOL_LAST);
    
    if (price < lower_band) return 2 * weightBB;  // Buy signal
    else if (price > upper_band) return -2 * weightBB;  // Sell signal
    return 0;  // Neutral
}

// Average True Range (ATR) Score
double CalculateATRScore()
{
    double atrArray[];
    CopyBuffer(atrHandle, 0, 0, 1, atrArray);
    double atr = atrArray[0];
    
    if (atr > 0) return 1 * weightATR;  // High volatility, positive signal
    else return -1 * weightATR;  // Low volatility
}

// Parabolic SAR Score
double CalculateParabolicSARScore()
{
    double sarArray[];
    CopyBuffer(sarHandle, 0, 0, 1, sarArray);
    double sar = sarArray[0];
    double price = SymbolInfoDouble(_Symbol, SYMBOL_LAST);
    
    if (price > sar) return 2 * weightSAR;  // Uptrend
    else if (price < sar) return -2 * weightSAR;  // Downtrend
    return 0;  // Neutral
}

// Ichimoku Cloud Score
double CalculateIchimokuScore()
{
    double senkouSpanA[];
    CopyBuffer(ichimokuHandle, 0, 0, 1, senkouSpanA);
    double price = SymbolInfoDouble(_Symbol, SYMBOL_LAST);
    
    if (price > senkouSpanA[0]) return 2 * weightIchimoku;  // Price above cloud (strong uptrend)
    else if (price < senkouSpanA[0]) return -2 * weightIchimoku;  // Price below cloud (strong downtrend)
    return 0;  // Neutral
}

// ADX Score
double CalculateADXScore()
{
    double adxArray[];
    CopyBuffer(adxHandle, 0, 0, 1, adxArray);
    double adx = adxArray[0];
    
    if (adx > 25) return 2 * weightADX;  // Strong trend
    else if (adx < 20) return -2 * weightADX;  // Weak trend
    return 0;  // Neutral
}

// OBV (Volume) Score
double CalculateVolumeScore()
{
    double obvArray[];
    CopyBuffer(obvHandle, 0, 0, 2, obvArray);
    double obv = obvArray[0];
    double prev_obv = obvArray[1];
    
    if (obv > prev_obv) return 2 * weightVolume;  // Increasing volume (buy signal)
    else if (obv < prev_obv) return -2 * weightVolume;  // Decreasing volume (sell signal)
    return 0;  // Neutral
}

//+------------------------------------------------------------------+
//| Calculate Total Score Function                                   |
//+------------------------------------------------------------------+
double CalculateTotalScore()
{
    double totalScore = 0;
    
    totalScore += CalculateMovingAverageScore();
    totalScore += CalculateMACDScore();
    totalScore += CalculateRSIScore();
    totalScore += CalculateStochasticScore();
    totalScore += CalculateBollingerBandsScore();
    totalScore += CalculateATRScore();
    totalScore += CalculateParabolicSARScore();
    totalScore += CalculateIchimokuScore();
    totalScore += CalculateADXScore();
    totalScore += CalculateVolumeScore();
    
    return totalScore;
}

//+------------------------------------------------------------------+
//| Expert Tick Function (Main Logic)                                |
//+------------------------------------------------------------------+
void OnTick()
{
    // Check if we're allowed to trade
    if(!TerminalInfoInteger(TERMINAL_TRADE_ALLOWED)) return;

    // Calculate the total score
    double totalScore = CalculateTotalScore();
    
    // Get the current open positions
    int totalPositions = PositionsTotal();
    
    // Calculate the appropriate lot size
    double dynamicLotSize = CalculateLotSize();
    
    // Debug output
    Print("Total Score: ", totalScore, " | Open Positions: ", totalPositions, " | Dynamic Lot Size: ", dynamicLotSize);
    
    // Check Buy Condition
    if(totalScore >= BuyThreshold && totalPositions == 0)
    {
        Print("Attempting to open Buy position. Score: ", totalScore, " | Lot Size: ", dynamicLotSize);
        if(trade.Buy(dynamicLotSize, _Symbol, 0, 0, 0, "Buy"))
        {
            Print("Buy order placed successfully. Ticket: ", trade.ResultOrder(), " | Price: ", trade.ResultPrice());
        }
        else
        {
            Print("Error placing Buy order. Error code: ", GetLastError());
        }
    }
    
    // Check Sell Condition
    else if(totalScore <= SellThreshold && totalPositions == 0)
    {
        Print("Attempting to open Sell position. Score: ", totalScore, " | Lot Size: ", dynamicLotSize);
        if(trade.Sell(dynamicLotSize, _Symbol, 0, 0, 0, "Sell"))
        {
            Print("Sell order placed successfully. Ticket: ", trade.ResultOrder(), " | Price: ", trade.ResultPrice());
        }
        else
        {
            Print("Error placing Sell order. Error code: ", GetLastError());
        }
    }
    
    // Check for position closure
    else if(totalPositions > 0)
    {
        ulong ticket = PositionGetTicket(0);
        if(PositionSelectByTicket(ticket))
        {
            long positionType = PositionGetInteger(POSITION_TYPE);
            if((totalScore < 0 && positionType == POSITION_TYPE_BUY) || 
               (totalScore > 0 && positionType == POSITION_TYPE_SELL))
            {
                Print("Attempting to close position. Score: ", totalScore);
                if(trade.PositionClose(ticket))
                {
                    Print("Position closed successfully");
                }
                else
                {
                    Print("Error closing position. Error code: ", GetLastError());
                }
            }
        }
    }
}
//+------------------------------------------------------------------+