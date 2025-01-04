//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
#property copyright "Your Name"
#property link      "http://www.yourwebsite.com"
#property version   "1.00"
#property strict

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

// Indicator handles
int maHandle, macdHandle, rsiHandle, stochHandle, bbHandle, atrHandle, sarHandle, ichimokuHandle, adxHandle, obvHandle;

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
    double totalScore = CalculateTotalScore();
    
    MqlTradeRequest request = {};
    MqlTradeResult result = {};
    
    // Check Buy Condition
    if (totalScore >= BuyThreshold && PositionsTotal() == 0)
    {
        Print("Strong Buy Signal: ", totalScore);
        request.action = TRADE_ACTION_DEAL;
        request.symbol = _Symbol;
        request.volume = 0.1;
        request.type = ORDER_TYPE_BUY;
        request.price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
        request.deviation = 3;
        request.magic = 123456;
        
        if (!OrderSend(request, result))
            Print("OrderSend error: ", GetLastError());
        else if (result.retcode != TRADE_RETCODE_DONE)
            Print("OrderSend failed: ", result.retcode);
    }
    
    // Check Sell Condition
    else if (totalScore <= SellThreshold && PositionsTotal() == 0)
    {
        Print("Strong Sell Signal: ", totalScore);
        request.action = TRADE_ACTION_DEAL;
        request.symbol = _Symbol;
        request.volume = 0.1;
        request.type = ORDER_TYPE_SELL;
        request.price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
        request.deviation = 3;
        request.magic = 123456;
        
        if (!OrderSend(request, result))
            Print("OrderSend error: ", GetLastError());
        else if (result.retcode != TRADE_RETCODE_DONE)
            Print("OrderSend failed: ", result.retcode);
    }
}
//+------------------------------------------------------------------+