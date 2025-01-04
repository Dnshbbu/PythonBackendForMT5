//+------------------------------------------------------------------+
//| IndicatorScores.mqh                                              |
//| Separate file for indicator scoring logic                        |
//+------------------------------------------------------------------+
#ifndef INDICATORSCORES_MQH
#define INDICATORSCORES_MQH

#include <Trade\Trade.mqh>

//--- Indicator Handles
int maHandle, macdHandle, rsiHandle, stochHandle, bbHandle, atrHandle, sarHandle, ichimokuHandle, adxHandle, obvHandle;

//+------------------------------------------------------------------+
//| Initialize Indicators                                            |
//+------------------------------------------------------------------+
int InitIndicators()
{
    if(useMA)
        maHandle = iMA(_Symbol, PERIOD_CURRENT, MA_Period, 0, MODE_SMA, PRICE_CLOSE);
    else
        maHandle = INVALID_HANDLE;

    if(useMACD)
        macdHandle = iMACD(_Symbol, PERIOD_CURRENT, MACD_FastEMA, MACD_SlowEMA, MACD_SignalSMA, PRICE_CLOSE);
    else
        macdHandle = INVALID_HANDLE;

    if(useRSI)
        rsiHandle = iRSI(_Symbol, PERIOD_CURRENT, RSI_Period, PRICE_CLOSE);
    else
        rsiHandle = INVALID_HANDLE;

    if(useStoch)
        stochHandle = iStochastic(_Symbol, PERIOD_CURRENT, StochK_Period, StochD_Period, 3, MODE_SMA, STO_LOWHIGH);
    else
        stochHandle = INVALID_HANDLE;

    if(useBB)
        bbHandle = iBands(_Symbol, PERIOD_CURRENT, BB_Period, 0, BB_Deviation, PRICE_CLOSE);
    else
        bbHandle = INVALID_HANDLE;

    if(useATR)
        atrHandle = iATR(_Symbol, PERIOD_CURRENT, ATR_Period);
    else
        atrHandle = INVALID_HANDLE;

    if(useSAR)
        sarHandle = iSAR(_Symbol, PERIOD_CURRENT, 0.02, 0.2);
    else
        sarHandle = INVALID_HANDLE;

    if(useIchimoku)
        ichimokuHandle = iIchimoku(_Symbol, PERIOD_CURRENT, 9, 26, 52);
    else
        ichimokuHandle = INVALID_HANDLE;

    if(useADX)
        adxHandle = iADX(_Symbol, PERIOD_CURRENT, ADX_Period);
    else
        adxHandle = INVALID_HANDLE;

    if(useVolume)
        obvHandle = iOBV(_Symbol, PERIOD_CURRENT, VOLUME_TICK);
    else
        obvHandle = INVALID_HANDLE;

    //--- Error Handling: Check if any handle failed to initialize
    if((useMA && maHandle == INVALID_HANDLE) ||
       (useMACD && macdHandle == INVALID_HANDLE) ||
       (useRSI && rsiHandle == INVALID_HANDLE) ||
       (useStoch && stochHandle == INVALID_HANDLE) ||
       (useBB && bbHandle == INVALID_HANDLE) ||
       (useATR && atrHandle == INVALID_HANDLE) ||
       (useSAR && sarHandle == INVALID_HANDLE) ||
       (useIchimoku && ichimokuHandle == INVALID_HANDLE) ||
       (useADX && adxHandle == INVALID_HANDLE) ||
       (useVolume && obvHandle == INVALID_HANDLE))
    {
        Print("Failed to initialize one or more indicators.");
        return INIT_FAILED;
    }

    return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Release Indicators                                               |
//+------------------------------------------------------------------+
void ReleaseIndicators()
{
    if(useMA && maHandle != INVALID_HANDLE) IndicatorRelease(maHandle);
    if(useMACD && macdHandle != INVALID_HANDLE) IndicatorRelease(macdHandle);
    if(useRSI && rsiHandle != INVALID_HANDLE) IndicatorRelease(rsiHandle);
    if(useStoch && stochHandle != INVALID_HANDLE) IndicatorRelease(stochHandle);
    if(useBB && bbHandle != INVALID_HANDLE) IndicatorRelease(bbHandle);
    if(useATR && atrHandle != INVALID_HANDLE) IndicatorRelease(atrHandle);
    if(useSAR && sarHandle != INVALID_HANDLE) IndicatorRelease(sarHandle);
    if(useIchimoku && ichimokuHandle != INVALID_HANDLE) IndicatorRelease(ichimokuHandle);
    if(useADX && adxHandle != INVALID_HANDLE) IndicatorRelease(adxHandle);
    if(useVolume && obvHandle != INVALID_HANDLE) IndicatorRelease(obvHandle);
}

//+------------------------------------------------------------------+
//| Calculate Moving Average Score                                   |
//+------------------------------------------------------------------+
double CalculateMovingAverageScore()
{
    if(!useMA) return 0.0;

    double maArray[];
    if(CopyBuffer(maHandle, 0, 0, 1, maArray) <= 0)
    {
        Print("Error copying MA buffer: ", GetLastError());
        return 0.0;
    }
    double ma = maArray[0];
    double price = SymbolInfoDouble(_Symbol, SYMBOL_LAST);

    if (price > ma) return 2 * weightMA;
    else if (price < ma) return -2 * weightMA;
    return 0.0;  // Neutral
}

//+------------------------------------------------------------------+
//| Calculate MACD Score                                             |
//+------------------------------------------------------------------+
double CalculateMACDScore()
{
    if(!useMACD) return 0.0;

    double macdMain[], macdSignal[];
    if(CopyBuffer(macdHandle, 0, 0, 1, macdMain) <= 0 ||
       CopyBuffer(macdHandle, 1, 0, 1, macdSignal) <= 0)
    {
        Print("Error copying MACD buffers: ", GetLastError());
        return 0.0;
    }

    if (macdMain[0] > macdSignal[0]) return 2 * weightMACD;
    else if (macdMain[0] < macdSignal[0]) return -2 * weightMACD;
    return 0.0;  // Neutral
}

//+------------------------------------------------------------------+
//| Calculate RSI Score                                              |
//+------------------------------------------------------------------+
double CalculateRSIScore()
{
    if(!useRSI) return 0.0;

    double rsiArray[];
    if(CopyBuffer(rsiHandle, 0, 0, 1, rsiArray) <= 0)
    {
        Print("Error copying RSI buffer: ", GetLastError());
        return 0.0;
    }
    double rsi = rsiArray[0];

    if (rsi < 30) return 2 * weightRSI;  // Oversold, buy signal
    else if (rsi > 70) return -2 * weightRSI;  // Overbought, sell signal
    return 0.0;  // Neutral
}

//+------------------------------------------------------------------+
//| Calculate Stochastic Score                                       |
//+------------------------------------------------------------------+
double CalculateStochasticScore()
{
    if(!useStoch) return 0.0;

    double kArray[], dArray[];
    if(CopyBuffer(stochHandle, 0, 0, 1, kArray) <= 0 ||
       CopyBuffer(stochHandle, 1, 0, 1, dArray) <= 0)
    {
        Print("Error copying Stochastic buffers: ", GetLastError());
        return 0.0;
    }
    double k_value = kArray[0];

    if (k_value < 20) return 2 * weightStoch;  // Oversold
    else if (k_value > 80) return -2 * weightStoch;  // Overbought
    return 0.0;  // Neutral
}

//+------------------------------------------------------------------+
//| Calculate Bollinger Bands Score                                  |
//+------------------------------------------------------------------+
double CalculateBollingerBandsScore()
{
    if(!useBB) return 0.0;

    double upperArray[], lowerArray[];
    if(CopyBuffer(bbHandle, 1, 0, 1, upperArray) <= 0 ||
       CopyBuffer(bbHandle, 2, 0, 1, lowerArray) <= 0)
    {
        Print("Error copying Bollinger Bands buffers: ", GetLastError());
        return 0.0;
    }
    double upper_band = upperArray[0];
    double lower_band = lowerArray[0];
    double price = SymbolInfoDouble(_Symbol, SYMBOL_LAST);

    if (price < lower_band) return 2 * weightBB;  // Buy signal
    else if (price > upper_band) return -2 * weightBB;  // Sell signal
    return 0.0;  // Neutral
}

//+------------------------------------------------------------------+
//| Calculate ATR Score                                              |
//+------------------------------------------------------------------+
double CalculateATRScore()
{
    if(!useATR) return 0.0;

    double atrArray[];
    if(CopyBuffer(atrHandle, 0, 0, 1, atrArray) <= 0)
    {
        Print("Error copying ATR buffer: ", GetLastError());
        return 0.0;
    }
    double atr = atrArray[0];

    if (atr > 0) return 1 * weightATR;  // High volatility, positive signal
    else return -1 * weightATR;  // Low volatility
}

//+------------------------------------------------------------------+
//| Calculate Parabolic SAR Score                                   |
//+------------------------------------------------------------------+
double CalculateParabolicSARScore()
{
    if(!useSAR) return 0.0;

    double sarArray[];
    if(CopyBuffer(sarHandle, 0, 0, 1, sarArray) <= 0)
    {
        Print("Error copying SAR buffer: ", GetLastError());
        return 0.0;
    }
    double sar = sarArray[0];
    double price = SymbolInfoDouble(_Symbol, SYMBOL_LAST);

    if (price > sar) return 2 * weightSAR;  // Uptrend
    else if (price < sar) return -2 * weightSAR;  // Downtrend
    return 0.0;  // Neutral
}

//+------------------------------------------------------------------+
//| Calculate Ichimoku Score                                        |
//+------------------------------------------------------------------+
double CalculateIchimokuScore()
{
    if(!useIchimoku) return 0.0;

    double senkouSpanA[];
    if(CopyBuffer(ichimokuHandle, 0, 0, 1, senkouSpanA) <= 0)
    {
        Print("Error copying Ichimoku buffer: ", GetLastError());
        return 0.0;
    }
    double price = SymbolInfoDouble(_Symbol, SYMBOL_LAST);

    if (price > senkouSpanA[0]) return 2 * weightIchimoku;  // Price above cloud (strong uptrend)
    else if (price < senkouSpanA[0]) return -2 * weightIchimoku;  // Price below cloud (strong downtrend)
    return 0.0;  // Neutral
}

//+------------------------------------------------------------------+
//| Calculate ADX Score                                             |
//+------------------------------------------------------------------+
double CalculateADXScore()
{
    if(!useADX) return 0.0;

    double adxArray[];
    if(CopyBuffer(adxHandle, 0, 0, 1, adxArray) <= 0)
    {
        Print("Error copying ADX buffer: ", GetLastError());
        return 0.0;
    }
    double adx = adxArray[0];

    if (adx > 25) return 2 * weightADX;  // Strong trend
    else if (adx < 20) return -2 * weightADX;  // Weak trend
    return 0.0;  // Neutral
}

//+------------------------------------------------------------------+
//| Calculate Volume (OBV) Score                                     |
//+------------------------------------------------------------------+
double CalculateVolumeScore()
{
    if(!useVolume) return 0.0;

    double obvArray[];
    if(CopyBuffer(obvHandle, 0, 0, 2, obvArray) <= 1)
    {
        Print("Error copying OBV buffers: ", GetLastError());
        return 0.0;
    }
    double obv = obvArray[0];
    double prev_obv = obvArray[1];

    if (obv > prev_obv) return 2 * weightVolume;  // Increasing volume (buy signal)
    else if (obv < prev_obv) return -2 * weightVolume;  // Decreasing volume (sell signal)
    return 0.0;  // Neutral
}

//+------------------------------------------------------------------+
//| Calculate Total Score                                           |
//+------------------------------------------------------------------+
double CalculateTotalScore(double &sumWeights)
{
    double totalScore = 0.0;
    sumWeights = 0.0;

    if(useMA)
    {
        totalScore += CalculateMovingAverageScore();
        sumWeights += weightMA;
    }
    if(useMACD)
    {
        totalScore += CalculateMACDScore();
        sumWeights += weightMACD;
    }
    if(useRSI)
    {
        totalScore += CalculateRSIScore();
        sumWeights += weightRSI;
    }
    if(useStoch)
    {
        totalScore += CalculateStochasticScore();
        sumWeights += weightStoch;
    }
    if(useBB)
    {
        totalScore += CalculateBollingerBandsScore();
        sumWeights += weightBB;
    }
    if(useATR)
    {
        totalScore += CalculateATRScore();
        sumWeights += weightATR;
    }
    if(useSAR)
    {
        totalScore += CalculateParabolicSARScore();
        sumWeights += weightSAR;
    }
    if(useIchimoku)
    {
        totalScore += CalculateIchimokuScore();
        sumWeights += weightIchimoku;
    }
    if(useADX)
    {
        totalScore += CalculateADXScore();
        sumWeights += weightADX;
    }
    if(useVolume)
    {
        totalScore += CalculateVolumeScore();
        sumWeights += weightVolume;
    }

    return totalScore;
}

#endif // INDICATORSCORES_MQH
