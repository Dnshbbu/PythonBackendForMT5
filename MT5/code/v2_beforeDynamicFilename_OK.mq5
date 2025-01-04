//+------------------------------------------------------------------+
//| Expert Advisor with multiple indicators and dynamic scoring      |
//+------------------------------------------------------------------+
#property copyright "Dinesh"
#property link      "http://www.yourwebsite.com"
#property version   "1.05"
#property strict

#include <Trade\Trade.mqh>
#include <Files\File.mqh>

// Lot size settings
input double minLot = 0.01;               // Minimum lot size (can be overridden by SYMBOL_VOLUME_MIN)
input double maxLot = 10000000000.0;      // Maximum lot size (adjust if needed)
input double lotStepInput = 0.01;         // Lot step size (can be overridden by SYMBOL_VOLUME_STEP)

// Indicator usage flags
input bool useMA = true;                  // Use Moving Average
input bool useMACD = true;                // Use MACD
input bool useRSI = true;                 // Use RSI
input bool useStoch = true;               // Use Stochastic
input bool useBB = true;                  // Use Bollinger Bands
input bool useATR = true;                 // Use ATR
input bool useVolume = true;              // Use OBV (On-Balance Volume)
input bool useFibo = true;                // Use Fibonacci
input bool useIchimoku = true;            // Use Ichimoku Cloud
input bool useSAR = true;                 // Use Parabolic SAR
input bool useADX = true;                 // Use ADX

// Weights for indicators
input double weightMA = 5.0;          // Weight for Moving Average
input double weightMACD = 5.0;        // Weight for MACD
input double weightRSI = 4.0;         // Weight for RSI
input double weightStoch = 4.0;       // Weight for Stochastic
input double weightBB = 3.0;          // Weight for Bollinger Bands
input double weightATR = 3.0;         // Weight for ATR
input double weightVolume = 3.0;      // Weight for OBV
input double weightFibo = 3.0;        // Weight for Fibonacci
input double weightIchimoku = 1.0;    // Weight for Ichimoku Cloud
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
input double RiskPercentage = 1.0;   // Risk percentage per trade (0-100)

// Indicator handles
int maHandle, macdHandle, rsiHandle, stochHandle, bbHandle, atrHandle, sarHandle, ichimokuHandle, adxHandle, obvHandle;

// Trading object
CTrade trade;

// Global variables for storing scores
string csvFileName = "trading_scores_all.csv";
double lastBuyScore = 0;
double lastSellScore = 0;

// Constant for original total weight (sum of all weights)
const double originalSumWeights = 44.0;

//+------------------------------------------------------------------+
//| Custom Round Function to Round to Nearest Lot Step               |
//+------------------------------------------------------------------+
double RoundLotSize(double lotSize, double lotStep, double minLot)
{
    // Round the lot size to the nearest lot step
    lotSize = MathRound(lotSize / lotStep) * lotStep;

    // Normalize the lot size to the number of decimal places in lotStep
    // For example, if lotStep is 0.01, normalize to 2 decimal places
    int decimals = 0;
    double tempStep = lotStep;
    while(MathRound(tempStep) != tempStep && decimals < 10)
    {
        tempStep *= 10;
        decimals++;
    }
    lotSize = NormalizeDouble(lotSize, decimals);

    // Ensure lot size is not less than the minimum allowed lot size
    if(lotSize < minLot)
    {
        lotSize = minLot;
    }

    return lotSize;
}

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
    // Initialize indicator handles based on usage flags
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

    // Create CSV file header if it doesn't exist
    if(!FileIsExist(csvFileName))
    {
        int fileHandle = FileOpen(csvFileName, FILE_WRITE | FILE_CSV |FILE_ANSI| FILE_COMMON, ',');
        if(fileHandle != INVALID_HANDLE)
        {
            string header = "Date,Time,Type,MA Score,MACD Score,RSI Score,Stoch Score,BB Score,ATR Score,SAR Score,Ichimoku Score,ADX Score,Volume Score,Total Score";
            FileWrite(fileHandle, header);
            FileClose(fileHandle);
        } 
        else 
        {
            Print("Error creating CSV file: ", GetLastError());
        }
    }

    return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    // Release indicator handles if they were initialized
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
//| Calculate Lot Size Function                                      |
//+------------------------------------------------------------------+
double CalculateLotSize()
{
    double balance = AccountInfoDouble(ACCOUNT_BALANCE); // Total available capital
    double currentPrice = SymbolInfoDouble(_Symbol, SYMBOL_BID); // Get the current price

    // Calculate initial lot size as total available capital divided by current price
    double lotSize = balance / currentPrice;

    // Retrieve symbol-specific lot size settings
    double symbolMinLot, symbolLotStep;
    bool isVolumeInfo = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN, symbolMinLot) &&
                        SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP, symbolLotStep);

    if(!isVolumeInfo)
    {
        // Fallback to input settings if symbol-specific info is unavailable
        symbolMinLot = minLot;
        symbolLotStep = lotStepInput;
        Print("Warning: Could not retrieve symbol-specific volume info. Using input settings.");
    }

    // Round lot size to the nearest valid lot step
    lotSize = RoundLotSize(lotSize, symbolLotStep, symbolMinLot);

    // Ensure lot size is within allowed range
    lotSize = MathMax(symbolMinLot, MathMin(maxLot, lotSize));

    // Debug: Print the rounded lot size
    PrintFormat("Calculated Lot Size: %.2f (Min: %.2f, Step: %.5f)", lotSize, symbolMinLot, symbolLotStep);

    return lotSize;
}

//+------------------------------------------------------------------+
//| Indicator Calculation Functions                                  |
//+------------------------------------------------------------------+

// Moving Average (MA) Score
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

// MACD Score
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

// RSI Score
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

// Stochastic Oscillator Score
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

// Bollinger Bands Score
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

// Average True Range (ATR) Score
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

// Parabolic SAR Score
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

// Ichimoku Cloud Score
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

// ADX Score
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

// OBV (Volume) Score
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
//| Calculate Total Score Function                                   |
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

//+------------------------------------------------------------------+
//| Expert Tick Function (Main Logic)                                |
//+------------------------------------------------------------------+
void OnTick()
{
    // Check if we're allowed to trade
    if(!TerminalInfoInteger(TERMINAL_TRADE_ALLOWED)) return;

    // Calculate the total score and sum of active weights
    double sumWeights = 0.0;
    double totalScore = CalculateTotalScore(sumWeights);

    // Avoid division by zero if no indicators are active
    if(sumWeights == 0.0)
    {
        Print("No indicators are active. Please enable at least one indicator.");
        return;
    }

    // Adjust thresholds based on active weights
    double dynamicBuyThreshold = BuyThreshold * sumWeights / originalSumWeights;
    double dynamicSellThreshold = SellThreshold * sumWeights / originalSumWeights;

    // Get the current open positions
    int totalPositions = PositionsTotal();

    // Calculate the appropriate lot size
    double dynamicLotSize = CalculateLotSize();

    // Debug output
    PrintFormat("Total Score: %.2f | Sum Weights: %.2f | Buy Threshold: %.2f | Sell Threshold: %.2f | Open Positions: %d | Dynamic Lot Size: %.5f",
                totalScore, sumWeights, dynamicBuyThreshold, dynamicSellThreshold, totalPositions, dynamicLotSize);

    // Check Buy Condition
    if(totalScore >= dynamicBuyThreshold && totalPositions == 0)
    {
        PrintFormat("Attempting to open Buy position. Score: %.2f | Lot Size: %.5f", totalScore, dynamicLotSize);

        if(trade.Buy(dynamicLotSize, _Symbol, 0, 0, 0, "Buy"))
        {
            PrintFormat("Buy order placed successfully. Ticket: %llu | Price: %.5f", trade.ResultOrder(), trade.ResultPrice());
            lastBuyScore = totalScore; // Store buy score
            WriteScoresToCSV("Buy", totalScore); // Write scores to CSV
            Print("Wrote Buy scores to CSV. Total Score: ", totalScore);
        }
        else
        {
            PrintFormat("Error placing Buy order. Error code: %d", GetLastError());
        }
    }

    // Check Sell Condition
    else if(totalScore <= dynamicSellThreshold && totalPositions == 0)
    {
        PrintFormat("Attempting to open Sell position. Score: %.2f | Lot Size: %.5f", totalScore, dynamicLotSize);
        if(trade.Sell(dynamicLotSize, _Symbol, 0, 0, 0, "Sell"))
        {
            PrintFormat("Sell order placed successfully. Ticket: %llu | Price: %.5f", trade.ResultOrder(), trade.ResultPrice());
            lastSellScore = totalScore; // Store sell score
            WriteScoresToCSV("Sell", totalScore); // Write scores to CSV
            Print("Wrote Sell scores to CSV. Total Score: ", totalScore);
        }
        else
        {
            PrintFormat("Error placing Sell order. Error code: %d", GetLastError());
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
                PrintFormat("Attempting to close position. Score: %.2f", totalScore);
                if(trade.PositionClose(ticket))
                {
                    Print("Position closed successfully");
                    // Determine if it was a buy or sell that was closed and write the appropriate score
                    if (positionType == POSITION_TYPE_BUY) {
                        WriteScoresToCSV("Sell (Close Buy)", totalScore); 
                        Print("Wrote Sell (Close Buy) scores to CSV. Total Score: ", totalScore);
                    } else {
                        WriteScoresToCSV("Buy (Close Sell)", totalScore);
                        Print("Wrote Buy (Close Sell) scores to CSV. Total Score: ", totalScore);
                    }
                }
                else
                {
                    PrintFormat("Error closing position. Error code: %d", GetLastError());
                }
            }
        }
    }
}

//+------------------------------------------------------------------+
//| Write Scores to CSV File                                         |
//+------------------------------------------------------------------+
void WriteScoresToCSV(string type, double totalScore)
{
    int fileHandle = FileOpen(csvFileName, FILE_CSV | FILE_READ | FILE_WRITE | FILE_COMMON| FILE_ANSI);

    if(fileHandle != INVALID_HANDLE)
    {
        Print("Reached WriteScoresToCSV function");
        MqlDateTime dt;
        TimeToStruct(TimeCurrent(), dt);
        string dateStr = StringFormat("%04d-%02d-%02d", dt.year, dt.mon, dt.day);
        string timeStr = StringFormat("%02d:%02d", dt.hour, dt.min);

        // Move to the end for appending
        FileSeek(fileHandle, 0, SEEK_END);

        // Prepare score values, setting to 0 if the indicator is not used
        double maScore = useMA ? CalculateMovingAverageScore() : 0.0;
        double macdScore = useMACD ? CalculateMACDScore() : 0.0;
        double rsiScore = useRSI ? CalculateRSIScore() : 0.0;
        double stochScore = useStoch ? CalculateStochasticScore() : 0.0;
        double bbScore = useBB ? CalculateBollingerBandsScore() : 0.0;
        double atrScore = useATR ? CalculateATRScore() : 0.0;
        double sarScore = useSAR ? CalculateParabolicSARScore() : 0.0;
        double ichimokuScore = useIchimoku ? CalculateIchimokuScore() : 0.0;
        double adxScore = useADX ? CalculateADXScore() : 0.0;
        double volumeScore = useVolume ? CalculateVolumeScore() : 0.0;

        string scoreString = StringFormat("%s,%s,%s,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n",
                                           dateStr, timeStr, type,
                                           maScore, macdScore,
                                           rsiScore, stochScore,
                                           bbScore, atrScore,
                                           sarScore, ichimokuScore,
                                           adxScore, volumeScore,
                                           totalScore);

        FileWriteString(fileHandle, scoreString);
        FileClose(fileHandle);
    }
    else
    {
        Print("Error opening CSV file: ", GetLastError());
    }
}
