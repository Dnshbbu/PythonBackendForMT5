//+------------------------------------------------------------------+
//| Expert Advisor with multiple indicators and dynamic scoring      |
//+------------------------------------------------------------------+
#property copyright "Dinesh"
#property link      "http://www.yourwebsite.com"
#property version   "1.07"
#property strict

#include <Trade\Trade.mqh>
#include <Files\File.mqh>
#include <MulIndicatorScores.mqh>  // Include the indicator scoring logic

//--- Global Variables
datetime startTime;
datetime endTime;
datetime userTime = TimeLocal();

//--- Lot size settings
input double inputMinLot = 0.01;          // Minimum lot size (can be overridden by SYMBOL_VOLUME_MIN)
input double maxLot = 10000000000.0;      // Maximum lot size (adjust if needed)
input double lotStepInput = 0.01;         // Lot step size (can be overridden by SYMBOL_VOLUME_STEP)

//--- Indicator usage flags
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

//--- Weights for indicators
input double weightMA = 3.0;          // Weight for Moving Average
input double weightMACD = 3.0;        // Weight for MACD
input double weightRSI = 5.0;         // Weight for RSI
input double weightStoch = 3.0;       // Weight for Stochastic
input double weightBB = 4.0;          // Weight for Bollinger Bands
input double weightATR = 3.0;         // Weight for ATR
input double weightVolume = 3.0;      // Weight for OBV
input double weightFibo = 3.0;        // Weight for Fibonacci
input double weightIchimoku = 1.0;    // Weight for Ichimoku Cloud
input double weightSAR = 1.0;         // Weight for Parabolic SAR
input double weightADX = 5.0;         // Weight for ADX

//--- Indicator settings
input int MA_Period = 10;
input int RSI_Period = 7;
input int StochK_Period = 14;
input int StochD_Period = 3;
input int MACD_FastEMA = 12;
input int MACD_SlowEMA = 26;
input int MACD_SignalSMA = 9;
input int BB_Period = 20;
input double BB_Deviation = 2.0;
input int ATR_Period = 14;
input int ADX_Period = 14;

//--- Thresholds for decision making
input double BuyThreshold = 40.0;    // Strong Buy threshold
input double SellThreshold = -40.0;  // Strong Sell threshold
input double RiskPercentage = 1.0;   // Risk percentage per trade (0-100)

//--- Trading object
CTrade trade;

//--- Global variables for storing scores
string dynamicCsvFileName;
double lastBuyScore = 0;
double lastSellScore = 0;

//--- Constant for original total weight (sum of all weights)
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
    // Initialize indicators using the included IndicatorScores.mqh file
    int initResult = InitIndicators();
    if(initResult != INIT_SUCCEEDED)
    {
        Print("Failed to initialize indicators.");
        return INIT_FAILED;
    }

    // Generate the dynamic CSV filename
    string symbol = _Symbol;
    startTime = TimeCurrent(); // Use current time as start time
    endTime = TimeCurrent(); 

    string mqlFilename = MQLInfoString(MQL_PROGRAM_NAME);

    dynamicCsvFileName = StringFormat("%s_%s_%s_%s.csv", 
        symbol,
        TimeToString(startTime, TIME_DATE),
        TimeToString(endTime, TIME_DATE),
        mqlFilename
    );

    // Replace spaces and colons with underscores in the filename
    StringReplace(dynamicCsvFileName, " ", "_");
    StringReplace(dynamicCsvFileName, ":", "_");

    // Create CSV file header
    int fileHandle = FileOpen(dynamicCsvFileName, FILE_WRITE | FILE_CSV | FILE_ANSI | FILE_COMMON);
    if(fileHandle != INVALID_HANDLE)
    {
        string header = "Date,Time,Type,MA Score,MACD Score,RSI Score,Stoch Score,BB Score,ATR Score,SAR Score,Ichimoku Score,ADX Score,Volume Score,Total Score";
        FileWrite(fileHandle, header);
        FileClose(fileHandle);
    } 
    else 
    {
        Print("Error creating CSV file: ", GetLastError());
        return INIT_FAILED;
    }

    return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    // Store the old filename
    string oldFileName = dynamicCsvFileName;
    
    // Regenerate the dynamic CSV filename with the updated end time
    string symbol = _Symbol;
    string mqlFilename = MQLInfoString(MQL_PROGRAM_NAME);
    int randomNumber = MathRand() % 9000 + 1000; // Random number between 1000 and 9999
    
    dynamicCsvFileName = StringFormat("%s_%s_%s_%s_%d.csv", 
        symbol,
        TimeToString(startTime, TIME_DATE),
        TimeToString(endTime, TIME_DATE),
        mqlFilename,
        randomNumber
    );
    
    // Replace spaces and colons with underscores in the filename
    StringReplace(dynamicCsvFileName, " ", "_");
    StringReplace(dynamicCsvFileName, ":", "_");
    
    // Rename the existing file with the updated filename
    if (FileIsExist(oldFileName, FILE_COMMON))
    {
        if (FileMove(oldFileName, FILE_COMMON, dynamicCsvFileName, FILE_COMMON))
        {
            Print("File successfully renamed to: ", dynamicCsvFileName);
        }
        else
        {
            Print("Error renaming file: ", GetLastError());
        }
    }
    else
    {
        Print("Original file not found: ", oldFileName);
    }

    // Release indicators using the included IndicatorScores.mqh file
    ReleaseIndicators();
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
        symbolMinLot = inputMinLot;
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
//| Expert Tick Function (Main Logic)                                |
//+------------------------------------------------------------------+
void OnTick()
{
    endTime = TimeCurrent(); // Update the end time

    // Check if trading is allowed
    if(!TerminalInfoInteger(TERMINAL_TRADE_ALLOWED)) return;

    // Calculate the total score and sum of active weights
    double sumWeights = 0.0;
    double totalScore = CalculateTotalScore(sumWeights); // From IndicatorScores.mqh

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
    int fileHandle = FileOpen(dynamicCsvFileName, FILE_CSV | FILE_READ | FILE_WRITE | FILE_COMMON| FILE_ANSI);

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
