// Dynamic threshold
// Fixed TP, SL
// Long only 
// WriteScoresToCSV functions (still present)


//+------------------------------------------------------------------+
//|                                           TradingFunctions.mqh   |
//+------------------------------------------------------------------+
#property strict

#include <GlobalVariables.mqh>
#include <TradingHelperFuncs.mqh>
#include <Trade\Trade.mqh>

#include <Zmq/Zmq.mqh>
#include <zmq_python.mqh>

CTrade trade;

//+------------------------------------------------------------------+
//| Initialization function                                          |
//+------------------------------------------------------------------+
int OnInitTradingFunc()
{
   // Initialize ATR for Stop Loss if enabled
   if(UseATRForStopLoss && useATR)
   {
      atrHandleStopLoss = iATR(_Symbol, _Period, ATR_Period);
      if(atrHandleStopLoss == INVALID_HANDLE)
      {
         Print("Failed to create ATR indicator handle for Stop Loss. Error: ", GetLastError());
         return(INIT_FAILED);
      }
   }

   // Initialize ATR for Trailing Stop if enabled
   if(UseTrailingStop && UseATRForTrailingStop && useATR)
   {
      atrHandleTrailingStop = iATR(_Symbol, _Period, ATRPeriodTrailing);
      if(atrHandleTrailingStop == INVALID_HANDLE)
      {
         Print("Failed to create ATR indicator handle for Trailing Stop. Error: ", GetLastError());
         return(INIT_FAILED);
      }
   }

   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Calculate stop loss price                                        |
//+------------------------------------------------------------------+
double CalculateStopLossPrice(int positionType)
{
   double stopLoss = 0.0;
   double currentPrice = SymbolInfoDouble(_Symbol, SYMBOL_ASK);

   // If using ATR for stop loss
   if (UseATRForStopLoss && useATR)
   {
      double atrArray[];
      if(CopyBuffer(atrHandleStopLoss, 0, 0, 1, atrArray) <= 0 || ArraySize(atrArray) < 1)
      {
         Print("Error copying ATR buffer for stop loss: ", GetLastError());
         return 0.0;
      }

      double atrValue = atrArray[0];
      stopLoss = currentPrice - atrValue; // For a buy position, stop loss below current price
   }
   // Use percentage for stop loss
   else
   {
      double stopLossAmount = StopLossPercentage / 100.0 * currentPrice;
      stopLoss = currentPrice - stopLossAmount;
   }

   // Ensure stop loss is valid
   if(stopLoss <= 0)
   {
      Print("Invalid Stop Loss price calculated.");
      return 0.0;
   }

   return stopLoss;
}

//+------------------------------------------------------------------+
//| Calculate take profit price                                      |
//+------------------------------------------------------------------+
double CalculateTakeProfitPrice(int positionType, double stopLossPrice)
{
   double takeProfit = 0.0;
   double entryPrice = SymbolInfoDouble(_Symbol, SYMBOL_ASK);

   // TP = Entry Price + (Entry Price - SL) * RiskRewardRatio
   takeProfit = entryPrice + (entryPrice - stopLossPrice) * RiskRewardRatio;

   // Ensure Take Profit is valid
   if(takeProfit <= 0)
   {
      Print("Invalid Take Profit price calculated.");
      return 0.0;
   }

   return takeProfit;
}

//+------------------------------------------------------------------+
//| Process Trading Logic                                            |
//+------------------------------------------------------------------+
void ProcessTradingLogic(double totalScore)
{
   endTime = TimeCurrent(); // Update the end time

   // Check if trading is allowed
   if(!TerminalInfoInteger(TERMINAL_TRADE_ALLOWED)) return;

   // Calculate the total score and sum of active weights
   double sumWeights = 0.0;
   //double totalScore = CalculateTotalScore(sumWeights); // From MulIndicatorScores.mqh

   // Calculate the maximum and minimum possible scores
   double maxScore, minScore;
   CalculateScoreRange(maxScore, minScore);
   double range = maxScore - minScore;

   // Initialize dynamic threshold for buying
   double DynamicBuyThreshold = 0.0;

   if(range != 0.0)
   {
      // Set BuyThreshold to the top 20% of the range
      DynamicBuyThreshold = minScore + 0.8 * range;
   }
   else
   {
      // Fallback to fixed threshold if range is zero (unlikely)
      DynamicBuyThreshold = BuyThreshold * sumWeights / originalSumWeights;
   }

   // Calculate the appropriate lot size
   double dynamicLotSize = CalculateLotSize();

   // Get the current open positions
   int totalPositions = PositionsTotal();

   // Debug output
   PrintFormat("Total Score: %.2f | Max Score: %.2f | Min Score: %.2f | Range: %.2f | Dynamic Buy Threshold: %.2f | Open Positions: %d | Dynamic Lot Size: %.5f",
               totalScore, maxScore, minScore, range, DynamicBuyThreshold, totalPositions, dynamicLotSize);

   // Check Buy Condition
   if(totalScore >= DynamicBuyThreshold && totalPositions == 0)
   {
      double stopLossPrice = CalculateStopLossPrice(POSITION_TYPE_BUY);
      double takeProfitPrice = CalculateTakeProfitPrice(POSITION_TYPE_BUY, stopLossPrice);
      PrintFormat("Attempting to open Buy position. Score: %.2f | Lot Size: %.5f | Stop Loss: %.5f | Take Profit: %.5f", totalScore, dynamicLotSize, stopLossPrice, takeProfitPrice);

      if(trade.Buy(dynamicLotSize, _Symbol, 0, stopLossPrice, takeProfitPrice, "Buy"))
      {
         PrintFormat("Buy order placed successfully. Ticket: %llu | Price: %.5f | TP: %.5f", trade.ResultOrder(), trade.ResultPrice(), takeProfitPrice);
         WriteScoresToCSV("Buy", totalScore, "Indicators");
         Print("Wrote Buy scores to CSV. Total Score: ", totalScore);
      }
      else
      {
         PrintFormat("Error placing Buy order. Error code: %d", GetLastError());
      }
   }
   // Check for position closure
   else if(totalPositions > 0)
   {
      ulong ticket = PositionGetTicket(0);
      if(PositionSelectByTicket(ticket))
      {
         ENUM_POSITION_TYPE positionType = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
         if(totalScore < 0 && positionType == POSITION_TYPE_BUY)
         {
            PrintFormat("Attempting to close position. Score: %.2f", totalScore);
            if(trade.PositionClose(ticket))
            {
               Print("Position closed successfully");
               WriteScoresToCSV("Close Buy", totalScore, "Indicator-based Closure");
               Print("Wrote Close Buy scores to CSV. Total Score: ", totalScore);
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
//| Process Trade Events                                             |
//+------------------------------------------------------------------+
void ProcessTradeEvents()
{
   // Iterate through all recent deals to find closures
   int totalDeals = HistoryDealsTotal();
   if(totalDeals == 0) return;

   // Get the last deal ticket
   ulong deal_ticket = HistoryDealGetTicket(totalDeals - 1);
   if(deal_ticket == 0) return;

   // Get deal type
   ENUM_DEAL_TYPE deal_type = (ENUM_DEAL_TYPE)HistoryDealGetInteger(deal_ticket, DEAL_TYPE);
   if(deal_type != DEAL_TYPE_BUY)
      return; // Not a closure we are interested in

   // Get the order ticket associated with the deal
   ulong order_ticket = HistoryDealGetInteger(deal_ticket, DEAL_ORDER);
   if(order_ticket == 0) return;

   // Retrieve order details
   if(!HistoryOrderSelect(order_ticket))
   {
      Print("Failed to select order: ", order_ticket);
      return;
   }

   double takeProfit = HistoryOrderGetDouble(order_ticket, ORDER_TP);
   double stopLoss = HistoryOrderGetDouble(order_ticket, ORDER_SL);
   double entryPrice = HistoryOrderGetDouble(order_ticket, ORDER_PRICE_OPEN);
   double closePrice = HistoryDealGetDouble(deal_ticket, DEAL_PRICE);

   string reason = "Unknown";

   if(takeProfit > 0 && closePrice >= takeProfit)
      reason = "Take-Profit";
   else if(stopLoss > 0 && closePrice <= stopLoss)
      reason = "Stop-Loss";
   else
      reason = "Indicator-based Closure";

   //// Log the closure reason
   //WriteScoresToCSV("Position Closed", 0.0, reason);
   //PrintFormat("Position closed. Reason: %s", reason);
}