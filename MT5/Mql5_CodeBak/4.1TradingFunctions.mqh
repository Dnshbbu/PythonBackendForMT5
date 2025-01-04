// Dynamic threshold
// Long only
// Trailing SL 
// Removed WriteScoresToCSV functions
// Partial TP, second half finish based on indicators
// UNFINISHED: Made by Claude, using Chatgpt now

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

// Modify these variables for partial profit-taking
input double PartialProfitPercentage = 0.5; // Percentage of profit to take (e.g., 0.5 for 0.5%)
input double BreakEvenBuffer = 10; // Points above entry to set break-even stop loss
bool isPartialProfitTaken = false;

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

   // Reset the partial profit flag
   isPartialProfitTaken = false;

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


void ProcessTradingLogic(double totalScore)
{
   endTime = TimeCurrent(); // Update the end time

   // Check if trading is allowed
   if(!TerminalInfoInteger(TERMINAL_TRADE_ALLOWED)) return;
   
   double sumWeights = 0.0;

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

   // Check Buy Condition
   if(totalScore >= DynamicBuyThreshold && totalPositions == 0)
   {
      double stopLossPrice = CalculateStopLossPrice(POSITION_TYPE_BUY);
      double takeProfitPrice = CalculateTakeProfitPrice(POSITION_TYPE_BUY);
      
      PrintFormat("Attempting to open Buy position. Score: %.2f | Lot Size: %.5f | Stop Loss: %.5f | Take Profit: %.5f", 
                  totalScore, dynamicLotSize, stopLossPrice, takeProfitPrice);

      // Open two positions: one with T/P and one without
      if(trade.Buy(dynamicLotSize / 2, _Symbol, 0, stopLossPrice, takeProfitPrice, "Buy with T/P"))
      {
         Print("First half of Buy position opened successfully with Take Profit");
         
         // Open second position without T/P
         if(trade.Buy(dynamicLotSize / 2, _Symbol, 0, stopLossPrice, 0, "Buy without T/P"))
         {
            Print("Second half of Buy position opened successfully without Take Profit");
         }
         else
         {
            PrintFormat("Error placing second Buy order. Error code: %d", GetLastError());
         }
      }
      else
      {
         PrintFormat("Error placing first Buy order. Error code: %d", GetLastError());
      }
   }
   // Check for position management (closure and trailing stop)
   else if(totalPositions > 0)
   {
      for(int i = totalPositions - 1; i >= 0; i--)
      {
         ulong ticket = PositionGetTicket(i);
         if(PositionSelectByTicket(ticket))
         {
            ENUM_POSITION_TYPE positionType = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
            if(positionType == POSITION_TYPE_BUY)
            {
               // Check for Take Profit
               if(CheckAndCloseTakeProfit(ticket))
               {
                  continue; // Skip to next position if this one was closed
               }

               // Check for full position closure based on indicators
               if(totalScore < 0)
               {
                  PrintFormat("Attempting to close position based on indicators. Score: %.2f", totalScore);
                  if(trade.PositionClose(ticket))
                  {
                     Print("Position closed successfully");
                  }
                  else
                  {
                     PrintFormat("Error closing position. Error code: %d", GetLastError());
                  }
               }
               // Apply trailing stop to positions without T/P
               else if(PositionGetDouble(POSITION_TP) == 0 && UseTrailingStop)
               {
                  ApplyTrailingStop(ticket);
               }
            }
         }
      }
   }
}




bool CheckAndCloseTakeProfit(ulong ticket)
{
   if(PositionSelectByTicket(ticket))
   {
      double takeProfit = PositionGetDouble(POSITION_TP);
      if(takeProfit > 0)  // Only check positions with a set Take Profit
      {
         double currentPrice = SymbolInfoDouble(_Symbol, SYMBOL_BID);
         if(currentPrice >= takeProfit)
         {
            if(trade.PositionClose(ticket))
            {
               PrintFormat("Position %d closed at Take Profit. Current price: %.5f, Take Profit: %.5f", 
                           ticket, currentPrice, takeProfit);
               return true; // Position closed successfully
            }
            else
            {
               PrintFormat("Error closing position %d at Take Profit. Error code: %d", 
                           ticket, GetLastError());
            }
         }
      }
   }
   return false; // Position not closed
}

//+------------------------------------------------------------------+
//| Take Partial Profit                                              |
//+------------------------------------------------------------------+
void TakePartialProfit(ulong ticket)
{
   if(PositionSelectByTicket(ticket))
   {
      double positionVolume = PositionGetDouble(POSITION_VOLUME);
      double openPrice = PositionGetDouble(POSITION_PRICE_OPEN);
      double currentPrice = SymbolInfoDouble(_Symbol, SYMBOL_BID);
      
      // Calculate profit in percentage
      double profitPercentage = (currentPrice - openPrice) / openPrice * 100;
      
      if(profitPercentage >= PartialProfitPercentage)
      {
         // Calculate the volume to close (half of the position)
         double volumeToClose = NormalizeDouble(positionVolume / 2, 2);
         
         // Close half of the position
         if(trade.PositionClosePartial(ticket, volumeToClose))
         {
            Print("Partial profit taken successfully");
            isPartialProfitTaken = true;
            
            // Modify the stop loss of the remaining position to break-even plus buffer
            double newStopLoss = NormalizeDouble(openPrice + BreakEvenBuffer * _Point, _Digits);
            if(trade.PositionModify(ticket, newStopLoss, 0)) // Set SL to break-even + buffer, TP remains 0
            {
               Print("Stop loss moved to break-even plus buffer for remaining position");
            }
            else
            {
               Print("Error moving stop loss to break-even plus buffer. Error code: ", GetLastError());
            }
         }
         else
         {
            PrintFormat("Error taking partial profit. Error code: %d", GetLastError());
         }
      }
   }
}

double CalculateTakeProfitPrice(ENUM_POSITION_TYPE positionType)
{
   double takeProfitPrice = 0.0;
   double currentPrice = SymbolInfoDouble(_Symbol, SYMBOL_ASK);

   if(positionType == POSITION_TYPE_BUY)
   {
      takeProfitPrice = currentPrice * (1 + PartialProfitPercentage / 100.0);
   }
   else if(positionType == POSITION_TYPE_SELL)
   {
      takeProfitPrice = currentPrice * (1 - PartialProfitPercentage / 100.0);
   }

   return NormalizeDouble(takeProfitPrice, _Digits);
}

//+------------------------------------------------------------------+
//| Trailing Stop function                                           |
//+------------------------------------------------------------------+
void ApplyTrailingStop(ulong ticket)
{
   if(PositionSelectByTicket(ticket))
   {
      double currentSL = PositionGetDouble(POSITION_SL);
      double currentPrice = SymbolInfoDouble(_Symbol, SYMBOL_BID);
      double openPrice = PositionGetDouble(POSITION_PRICE_OPEN);
      
      // Calculate new stop loss based on percentage
      double newSL = NormalizeDouble(currentPrice * (1 - TrailingStopPercent / 100), _Digits);
      
      // Calculate the minimum price movement required to adjust the trailing stop
      double minPriceMove = openPrice * TrailingStopStepPercent / 100;
      
      // Only update if the new stop loss is higher than the current one
      // and the price has moved at least the minimum required amount
      if(newSL > currentSL && currentPrice - openPrice >= minPriceMove)
      {
         if(trade.PositionModify(ticket, newSL, 0)) // Keep TP at 0
         {
            Print("Trailing stop updated for ticket ", ticket, ". New stop loss: ", newSL);
         }
         else
         {
            Print("Error updating trailing stop for ticket ", ticket, ". Error code: ", GetLastError());
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

   double stopLoss = HistoryOrderGetDouble(order_ticket, ORDER_SL);
   double entryPrice = HistoryOrderGetDouble(order_ticket, ORDER_PRICE_OPEN);
   double closePrice = HistoryDealGetDouble(deal_ticket, DEAL_PRICE);

   string reason = "Unknown";

   if(stopLoss > 0 && closePrice <= stopLoss)
      reason = "Stop-Loss";
   else if(isPartialProfitTaken && MathAbs(closePrice - entryPrice) < _Point)
      reason = "Break-Even (After Partial Profit)";
   else
      reason = "Indicator-based Closure";

   PrintFormat("Position closed. Reason: %s, Entry Price: %.5f, Close Price: %.5f", reason, entryPrice, closePrice);
   
   // Reset the partial profit flag after full closure
   isPartialProfitTaken = false;
}