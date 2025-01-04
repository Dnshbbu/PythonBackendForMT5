//+------------------------------------------------------------------+
//|                                                  SNR_Trader.mq5 |
//|                        Supply_Resistance                       |
//+------------------------------------------------------------------+
#property strict

// Input Parameters
input int     LookBackBars           = 50;          // Number of bars to look back for swing points
input double  ConfirmationRSI        = 30.0;        // RSI threshold for buy (oversold)
input double  ConfirmationRSI_Sell   = 70.0;        // RSI threshold for sell (overbought)
input int     RSIPeriod              = 14;          // RSI Period
input double  LotSize                = 1;         // Lot size for trades
input double  StopLossPips           = 50;          // Stop-loss in pips
input double  TakeProfitPips         = 100;         // Take-profit in pips
input ENUM_TIMEFRAMES TimeFrame      = PERIOD_H1;   // Timeframe for swing detection

// Global Variables
double SupportLevels[];
double ResistanceLevels[];

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
   // Initialize arrays
   ArraySetAsSeries(SupportLevels, true);
   ArraySetAsSeries(ResistanceLevels, true);
   
   
   // Draw initial support and resistance lines
   DetectSupportResistance();
   
   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   // Remove all support and resistance lines on deinitialization
   ObjectsDeleteAll(0, OBJ_HLINE, -1, -1);
  }


//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
   // Update support and resistance levels periodically
   static datetime lastUpdate = 0;
   
   datetime currentTime[];
   // Copy the latest bar time
   if(CopyTime(_Symbol, TimeFrame, 0, 1, currentTime) != 1)
     {
      Print("Failed to copy time data.");
      return;
     }
   
   if(currentTime[0] != lastUpdate)
     {
      lastUpdate = currentTime[0];
      DetectSupportResistance();
     }
   
   // Check for trade opportunities
   CheckForTrades();
  }

//+------------------------------------------------------------------+
//| Function to detect support and resistance levels                 |
//+------------------------------------------------------------------+
void DetectSupportResistance()
  {
   // Clear previous levels
   ArrayResize(SupportLevels, 0);
   ArrayResize(ResistanceLevels, 0);
   
   // Copy high and low data
   double highArray[];
   double lowArray[];
   int copiedHigh = CopyHigh(_Symbol, TimeFrame, 0, LookBackBars + 10, highArray);
   int copiedLow  = CopyLow(_Symbol, TimeFrame, 0, LookBackBars + 10, lowArray);
   
   if(copiedHigh < LookBackBars || copiedLow < LookBackBars)
     {
      Print("Failed to copy sufficient high or low data.");
      return;
     }
   
   // Identify swing lows and swing highs
   for(int i = 5; i < LookBackBars; i++) // start from 5 to allow swingRange =5
     {
      // Swing High
      if(IsSwingHigh(i, highArray))
        {
         double swingHigh = highArray[i];
         // Avoid duplicate levels
         if(!LevelExists(ResistanceLevels, swingHigh))
           {
            ArrayResize(ResistanceLevels, ArraySize(ResistanceLevels) +1);
            ResistanceLevels[ArraySize(ResistanceLevels)-1] = swingHigh;
            // Draw resistance line
            string res_name = "Resistance_" + IntegerToString(i);
            DrawHLine(res_name, swingHigh, clrRed);
           }
        }
      
      // Swing Low
      if(IsSwingLow(i, lowArray))
        {
         double swingLow = lowArray[i];
         // Avoid duplicate levels
         if(!LevelExists(SupportLevels, swingLow))
           {
            ArrayResize(SupportLevels, ArraySize(SupportLevels) +1);
            SupportLevels[ArraySize(SupportLevels)-1] = swingLow;
            // Draw support line
            string sup_name = "Support_" + IntegerToString(i);
            DrawHLine(sup_name, swingLow, clrGreen);
           }
        }
     }
  }

//+------------------------------------------------------------------+
//| Function to check if a level already exists                      |
//+------------------------------------------------------------------+
bool LevelExists(double &levels[], double level)
  {
   for(int i=0; i<ArraySize(levels); i++)
      if(MathAbs(levels[i] - level) < _Point *10) // Adjust the precision as needed
         return(true);
   return(false);
  }

//+------------------------------------------------------------------+
//| Function to draw horizontal lines for support/resistance         |
//+------------------------------------------------------------------+
void DrawHLine(string name, double price, color clrLine)
  {
   // Check if the line already exists
   if(ObjectFind(0, name) <0)
     {
      // Create the horizontal line
      if(!ObjectCreate(0, name, OBJ_HLINE, 0, 0, price))
        {
         Print("Failed to create object: ", name);
         return;
        }
      ObjectSetInteger(0, name, OBJPROP_COLOR, clrLine);
      ObjectSetInteger(0, name, OBJPROP_STYLE, STYLE_DASH);
      ObjectSetInteger(0, name, OBJPROP_WIDTH, 1);
     }
   else
     {
      // Update the line price if it exists
      ObjectSetDouble(0, name, OBJPROP_PRICE, price);
     }
  }

//+------------------------------------------------------------------+
//| Function to determine if a bar is a swing high                   |
//+------------------------------------------------------------------+
bool IsSwingHigh(int index, double &highArray[])
  {
   // A swing high is higher than the specified number of bars before and after
   int swingRange = 5; // Number of bars to check on each side
   double currentHigh = highArray[index];
   for(int i = index +1; i <= index + swingRange && i < ArraySize(highArray); i++)
      if(currentHigh <= highArray[i])
         return(false);
   for(int i = index - swingRange; i < index; i++)
      if(currentHigh <= highArray[i])
         return(false);
   return(true);
  }

//+------------------------------------------------------------------+
//| Function to determine if a bar is a swing low                    |
//+------------------------------------------------------------------+
bool IsSwingLow(int index, double &lowArray[])
  {
   // A swing low is lower than the specified number of bars before and after
   int swingRange =5; // Number of bars to check on each side
   double currentLow = lowArray[index];
   for(int i = index +1; i <= index + swingRange && i < ArraySize(lowArray); i++)
      if(currentLow >= lowArray[i])
         return(false);
   for(int i = index - swingRange; i < index; i++)
      if(currentLow >= lowArray[i])
         return(false);
   return(true);
  }

//+------------------------------------------------------------------+
//| Function to check for trade opportunities                       |
//+------------------------------------------------------------------+
void CheckForTrades()
  {
   // Get current price
   double currentPrice = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   
   // Check for Buy Opportunities (Support)
   for(int i=0; i<ArraySize(SupportLevels); i++)
     {
      double support = SupportLevels[i];
      // If current price is near support (within a certain pips)
      double distance = MathAbs(currentPrice - support);
      double pipValue = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
      double threshold = StopLossPips * pipValue;
      if(distance <= threshold)
        {
         // Check if there's no open buy position
         if(!PositionExists(POSITION_TYPE_BUY))
           {
            // RSI Confirmation
            double rsi = GetRSI(0);
            if(rsi <= ConfirmationRSI)
              {
               // Execute Buy Order
               ExecuteBuy(support);
              }
           }
        }
     }
   
   // Check for Sell Opportunities (Resistance)
   for(int i=0; i<ArraySize(ResistanceLevels); i++)
     {
      double resistance = ResistanceLevels[i];
      // If current price is near resistance (within a certain pips)
      double distance = MathAbs(currentPrice - resistance);
      double pipValue = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
      double threshold = StopLossPips * pipValue;
      if(distance <= threshold)
        {
         // Check if there's no open sell position
         if(!PositionExists(POSITION_TYPE_SELL))
           {
            // RSI Confirmation
            double rsi = GetRSI(0);
            if(rsi >= ConfirmationRSI_Sell)
              {
               // Execute Sell Order
               ExecuteSell(resistance);
              }
           }
        }
     }
  }

//+------------------------------------------------------------------+
//| Function to get RSI value at a certain shift                     |
//+------------------------------------------------------------------+
double GetRSI(int shift)
  {
   // Initialize RSI handle
   int handle = iRSI(_Symbol, TimeFrame, RSIPeriod, PRICE_CLOSE);
   if(handle == INVALID_HANDLE)
     {
      Print("Failed to get RSI handle!");
      return(50.0); // Neutral value
     }
   
   double rsi[];
   // Copy RSI value at shift
   if(CopyBuffer(handle, 0, shift, 1, rsi) <=0)
     {
      Print("Failed to copy RSI data!");
      IndicatorRelease(handle);
      return(50.0);
     }
   
   // Release the handle
   IndicatorRelease(handle);
   
   return(rsi[0]);
  }

//+------------------------------------------------------------------+
//| Function to check if a position of a certain type exists         |
//+------------------------------------------------------------------+
bool PositionExists(ENUM_POSITION_TYPE positionType)
  {
   for(int i = 0; i < PositionsTotal(); i++)
     {
      ulong ticket = PositionGetTicket(i);
      if(ticket != 0 && PositionSelectByTicket(ticket))
        {
         if(PositionGetInteger(POSITION_TYPE) == positionType)
           return true;
        }
     }
   return false;
  }

//+------------------------------------------------------------------+
//| Function to execute a Buy Order                                  |
//+------------------------------------------------------------------+
void ExecuteBuy(double entryPrice)
  {
   double sl = entryPrice - StopLossPips * SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   double tp = entryPrice + TakeProfitPips * SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   
   // Adjust for symbol's decimal places
   sl = NormalizeDouble(sl, _Digits);
   tp = NormalizeDouble(tp, _Digits);
   
   MqlTradeRequest request;
   MqlTradeResult result;
   ZeroMemory(request);
   ZeroMemory(result);
   
   request.action   = TRADE_ACTION_DEAL;
   request.symbol   = _Symbol;
   request.volume   = (double)LotSize;
   request.type     = ORDER_TYPE_BUY;
   request.price    = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   request.sl       = sl;
   request.tp       = tp;
   request.deviation= 10;
   request.magic    = 123456;
   request.comment  = "Support Buy Order";
   
   if(!OrderSend(request, result))
      Print("Buy Order Send Failed: ", result.comment);
   else
      Print("Buy Order Sent Successfully. Ticket#: ", result.order);
  }

//+------------------------------------------------------------------+
//| Function to execute a Sell Order                                 |
//+------------------------------------------------------------------+
void ExecuteSell(double entryPrice)
  {
   double sl = entryPrice + StopLossPips * SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   double tp = entryPrice - TakeProfitPips * SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   
   // Adjust for symbol's decimal places
   sl = NormalizeDouble(sl, _Digits);
   tp = NormalizeDouble(tp, _Digits);
   
   MqlTradeRequest request;
   MqlTradeResult result;
   ZeroMemory(request);
   ZeroMemory(result);
   
   request.action   = TRADE_ACTION_DEAL;
   request.symbol   = _Symbol;
   request.volume   = LotSize;
   request.type     = ORDER_TYPE_SELL;
   request.price    = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   request.sl       = sl;
   request.tp       = tp;
   request.deviation= 10;
   request.magic    = 123456;
   request.comment  = "Resistance Sell Order";
   
   if(!OrderSend(request, result))
      Print("Sell Order Send Failed: ", result.comment);
   else
      Print("Sell Order Sent Successfully. Ticket#: ", result.order);
  }
//+------------------------------------------------------------------+