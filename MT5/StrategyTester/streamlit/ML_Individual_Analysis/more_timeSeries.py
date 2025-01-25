import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

class EnhancedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size//2, batch_first=True)
        self.dropout2 = nn.Dropout(0.2)
        self.linear1 = nn.Linear(hidden_size//2, 32)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(32, 1)
    
    def forward(self, x):
        lstm1_out, _ = self.lstm1(x)
        lstm1_out = self.dropout1(lstm1_out)
        lstm2_out, _ = self.lstm2(lstm1_out)
        lstm2_out = self.dropout2(lstm2_out[:, -1, :])
        linear1_out = self.relu(self.linear1(lstm2_out))
        predictions = self.linear2(linear1_out)
        return predictions

class TimeSeriesAnalysis:
    def __init__(self, device='cpu'):
        self.device = device
        self.lookback = 10  # Increased lookback
        self.feature_scaler = RobustScaler()
        self.target_scaler = RobustScaler()
        self.model = None
        
    def prepare_data(self, df):
        # Feature engineering
        df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
        df = df.set_index('datetime')
        
        feature_cols = ['EntryScore_SR', 'EntryScore_Pullback', 
                       'EntryScore_EMA', 'EntryScore_AVWAP']
        
        # Create additional features
        df_ml = df[['Price'] + feature_cols].copy()
        df_ml['Price_pct_change'] = df_ml['Price'].pct_change()
        df_ml['Price_MA5'] = df_ml['Price'].rolling(5).mean()
        df_ml['Price_MA10'] = df_ml['Price'].rolling(10).mean()
        df_ml['Price_std5'] = df_ml['Price'].rolling(5).std()
        
        # Handle inf and nan values
        df_ml = df_ml.replace([np.inf, -np.inf], np.nan)
        df_ml = df_ml.ffill().bfill()
        
        # Scale features and target separately
        price_data = df_ml[['Price']].values
        feature_data = df_ml.drop('Price', axis=1).values
        
        self.target_scaler = RobustScaler().fit(price_data)
        self.feature_scaler = RobustScaler().fit(feature_data)
        
        df_ml['Price'] = self.target_scaler.transform(price_data)
        df_ml[df_ml.columns[1:]] = self.feature_scaler.transform(feature_data)
        
        print("\nData Quality Check:")
        print(f"Missing values:\n{df_ml.isnull().sum()}")
        print(f"\nFeature statistics:\n{df_ml.describe()}")
        
        return df_ml
    
    def create_sequences(self, data):
        X, y = [], []
        for i in range(len(data) - self.lookback):
            X.append(data.iloc[i:(i + self.lookback)].values)
            y.append(data.iloc[i + self.lookback]['Price'])
        return torch.FloatTensor(np.array(X)), torch.FloatTensor(y)
    
    def train_model(self, train_data, epochs=200, batch_size=32):
        X, y = self.create_sequences(train_data)
        X, y = X.to(self.device), y.to(self.device)
        
        # Initialize model
        self.model = EnhancedLSTM(X.shape[2]).to(self.device)
        
        # Training parameters
        criterion = nn.HuberLoss()  # More robust to outliers
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)
        
        # Training loop with early stopping
        best_loss = float('inf')
        patience = 15
        patience_counter = 0
        train_losses = []
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            batches = 0
            
            # Mini-batch training
            for i in range(0, len(X), batch_size):
                batch_X = X[i:i+batch_size]
                batch_y = y[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                batches += 1
            
            avg_loss = total_loss / batches
            train_losses.append(avg_loss)
            scheduler.step(avg_loss)
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Training stopped early at epoch {epoch}")
                break
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Loss: {avg_loss:.6f}")
        
        return train_losses
    

    def predict(self, test_data):
        try:
            self.model.eval()
            X, _ = self.create_sequences(test_data)
            X = X.to(self.device)
            
            with torch.no_grad():
                predictions = self.model(X).cpu().numpy()
            
            # Inverse transform predictions
            predictions_rescaled = self.target_scaler.inverse_transform(predictions)
            
            # Pad predictions to match test_data length
            full_predictions = np.zeros(len(test_data))
            full_predictions[:] = np.nan
            full_predictions[self.lookback:self.lookback + len(predictions)] = predictions_rescaled.squeeze()
            
            # Forward fill remaining values
            full_predictions = pd.Series(full_predictions).fillna(method='ffill').fillna(method='bfill').values
            
            return full_predictions
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return np.zeros(len(test_data))

    def evaluate_model(self, y_true, y_pred):
        metrics = {
            'r2': r2_score(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        }
        return metrics
    
    def plot_results(self, actual, predictions, train_losses=None):
        plt.figure(figsize=(15, 10))
        
        # Plot predictions vs actual
        plt.subplot(2, 1, 1)
        plt.plot(actual, label='Actual', color='black', alpha=0.7)
        plt.plot(predictions, label='Predictions', color='blue', alpha=0.7)
        plt.title('Price Predictions vs Actual')
        plt.xlabel('Time Steps')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        
        # Plot training losses
        if train_losses:
            plt.subplot(2, 1, 2)
            plt.plot(train_losses, label='Training Loss')
            plt.title('Training Loss Over Time')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.grid(True)
        
        plt.tight_layout()
        plt.show()




def run_analysis(csv_path, train_size=0.8):
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Load and prepare data
        df = pd.read_csv(csv_path)
        print(f"Loaded data shape: {df.shape}")
        
        analysis = TimeSeriesAnalysis(device)
        df_ml = analysis.prepare_data(df)
        
        # Split data
        train_idx = int(len(df_ml) * train_size)
        train_data = df_ml.iloc[:train_idx]
        test_data = df_ml.iloc[train_idx:]
        
        print(f"\nTraining data shape: {train_data.shape}")
        print(f"Test data shape: {test_data.shape}")
        
        # Train model
        print("\nTraining model...")
        train_losses = analysis.train_model(train_data)
        
        # Make predictions
        predictions = analysis.predict(test_data)
        actual = test_data['Price'].values
        
        # Inverse transform actual values
        actual = analysis.target_scaler.inverse_transform(actual.reshape(-1, 1)).squeeze()
        
        # Ensure same length
        min_len = min(len(actual), len(predictions))
        actual = actual[:min_len]
        predictions = predictions[:min_len]
        
        # Calculate metrics
        metrics = analysis.evaluate_model(actual, predictions)
        
        print("\nModel Performance:")
        print("-" * 50)
        print(f"R² Score: {metrics['r2']:.3f}")
        print(f"MSE: {metrics['mse']:.6f}")
        print(f"MAE: {metrics['mae']:.6f}")
        print(f"RMSE: {metrics['rmse']:.6f}")
        print(f"MAPE: {metrics['mape']:.2f}%")
        
        # Plot results
        analysis.plot_results(actual, predictions, train_losses)
        
        # Return more detailed results
        results = {
            'metrics': metrics,
            'predictions': predictions,
            'actual': actual,
            'train_losses': train_losses
        }
        
        return analysis, results
        
    except Exception as e:
        print(f"Error in run_analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

# Add detailed analysis function
def analyze_predictions(results):
    """Analyze prediction results in detail"""
    if results is None:
        return
    
    predictions = results['predictions']
    actual = results['actual']
    
    # Calculate direction accuracy
    actual_dir = np.sign(np.diff(actual))
    pred_dir = np.sign(np.diff(predictions))
    direction_accuracy = np.mean(actual_dir == pred_dir) * 100
    
    # Calculate prediction statistics
    error = actual - predictions
    
    print("\nDetailed Analysis:")
    print("-" * 50)
    print(f"Direction Accuracy: {direction_accuracy:.2f}%")
    print(f"\nError Statistics:")
    print(f"Mean Error: {np.mean(error):.6f}")
    print(f"Error Std: {np.std(error):.6f}")
    print(f"Max Error: {np.max(np.abs(error)):.6f}")
    
    # Print success rate within different thresholds
    thresholds = [0.001, 0.005, 0.01, 0.02]
    print("\nPrediction Success Rate:")
    for threshold in thresholds:
        success_rate = np.mean(np.abs(error) < threshold) * 100
        print(f"Within {threshold:.3f}: {success_rate:.2f}%")


if __name__ == "__main__":
    csv_path = r"C:\Users\StdUser\Desktop\MyProjects\Backtesting\MT5\StrategyTester\streamlit\processed_data\SYM_10028030_all_details_processed_20250124_234257.csv"
    analysis, results = run_analysis(csv_path)
    
    if results is not None:
        analyze_predictions(results)