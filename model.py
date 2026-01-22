import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


class MarketPredictor:
    def __init__(self, symbol, period="2y"):
        # Initialize the market predictor
        #
        # Parameters:
        # symbol (str): Stock symbol (e.g., "AAPL")
        # period (str): Time period for data ("1y", "2y", "5y", etc.)
        self.symbol = symbol
        self.period = period
        self.model = None
        self.scaler = StandardScaler()
        self.data = None
        self.feature_cols = []
        
    def fetch_data(self):
        # Fetch stock data from Yahoo Finance
        try:
            stock = yf.Ticker(self.symbol)
            self.data = stock.history(period=self.period)
            print(f"Successfully fetched {len(self.data)} days of data for {self.symbol}")
            return self.data
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None
    
    def create_features(self):
        # Create technical indicators and features
        if self.data is None:
            raise ValueError("No data available. Call fetch_data() first.")
        
        df = self.data.copy()
        
        # Basic price features
        df['MA_5'] = df['Close'].rolling(window=5).mean()
        df['MA_10'] = df['Close'].rolling(window=10).mean()
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        
        # Price changes
        df['Price_Change'] = df['Close'].pct_change()
        df['Volume_Change'] = df['Volume'].pct_change()
        
        # Volatility
        df['Volatility'] = df['Close'].rolling(window=10).std()
        
        # RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        
        # Lag features
        for i in range(1, 6):
            df[f'Close_lag_{i}'] = df['Close'].shift(i)
            df[f'Volume_lag_{i}'] = df['Volume'].shift(i)
            
        df['Target'] = df['Close'].shift(-1)
        
        self.data = df
        self.feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume',
                            'MA_5', 'MA_10', 'MA_20', 'Price_Change',
                            'Volume_Change', 'Volatility', 'RSI',
                            'BB_Middle', 'BB_Upper', 'BB_Lower',
                            'Close_lag_1', 'Close_lag_2', 'Close_lag_3',
                            'Close_lag_4', 'Close_lag_5']
        
        print("Features created successfully")
        return self.data
    
    def prepare_data(self):
        # Prepare data for training
        if self.data is None:
            raise ValueError("No data available. Call create_features() first.")
        clean_data = self.data[self.feature_cols + ['Target']].dropna()
        
        # Separate features and target
        X = clean_data[self.feature_cols]
        y = clean_data['Target']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self):
        # Train the linear regression model
        try:
            # Prepare data
            X_train, X_test, y_train, y_test = self.prepare_data()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            self.model = LinearRegression()
            self.model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred_train = self.model.predict(X_train_scaled)
            y_pred_test = self.model.predict(X_test_scaled)
            
            # Evaluate model
            train_mse = mean_squared_error(y_train, y_pred_train)
            test_mse = mean_squared_error(y_test, y_pred_test)
            train_mae = mean_absolute_error(y_train, y_pred_train)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            
            print("="*50)
            print("MODEL EVALUATION RESULTS")
            print("="*50)
            print(f"Training MSE: {train_mse:.4f}")
            print(f"Testing MSE:  {test_mse:.4f}")
            print(f"Training MAE: {train_mae:.4f}")
            print(f"Testing MAE:  {test_mae:.4f}")
            print(f"Training R²:  {train_r2:.4f}")
            print(f"Testing R²:   {test_r2:.4f}")
            print("="*50)
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'Feature': self.feature_cols,
                'Coefficient': self.model.coef_
            }).sort_values('Coefficient', key=abs, ascending=False)
            
            print("\nTop 10 Most Important Features:")
            print(feature_importance.head(10))
            
            return X_train, X_test, y_train, y_test, y_pred_train, y_pred_test
            
        except Exception as e:
            print(f"Error during model training: {e}")
            return None
    
    def predict_next_price(self):
        # Predict the next closing price
        if self.model is None:
            raise ValueError("Model not trained.")
        
        if self.data is None:
            raise ValueError("No data available.")
        
        # Get latest features
        clean_data = self.data[self.feature_cols].dropna()
        latest_features = clean_data.iloc[-1:].values
        latest_features_scaled = self.scaler.transform(latest_features)
        
        # Make prediction
        predicted_price = self.model.predict(latest_features_scaled)[0]
        
        return predicted_price
    
    def get_recommendation(self):
        # Get buy/sell recommendation based on prediction
        if self.data is None:
            raise ValueError("No data available.")
        
        current_price = self.data['Close'].iloc[-1]
        predicted_price = self.predict_next_price()
        
        print("\n" + "="*50)
        print("PREDICTION RESULTS")
        print("="*50)
        print(f"Current Price: ${current_price:.2f}")
        print(f"Predicted Next Price: ${predicted_price:.2f}")
        
        # Calculate percentage change
        change_percent = ((predicted_price - current_price) / current_price) * 100
        
        if change_percent > 2:
            recommendation = "BUY"
        elif change_percent < -2:
            recommendation = "SELL"
        else:
            recommendation = "HOLD"
            
        print(f"Expected Change: {change_percent:.2f}%")
        print(f"Recommendation: {recommendation}")
        
        return {
            'current_price': current_price,
            'predicted_price': predicted_price,
            'change_percent': change_percent,
            'recommendation': recommendation
        }
    
    def visualize_results(self, X_test, y_test, y_pred_test):
        # Create visualization of predictions anf actual values
        plt.figure(figsize=(12, 6))
        
        # Plot 1: Actual vs Predicted
        plt.subplot(1, 2, 1)
        plt.scatter(y_test, y_pred_test, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Prices')
        plt.ylabel('Predicted Prices')
        plt.title('Actual vs Predicted Prices')
        
        # Plot 2: Time series of predictions
        plt.subplot(1, 2, 2)
        test_dates = self.data.index[-len(y_test):]
        plt.plot(test_dates, y_test.values, label='Actual', marker='o', markersize=3)
        plt.plot(test_dates, y_pred_test, label='Predicted', marker='s', markersize=3)
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title('Price Prediction Over Time')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('market_prediction_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Results visualization saved as 'market_prediction_results.png'")

def main():
    print("Market Regression Prediction AI Model")
    print("="*60)
    
    symbol = input("Enter stock symbol (e.g., AAPL, GOOGL): ").upper()
    
    predictor = MarketPredictor(symbol, period="10y")
    
    print("\nFetching data...")
    data = predictor.fetch_data()
    
    if data is None:
        print("Failed to fetch data. Exiting.")
        return
    
    print("\nCreating features...")
    predictor.create_features()
    
    print("\nTraining model...")
    results = predictor.train_model()
    
    if results is None:
        print("Model training failed. Exiting.")
        return
    
    print("\nMaking prediction...")
    recommendation = predictor.get_recommendation()
    
    X_train, X_test, y_train, y_test, y_pred_train, y_pred_test = results
    predictor.visualize_results(X_test, y_test, y_pred_test)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
