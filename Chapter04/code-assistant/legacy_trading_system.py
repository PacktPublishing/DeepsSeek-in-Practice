# legacy_trading_system.py - TradeFin Corp Legacy Trading System (circa 2018)
import time
import sqlite3
import json
import requests
from datetime import datetime
import threading

# Global variables - not thread safe!
current_positions = {}
account_balance = 1000000.0
risk_limit = 0.02
api_key = "hardcoded-api-key-123"
trading_enabled = True

class TradingSystem:
    def __init__(self):
        self.db_connection = sqlite3.connect('trading.db', check_same_thread=False)
        self.setup_database()
        
    def setup_database(self):
        # No proper error handling
        cursor = self.db_connection.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY,
                symbol TEXT,
                quantity INTEGER,
                price REAL,
                timestamp TEXT,
                type TEXT
            )
        ''')
        
    def get_market_data(self, symbol):
        # Hardcoded API endpoint, no retry logic
        try:
            response = requests.get(
                f"https://api.tradingdata.com/v1/quote/{symbol}",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=5
            )
            return json.loads(response.text)
        except:
            # Swallowing all exceptions
            return None
            
    def calculate_position_size(self, symbol, price):
        # Complex risk calculation in single function
        global account_balance, risk_limit, current_positions
        
        current_risk = 0
        for pos in current_positions.values():
            current_risk += abs(pos['quantity'] * pos['price']) / account_balance
            
        available_risk = risk_limit - current_risk
        if available_risk <= 0:
            return 0
            
        # No validation of inputs
        position_value = available_risk * account_balance
        quantity = int(position_value / price)
        
        return quantity
        
    def execute_trade(self, symbol, quantity, trade_type):
        # No transaction management
        global current_positions, account_balance, trading_enabled
        
        if not trading_enabled:
            return False
            
        market_data = self.get_market_data(symbol)
        if not market_data:
            return False
            
        price = market_data['price']
        
        # No proper logging or audit trail
        trade_id = int(time.time())
        
        cursor = self.db_connection.cursor()
        cursor.execute(
            "INSERT INTO trades VALUES (?, ?, ?, ?, ?, ?)",
            (trade_id, symbol, quantity, price, str(datetime.now()), trade_type)
        )
        
        # Update positions without proper validation
        if symbol in current_positions:
            if trade_type == 'buy':
                current_positions[symbol]['quantity'] += quantity
            else:
                current_positions[symbol]['quantity'] -= quantity
        else:
            current_positions[symbol] = {
                'quantity': quantity if trade_type == 'buy' else -quantity,
                'price': price
            }
            
        # Update balance without proper accounting
        if trade_type == 'buy':
            account_balance -= quantity * price
        else:
            account_balance += quantity * price
            
        return True
        
    def momentum_strategy(self):
        # Strategy logic mixed with execution logic
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
        
        while trading_enabled:
            for symbol in symbols:
                market_data = self.get_market_data(symbol)
                if not market_data:
                    continue
                    
                current_price = market_data['price']
                
                # Get historical data for momentum calculation
                # This is a simplified momentum indicator
                historical_response = requests.get(
                    f"https://api.tradingdata.com/v1/historical/{symbol}",
                    headers={"Authorization": f"Bearer {api_key}"}
                )
                
                if historical_response.status_code != 200:
                    continue
                    
                historical_data = json.loads(historical_response.text)
                
                # Calculate 10-period momentum
                if len(historical_data) < 10:
                    continue
                    
                old_price = historical_data[-10]['close']
                momentum = (current_price - old_price) / old_price
                
                # Trading logic
                if momentum > 0.05:  # 5% momentum threshold
                    # Buy signal
                    if symbol not in current_positions or current_positions[symbol]['quantity'] <= 0:
                        quantity = self.calculate_position_size(symbol, current_price)
                        if quantity > 0:
                            self.execute_trade(symbol, quantity, 'buy')
                            print(f"Bought {quantity} shares of {symbol} at ${current_price}")
                            
                elif momentum < -0.05:  # -5% momentum threshold
                    # Sell signal
                    if symbol in current_positions and current_positions[symbol]['quantity'] > 0:
                        quantity = current_positions[symbol]['quantity']
                        self.execute_trade(symbol, quantity, 'sell')
                        print(f"Sold {quantity} shares of {symbol} at ${current_price}")
                        
            time.sleep(60)  # Check every minute
            
    def run_trading_system(self):
        # No proper error handling or graceful shutdown
        try:
            momentum_thread = threading.Thread(target=self.momentum_strategy)
            momentum_thread.start()
        except Exception as e:
            print(f"Error: {e}")
            
    def get_portfolio_summary(self):
        # No proper data validation or error handling
        global current_positions, account_balance
        
        total_value = account_balance
        for symbol, position in current_positions.items():
            market_data = self.get_market_data(symbol)
            if market_data:
                total_value += position['quantity'] * market_data['price']
                
        return {
            'cash': account_balance,
            'positions': current_positions,
            'total_value': total_value
        }

# Main execution
if __name__ == "__main__":
    system = TradingSystem()
    system.run_trading_system()
