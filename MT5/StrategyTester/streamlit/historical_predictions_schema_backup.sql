CREATE TABLE historical_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    datetime TIMESTAMP,
                    actual_price REAL,
                    predicted_price REAL,
                    error REAL,
                    price_change REAL,
                    predicted_change REAL,
                    price_volatility REAL,
                    run_id TEXT,
                    source_table TEXT,
                    model_name TEXT
                )