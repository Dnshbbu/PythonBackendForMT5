# PythonBackendForMT5

# Get tree structure

- tree .\streamlit\ /F /A | findstr /V "__pycache__" | findstr /V ".pyc"
- tree . /F /A | findstr /V "__pycache__" | findstr /V ".pyc" | findstr /V "logs" | findstr /V "models"  | findstr /V "ML_Individual_Analysis"| findstr /V ".log" | findstr /V ".csv"| findstr /V ".ipynb"
- To remove already tracked git files: git rm -r --cached MT5/StrategyTester/streamlit/models/
- ls -File

# Recursively remove folder

Remove-Item -Path "C:\path\to\folder" -Recurse -Force

# To run streamlit app

streamlit run .\MT5\StrategyTester\streamlit\mt5LogExplorer.py

# StreamlitServer

Main page: *mt5LogExplorer.py*

log_explorer_page       // working
server_control_page      // working
    uses zmQServer.py
            uses database_manager.py
scripts_page                 // working
ml_analysis_page.py     // working
prediction_page (a function in mt5LogExplorer.py) // working
    uses ModelPipeline from model_pipeline.py

# Individual Files

model_predictor.py             // to predict using a trained model
realtime_price_predictor.py     // to predict realtime price
    uses model_predictor.py
xgboost_train_model.py         // to train the model
    uses xgboost_trainer.py

## Helperfunctions

database_manager.py (zmQServer.py)
xgboost_trainer (xg_train_model)
model_pipeline.py (prediction page and usingXgboost)

# Deprecated but was working once

usingXgboost
    uses create_pipeline_from_analyzer from model_pipeline.py

# Removed

realtime_monitoring_page.py     // currently not working so removed
model_manager.py

# To add a new model

modified:   MT5/StrategyTester/streamlit/model_config.json
modified:   MT5/StrategyTester/streamlit/model_implementations.py
modified:   MT5/StrategyTester/streamlit/model_trainer.py
modified:   MT5/StrategyTester/streamlit/train_models.py

# Run mlflow in another terminal

mlflow ui --backend-store-uri "sqlite:///mlflow.db"
streamlit run .\mt5LogExplorer.py
(for git)
(for train)
(for predict)

# Unit Tests

## Run all test cases

python -m pytest tests\ -v --cov=. --cov-report=term-missing --durations=0 -W always

### ### (with ignore warnings)

python -m pytest tests\ -v --cov=. --cov-report=term-missing --durations=0 -W ignore

## Run a specific test file

python -m pytest tests/test_model_trainer.py -v

## Run a specific test case

python -m pytest tests/test_model_trainer.py::TestTimeSeriesModelTrainer::test_initialization -v

## Run tests matching a pattern

python -m pytest -k "model" -v
