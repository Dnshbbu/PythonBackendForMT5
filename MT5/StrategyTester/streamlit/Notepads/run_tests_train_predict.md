1. I want you to go to the location "C:\Users\StdUser\Desktop\MyProjects\Backtesting\MT5\StrategyTester\streamlit\" in terminal and execute the below

2.run "python -m pytest tests\ -v --cov=. --cov-report=term-missing --durations=0 -W always"

    if there are any failures(any other errors ), inform me.

    with my permission, try to fix them.

    if no errors, move to next step.

3. run "python .\train_models.py" and wait for it to complete and investigate the output,

   if there are any errors, inform me.

   with my permission, try to fix them.

   if no errors, move to next step.
4. set the model_name in the main() of run_predictions.py to the xgboost model created in the previous step and then

   run 'python .\run_prediction.py' and wait for it to complete and investigate the output,

   if there are any errors, inform me.

   with my permission, try to fix them.

   if no errors, move to next step.
5. Give me summary of above all steps

   (future addition: save to result file)
