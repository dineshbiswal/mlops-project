from train_model import train_model
import optuna
import mlflow
import shutil
import os

def objective(trial):   

    n_estimators = trial.suggest_int('n_estimators', 10, 100)
    max_depth = trial.suggest_int('max_depth', 2, 25)
    accuracy= train_model(n_estimators=n_estimators, max_depth=max_depth)       
    return accuracy



study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)


mlflow.set_experiment("Model_HyperTuning_125")
print('Best trial:')
trial = study.best_trial
print('  Value: {}'.format(trial.value))
mlflow.log_metric("accuracy", trial.value) 
best_model = "iris_model"
print('  Params: ')
for key, value in trial.params.items():
    mlflow.log_param(key, value)
    best_model=best_model+"_"+ str(value)
    print('    {}: {}'.format(key, value))


print(best_model)
mlflow.log_artifact("models/"+best_model+".pkl")

# Copy the best performing model for further processingn 
destination = "path/to/destination/file.txt"

# Copy the file
shutil.copy("models/"+best_model+".pkl", "./BestModels/"+best_model+".pkl")







