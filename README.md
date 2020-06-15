# Train, save & serve Scikit-learn model using MLFlow

Training experiment based on https://docs.databricks.com/_static/notebooks/mlflow/mlflow-quick-start-training.html

# Running the script

1. From root folder run the following command to build the environment
    ```conda env create --file env.yml```

2. Activate the environment using
    ```conda activate diabetes_env```

3. Run `train.py` to create the models. All models are saved with each their own run_id in the mlruns folder. The run_id 'ce63a1c5dcdd4719b9efedb3493a4321' with corresponding model is used in the following step to serve the model. You can run the following command in your terminal to look at all the saved output from all the models: 
    ```mlflow ui```

4. Navigate to the mlruns folder and run the following command in your terminal (port 8001 is used here, which is also the premise for predict.py):
    ```mlflow models serve -m 1/ce63a1c5dcdd4719b9efedb3493a4321/artifacts/model -h 0.0.0.0 -p 8001```

5. Run `predict.py` to make a call to your served model