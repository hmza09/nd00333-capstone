*NOTE:* This file is a template that you can use to create the README for your project. The *TODO* comments below will highlight the information you should be sure to include.

# Udacity Capstone Project: Automl & HyperDrive Experiment

The current project uses machine learning to predict patients’ survival based on their medical data. 

I create two models in the environment of Azure Machine Learning Studio: one using AutoML and one customized model whose hyperparameters are tuned using HyperDrive, then compare the performance of both models and deploy the best performing model as a service using Azure Container Instances (ACI).

## Project Set Up and Installation

I ussed the provided workspace and environment, so everything was pre-installed by Udacity course.
Following scripts were used in this project:

- `automl.ipynb`: for the AutoML experiment
- `hyperparameter_tuning.ipynb`: for the HyperDrive experiment
- `heart_failure_clinical_records_dataset.csv`: the dataset file  taken from Kaggle
- `train.py`: a basic script for manipulating the data used in the HyperDrive experiment; modified script given in first project
- `scoring_file_v_1_0_0.py`: the script used to deploy the model which is downloaded from within Azure Machine Learning Studio
- `env.yml`: the environment file which is also downloaded from within Azure Machine Learning Studio

![](https://github.com/hmza09/nd00333-capstone/blob/master/starter_file/screenshots/02-scripts.PNG)
## Dataset

### Overview

The dataset used is taken from [Kaggle](https://www.kaggle.com/andrewmvd/heart-failure-clinical-data) and -as we can read in the original [Research article](https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-020-1023-5)- the data comes from 299 patients with heart failure collected at the Faisalabad Institute of Cardiology and at the Allied Hospital in Faisalabad (Punjab, Pakistan), during April–December 2015. The patients consisted of 105 women and 194 men, and their ages range between 40 and 95 years old.

The dataset contains 13 features:

| Feature | Explanation | Measurement |
| :---: | :---: | :---: |
| *age* | Age of patient | Years (40-95) |
| *anaemia* | Decrease of red blood cells or hemoglobin | Boolean (0=No, 1=Yes) |
| *creatinine-phosphokinase* | Level of the CPK enzyme in the blood | mcg/L |
| *diabetes* | Whether the patient has diabetes or not | Boolean (0=No, 1=Yes) |
| *ejection_fraction* | Percentage of blood leaving the heart at each contraction | Percentage |
| *high_blood_pressure* | Whether the patient has hypertension or not | Boolean (0=No, 1=Yes) |
| *platelets* | Platelets in the blood | kiloplatelets/mL	|
| *serum_creatinine* | Level of creatinine in the blood | mg/dL |
| *serum_sodium* | Level of sodium in the blood | mEq/L |
| *sex* | Female (F) or Male (M) | Binary (0=F, 1=M) |
| *smoking* | Whether the patient smokes or not | Boolean (0=No, 1=Yes) |
| *time* | Follow-up period | Days |
| *DEATH_EVENT* | Whether the patient died during the follow-up period | Boolean (0=No, 1=Yes) |

### Task

The task was to classify patients based on their odd of survival, the prediction is based on features included in above table.

### Access

I  uploaded the data on azure ml studio, also it was available on my github repository and provided the link in notebook.
![https://github.com/hmza09/nd00333-capstone/blob/master/starter_file/heart_failure_clinical_records_dataset.csv](https://github.com/hmza09/nd00333-capstone/blob/master/starter_file/heart_failure_clinical_records_dataset.csv)

![](https://github.com/hmza09/nd00333-capstone/blob/master/starter_file/screenshots/01-dataset.PNG)

## Automated ML

Below you can see an overview of the `automl` settings and configuration I used for the AutoML run:

![](https://github.com/hmza09/nd00333-capstone/blob/master/starter_file/screenshots/03-automl_config.PNG)

`"n_cross_validations": 2`

This parameter sets how many cross validations to perform, based on the same number of folds (number of subsets). As one cross-validation could result in overfit, in my code I chose 2 folds for cross-validation; thus the metrics are calculated with the average of the 2 validation metrics.

`"primary_metric": 'accuracy'`

I chose accuracy as the primary metric as it is the default metric used for classification tasks.

`"enable_early_stopping": True`

It defines to enable early termination if the score is not improving in the short term. In this experiment, it could also be omitted because the _experiment_timeout_minutes_ is already defined below.

`"max_concurrent_iterations": 4`

It represents the maximum number of iterations that would be executed in parallel.

`"experiment_timeout_minutes": 20`

This is an exit criterion and is used to define how long, in minutes, the experiment should continue to run. To help avoid experiment time out failures, I used the value of 20 minutes.

`"verbosity": logging.INFO`

The verbosity level for writing to the log file.

`compute_target = compute_target`

The Azure Machine Learning compute target to run the Automated Machine Learning experiment on.

`task = 'classification'`

This defines the experiment type which in this case is classification. Other options are _regression_ and _forecasting_.

`training_data = dataset`

The training data to be used within the experiment. It should contain both training features and a label column - see next parameter.

`label_column_name = 'DEATH_EVENT'` 

The name of the label column i.e. the target column based on which the prediction is done.

`path = project_folder`

The full path to the Azure Machine Learning project folder.

`featurization = 'auto'`

This parameter defines whether featurization step should be done automatically as in this case (_auto_) or not (_off_).

`debug_log = 'automl_errors.log`

The log file to write debug information to.

### Results

- Model Run Widget

![](https://github.com/hmza09/nd00333-capstone/blob/master/starter_file/screenshots/04-automl_runwidget.PNG)

- Metrics

![](https://github.com/hmza09/nd00333-capstone/blob/master/starter_file/screenshots/05-automl_metrics.PNG)

- Best Performance Model

![](https://github.com/hmza09/nd00333-capstone/blob/master/starter_file/screenshots/06-automl_model.PNG)

## Hyperparameter Tuning

**Parameter sampler**

I specified the parameter sampler as such:

```
ps = RandomParameterSampling(
    {
        '--C' : choice(0.001,0.01,0.1,1,10,20,50,100,200,500,1000),
        '--max_iter': choice(50,100,200,300)
    }
)
```

I chose discrete values with _choice_ for both parameters, _C_ and _max_iter_.

_C_ is the Regularization while _max_iter_ is the maximum number of iterations.

_RandomParameterSampling_ is one of the choices available for the sampler and I chose it because it is the faster and supports early termination of low-performance runs. If budget is not an issue, we could use _GridParameterSampling_ to exhaustively search over the search space or _BayesianParameterSampling_ to explore the hyperparameter space.

**Early stopping policy**

An early stopping policy is used to automatically terminate poorly performing runs thus improving computational efficiency. I chose the _BanditPolicy_ which I specified as follows:
```
policy = BanditPolicy(evaluation_interval=2, slack_factor=0.1)
```

- Two hyperparameters tunned in this model

![](https://github.com/hmza09/nd00333-capstone/blob/master/starter_file/screenshots/07-hyperdrive_config.PNG)

- Run Widget

![](https://github.com/hmza09/nd00333-capstone/blob/master/starter_file/screenshots/08-hyperdrive_run.PNG)

### Results

- Model with different Hyperparameter tunning and Metrics

![](https://github.com/hmza09/nd00333-capstone/blob/master/starter_file/screenshots/09-hyperdrive_model.PNG)

- Register Model with RunID

![](https://github.com/hmza09/nd00333-capstone/blob/master/starter_file/screenshots/10-hyperdrive_register.PNG)
## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
