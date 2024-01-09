# ML Zoomcamp midterm project

In this midterm project I'm using some data collected by a gaming application.

The user will play soe games and generate some virtual coins. These coins can be converted into real money and be earned by the players.

Teh scope of the project is to create a model that can predict che amount of real money from a collection of features.

## Data

The data is collected in `data/` folder. Inside there is a dedicated `README.md` file that describes the metadata.

In general the data itself is tabulated and fairly clean. Some transformatin is required and a bit of EDA has been done in `notebooks/eda.ipynb`.

The final and cleaned dataset is available in `data/clean_data.csv`.

## Model evaluation

The notebook `notebooks/model_eval.ipynb` contains all the instructions and exploratory method to test several models and assess their quality.

The tested algorithms are:

- LinearRegression
- RidgeRegression
- DecisionTreeRegressor
- RandomForestRegressor
- XGBoost with 'reg:squarederror' objective

When possible, the hyperparameters were tuned to find the best performances. There were assessed via:

- *Root Mean Squared Error score*: how different are the predicted values from the target ones
- *R squared score*: how well does the model fit to the data

## Deployment

Once the best model has been found, it can be deployed both locally (for quick testing) or remotelly on the cloud.

The necessary components are in `scripts/`:
- *train.py*: will get the data, split and create the model to be saved ina pickle format in `model/` folder
- *predict.py*: will be wrapped into a Flask app to be pinged from a testing notebook (`test_webservice.ipynb`)

The file `deploy/Dockerfile` will help create an isolated environement whether we deploy locally or on the cloud.

Dependencies are available from `dependencies.txt` file.