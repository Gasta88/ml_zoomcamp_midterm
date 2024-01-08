import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

N_ESTIMATORS = 110
MAX_DEPTH = 7
RANDOM_STATE = 1
OUTPUT_FILE = "../model/rf.bin"

print("Loading data...")
df = pd.read_csv("../data/clean_data.csv")

print("Transforming data...")
df.columns = df.columns.str.lower().str.replace(" ", "_")
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

df_full_train.reset_index(drop=True)
df_test.reset_index(drop=True)

y_full_train = df_full_train["amount"].values
y_test = df_test["amount"].values

del df_full_train["amount"]
del df_test["amount"]

cat = ["provider", "countrycode", "market"]
num = ["usercurrencyamount", "coins"]

print("Training model...")
dv = DictVectorizer(sparse=False)

X_full_train = dv.fit_transform(df_full_train[cat + num].to_dict(orient="records"))
X_test = dv.transform(df_test[cat + num].to_dict(orient="records"))

model = RandomForestRegressor(
    n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, random_state=RANDOM_STATE
)
model.fit(X_full_train, y_full_train)

y_pred = model.predict_proba(X_test)[:, 1]

print("RMSE: ", mean_squared_error(y_test, y_pred))
print("R2: ", r2_score(y_test, y_pred))
print("Saving model...")
with open(OUTPUT_FILE, "wb") as f_out:
    pickle.dump((dv, model), f_out)
