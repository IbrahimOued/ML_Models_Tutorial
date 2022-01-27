import pickle
import json
import pandas as pd
from config import Config
from sklearn.metrics import mean_squared_error, mean_absolute_error
Config.MODELS_PATH.mkdir(parents=True, exist_ok=True)


X_test = pd.read_csv(str(Config.FEATURES_PATH / "test_features.csv"))
y_test = pd.read_csv(str(Config.FEATURES_PATH / "test_labels.csv"))

# Restauration du modèle
model = pickle.load(open(str(Config.MODELS_PATH/"model.pk"), mode='rb'))
r_squared = model.score(X_test, y_test)
# Prediction
y_pred = model.predict(X_test)
rmse = mean_squared_error(y_pred=y_pred, y_true=y_test)
rmae = mean_absolute_error(y_pred=y_pred, y_true=y_test)

with open(str(Config.METRICS_FILE_PATH), mode='w') as f:
    json.dump(dict(r_squared=r_squared, rmse=rmse, rmae=rmae), f)
