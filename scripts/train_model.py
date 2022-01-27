from operator import mod
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from config import Config

Config.MODELS_PATH.mkdir(parents=True, exist_ok=True)

X_train = pd.read_csv(str(Config.FEATURES_PATH/"train_features.csv"))
y_train = pd.read_csv(str(Config.FEATURES_PATH/"train_labels.csv"))

# Entrainement du modèle
model = LinearRegression()
model.fit(X_train, y_train.to_numpy().ravel())
# Enregistrement du modèle
pickle.dump(model, open(str(Config.MODELS_PATH / "model.pk"), mode='wb'))
