from operator import mod
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from config import Config

Config.MODELS_PATH.mkdir(parents=True, exist_ok=True)

X_train = pd.read_csv(str(Config.FEATURES_PATH/"train_features.csv"))
y_train = pd.read_csv(str(Config.FEATURES_PATH/"train_labels.csv"))

# Entrainement du modèle
model = RandomForestRegressor(
    n_estimators=150, max_depth=5, random_state=Config.RANDOM_SEED)
model.fit(X_train, y_train.to_numpy().ravel())
# Enregistrement du modèle
pickle.dump(model, open(str(Config.MODELS_PATH / "model.pk"), mode='wb'))
