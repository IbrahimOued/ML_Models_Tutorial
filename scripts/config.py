from pathlib import Path


class Config:
    RANDOM_SEED = 28
    TEST_SIZE = .2
    ASSETS_PATH = Path("./assets")
    ORIGINAL_DATASET_FILE_PATH = ASSETS_PATH/"original_dataset"/"udemy_courses.csv"
    DATASET_PATH = ASSETS_PATH/"data"  # Dossier pour notre dataset
    FEATURES_PATH = ASSETS_PATH/"features"  # Dossier pour les features
    MODELS_PATH = ASSETS_PATH/"models"  # Dossier pour les modèles
    METRICS_FILE_PATH = ASSETS_PATH/"metrics.json"  # Fichier pour les métriques
