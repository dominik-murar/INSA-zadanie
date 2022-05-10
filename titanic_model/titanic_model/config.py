import pathlib

import titanic_model


PACKAGE_ROOT = pathlib.Path(titanic_model.__file__).resolve().parent
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"
DATASET_DIR = PACKAGE_ROOT / "datasets"

MODEL_NAME = "titanic_model"

# data
TESTING_DATA_FILE = "test.csv"
TRAINING_DATA_FILE = "train.csv"
TARGET = "Survived"


# variables
FEATURES = ["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch",
            "Ticket", "Fare", "Cabin", "Embarked"]

# numerical variables with NA in train set
NUMERICAL_VARS_WITH_NA = ["Age", "Cabin", "Embarked"]

# Cabin variable for ship part generation
CABIN_TO_SHIP_PART = ["Cabin", "ShipPart"]

# variables to log transform
NUMERICALS_LOG_VARS = ["Pclass", "Age", "SibSp", "Parch", "Fare"]

NUMERICAL_NA_NOT_ALLOWED = [
    feature for feature in FEATURES if feature not in NUMERICAL_VARS_WITH_NA]
