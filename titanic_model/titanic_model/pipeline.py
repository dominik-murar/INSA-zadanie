from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectFromModel

from titanic_model import preprocessors as pp
from titanic_model import config


price_pipe = Pipeline(
    [
        ("keep_columns",
         pp.KeepColumnsTransformer(variables=config.FEATURES)
         ),
        (
            "numerical_inputer",
            pp.NumericalImputer(variables=config.NUMERICAL_VARS_WITH_NA),
        ),
        ("ship_part_generator", pp.ShipPartGenerator(
            variables=config.CABIN, target_variables=config.SHIP_PART)),
        ("log_transformer", pp.LogTransformer(
            variables=config.NUMERICALS_LOG_VARS)),
        ("scaler", MinMaxScaler()),
        ("select_important_features", SelectFromModel(
            Lasso(alpha=0.005, random_state=0))),
        ("logistic_regression", LogisticRegression(random_state=0))
    ]
)
