import numpy as np
from titanic_model import config
from titanic_model.data_management import load_pipeline

_titanic_pipe = load_pipeline()


def make_prediction(input_data):
    data = input_data
    prediction = _titanic_pipe.predict(data[config.FEATURES])
    output = np.exp(prediction)

    response = {"prediction": output}

    return response
