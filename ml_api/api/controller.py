from flask import Blueprint, request, jsonify
from titanic_model.predict import make_prediction

from api.config import get_logger
from api import __version__ as api_version

_logger = get_logger(logger_name=__name__)


prediction_app = Blueprint('prediction_app', __name__)


@prediction_app.route('/health', methods=['GET'])
def health():
    if request.method == 'GET':
        _logger.info('health status OK')
        return 'ok'

@prediction_app.route('/version', methods=['GET'])
def version():
    if request.method == 'GET':
        return jsonify({'api_version': api_version})



@prediction_app.route('/v1/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        json_data = request.get_json()
        _logger.info(f'Inputs: {json_data}')

        result = make_prediction(input_data=json_data)
        _logger.info(f'Outputs: {result}')

        predictions = result.get('predictions')

        return jsonify({'predictions': list(predictions)})