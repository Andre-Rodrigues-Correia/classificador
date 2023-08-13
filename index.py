import io
import logging


from matplotlib.backends.backend_template import FigureCanvas
from flask_cors import CORS

from train.train import Train
from train.prepare_training import prepare_models
from classifier.classifier import Classifier
from flask import Flask, make_response, request
import os.path


app = Flask(__name__)


@app.route('/classifier', methods=["POST"])
def get_classification():
    logging.info('request body: {}'.format(request.get_json()))
    content = request.get_json()
    text = content['text']
    predicted_lang = Classifier.classifier_text(text)
    response = {
        'text': text,
        'language': predicted_lang
    }
    return make_response(response, 200)


@app.route('/train', methods=["POST"])
def train_model():
    if os.path.isfile('datasets/model.pickle'):
        return make_response({
            'message': 'modelo já treinado!'
        }, 400)
    else:
        prepare_models()
        Train.train_model()

    return make_response({
        'message': 'modelo treinado com sucesso!'
    }, 200)


@app.route('/confusion_matriz', methods=["GET"])
def plot_png():
    fig = Train.create_confusion_matriz()
    output = io.BytesIO()
    FigureCanvas(fig).print_figure(output)
    return make_response(output.getvalue())


@app.route('/accuracy', methods=["GET"])
def get_accuracy():
    accuracy_score = Train.get_accuracy()
    return make_response({
        'message': 'accuracy measured by a validation dataset',
        'accuracy': accuracy_score
    })


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    CORS(app)
    app.run()
