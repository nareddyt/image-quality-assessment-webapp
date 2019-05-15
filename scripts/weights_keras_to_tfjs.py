from keras.backend import manual_variable_initialization
import tensorflowjs as tfjs
import argparse
from handlers.model_builder import Nima

manual_variable_initialization(True)


def main(keras_model_weights_in):
    # Build keras model and load weights
    nima = Nima('MobileNet', weights=None)
    nima.build()
    nima.nima_model.load_weights(keras_model_weights_in)
    print(nima.nima_model.summary())

    # Save as full model + weights file
    nima.nima_model.save('./keras_model/full_model_aesthetic.h5')

    # Convert to tfjs model
    tfjs.converters.save_keras_model(nima.nima_model, './web_model/')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', '--keras-model-weights-in', help='TODO', required=True)

    args = parser.parse_args()

    main(**args.__dict__)
