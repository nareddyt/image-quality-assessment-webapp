import tensorflowjs as tfjs
import argparse
from handlers.model_builder import Nima


# FIXME: This script doesn't work as expected. Use the bash command in keras_to_tfjs.sh

def main(keras_model_directory, web_model_directory):

    # Build keras model and load weights
    nima = Nima('MobileNet', weights=None)
    nima.build()
    nima.nima_model.load_weights(keras_model_directory)

    # Convert to tfjs model
    tfjs.converters.save_keras_model(nima, web_model_directory)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', '--keras-model-directory', help='TODO', required=True)
    parser.add_argument('-w', '--web-model-directory', help='TODO', required=True)

    args = parser.parse_args()

    main(**args.__dict__)
