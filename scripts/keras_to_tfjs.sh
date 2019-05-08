#!/usr/bin/env bash

~/.local/bin/tensorflowjs_converter \
    --input_format=keras \
    ../image-quality-assessment/models/MobileNet/weights_mobilenet_aesthetic_0.07.hdf5 \
    ./web_model
