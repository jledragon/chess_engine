#!/bin/sh
python setup.py install --user
pip install --no-build-isolation -e .
python chess_training_loop.py A2C