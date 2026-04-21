The simplest way to run this is to run:

```docker build -t chess_engine .
docker run --gpus all
```

Alternatively, use a local pip/conda environment to install the requirements, build with:

```python setup.py install --user
pip install --no-build-isolation -e .
```

and then run `python chess_training_loop.py <algorithm>`.