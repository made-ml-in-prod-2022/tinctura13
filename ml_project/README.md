ml-project
==============================

ML in prod course at MADE 2022

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make report` and so on
    ├── README.md          <- The top-level README for developers using this project.
    ├── configs            <- YAML configs for the project.
    |
    ├── data
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- Makes project pip installable (pip install -e .) so src can be imported
    |
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    |   ├── configs        <- Configs for the project pipelines.
    │   ├── features       <- Scripts to turn raw data into features for modeling.
    │   │   └── build_features.py
    |   |
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions.
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    |   |
    │   └── utils
    |       ├── scaler.py  <- Custrom tarnsformer (standard scaler).
    |       └── utils.py   <- Utils to evaluate, serialize and deserialize models.
    │
    ├── visualize.py       <- Function to create EDA with Pandas Profiling.
    ├── train_pipleine.py  <- Entry point for model training.
    ├── infer_pipleine.py  <- Entry point for model inference.
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
