#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = tinctura13
PYTHON_INTERPRETER = python3

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies
requirements:
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

## Set up python interpreter environment
create_environment:
	$(PYTHON_INTERPRETER) -m pip install -q virtualenv virtualenvwrapper
	$(PYTHON_INTERPRETER) -m venv .venv

report:
	$(PYTHON_INTERPRETER) visualize.py

train-lr:
	$(PYTHON_INTERPRETER) train_pipeline.py model=lr

train-rf:
	$(PYTHON_INTERPRETER) train_pipeline.py model=rf

inference:
	$(PYTHON_INTERPRETER) infer_pipeline.py 
