PYTHON=python

.PHONY: install eda train serve test

install:
	$(PYTHON) -m pip install -r requirements.txt

eda:
	$(PYTHON) -m src.main --task eda

train:
	$(PYTHON) -m src.main --task train

serve:
	$(PYTHON) -m src.main --task serve

test:
	pytest -q
