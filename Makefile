SHELL := /bin/bash

install:
	wget https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip
	unzip scifact
	python3 -m venv venv
	source venv/bin/activate
	pip install -r requirements.txt

clean:
	-deactivate
	-rm -rf venv
	-rm -rf scifact
	-rm scifact.zip
