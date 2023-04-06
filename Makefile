SHELL := /bin/bash

.PHONY: run
run:
	source venv/bin/activate
	gradio app.py
