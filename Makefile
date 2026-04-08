.PHONY: install serve simulate test clean

install:
	uv pip install -e .

serve:
	cco serve

simulate:
	cco simulate --task easy

inference:
	python inference.py

test:
	pytest tests/

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache
	find . -name "__pycache__" -exec rm -rf {} +
