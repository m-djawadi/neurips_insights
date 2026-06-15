.PHONY: install install-all smoke stats topics llm clean

install:
	pip install -e .

install-all:
	pip install -e ".[all]"

# Quick end-to-end check on a tiny slice (20 papers from one year)
smoke:
	neurips-insights --scrape --start 2023 --end 2023 --limit-per-year 20 --stats

stats:
	neurips-insights --stats

topics:
	neurips-insights --topics --n-topics 20

llm:
	neurips-insights --llm

clean:
	rm -rf data __pycache__ */__pycache__ *.egg-info build dist
