help:  ## Show help
	@grep -E '^[.a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

init: ## Init env
	python -m pip install -U pip
	pip install -r requirements.txt
	pip install -e .

init-dev:  ## Init dev env
	python -m pip install -U pip
	pip install -r requirements.txt
	pip install -e ".[dev]"

format:  ## Run formatting
	black .
	isort . --skip-gitignore --profile black
