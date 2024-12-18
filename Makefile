.PHONY: Makefile
CONTAINER_NAME := grammar-induction

# Docker-related and running stuff inside docker:
dc/% dc_%:  ### Run any action inside docker (replace % with any action from below)
	docker compose run --rm ${CONTAINER_NAME} make $*

# Run all
all: check test ### Run all checks and tests (lints, mypy, tests...)

all_ff: check_ff test ### Run all checks and tests, but fail on first that returns error (lints, mypy, tests...)

# Separate command versions for github actions
ci/test:
	PYTHONPATH=./src pytest tests/ --durations=10 --junit-xml=test-results.xml

ci/check: ci/check/lint/black ci/check/lint/deps ci/check/lint/flake8 ci/check/lint/isort ci/check/mypy

ci/check/lint/black ci/lint/black:
	black src/ --diff --check --quiet

ci/check/lint/deps ci/lint/deps:
	fawltydeps --detailed

ci/check/lint/flake8 ci/lint/flake8:
	flake8 src/

ci/check/lint/isort ci/lint/isort:
	isort src/ --diff --check --quiet

ci/check/mypy ci/lint/mypy:
	PYTHONPATH=./src mypy src/ --show-error-codes --show-traceback --implicit-reexport --junit-xml=mypy-results.xml

check/lint/black lint/black black-check:  ### Run black lint check (code formatting)
	-black src/ --diff --check --color

check_ff/lint/black lint_ff/black:
	black src/ --diff --check --color

check/lint/deps:
	-fawltydeps --detailed

check_ff/lint/deps:
	fawltydeps --detailed

check/lint/flake8 lint/flake8 flake8-check:  ### Run flake8 lint check (pep8 etc.)
	-flake8 src/

check_ff/lint/flake8 lint_ff/flake8:
	flake8 src/

check/lint/isort lint/isort isort-check:  ### Run isort lint check (import sorting)
	-isort src/ --diff --check --color

check_ff/lint/isort lint_ff/isort:
	isort src/ --diff --check --color

check/mypy check_ff/mypy lint/mypy lint_ff/mypy mypy:  ### Run mypy check (type checking)
	PYTHONPATH=./src mypy src/ --show-error-codes --show-traceback --implicit-reexport

check/lint lint: check/lint/black check/lint/deps check/lint/flake8 check/lint/isort  ### Run all lightweight lint checks (no mypy)

check_ff/lint lint_ff: check_ff/lint/black check_ff/lint/deps check_ff/lint/flake8 check_ff/lint/isort  ### Run all lightweight lint checks, but fail on first that returns error

check lint_full full_lint: check/lint check/mypy  ### Run all lint checks and mypy

check_ff lint_full_ff full_lint_ff: check_ff/lint check_ff/mypy  ### Run all lint checks and mypy, but fail on first that returns error

lint_fix lint/fix:  ### Automatically fix lint problems (only reported by black and isort)
	black .
	isort .

test:  ### Run all unittests
	PYTHONPATH=./src pytest tests

bash:  ### Open bash console (useful when prefixed with dc/, as it opens bash inside docker)
	bash

copier_update:  ### Run copier to update project from template
	@TEMPLATES=$$(ls -a .copier | grep -i ".copier-answers" | sed -e 's/\.copier-answers\.//' -e 's/\.yml//'); \
	echo "Available templates:"; \
	echo "$$TEMPLATES" | sed 's/^/* /'; \
	read -p "Enter the template name: " TEMPLATE; \
	if echo "$$TEMPLATES" | grep -q "^$$TEMPLATE$$"; then \
		copier update --answers-file .copier/.copier-answers.$$TEMPLATE.yml --skip-answered; \
	else \
		echo "Error: Template '$$TEMPLATE' not found."; \
		exit 1; \
	fi

run:  ### Run the selected driver script (pass DRIVER=name as argument)
	@DRIVERS=$$(ls src/*_driver.py | xargs -n 1 basename | sed 's/_driver.py//'); \
	if [ -z "$(DRIVER)" ]; then \
		echo "Available drivers:"; \
		echo "$$DRIVERS" | sed 's/^/* /'; \
		read -p "Enter the driver name: " USER_DRIVER; \
	else \
		USER_DRIVER="$(DRIVER)"; \
	fi; \
	if echo "$$DRIVERS" | grep -q "^$$USER_DRIVER$$"; then \
		python src/$$USER_DRIVER"_driver.py"; \
	else \
		echo "Error: Driver '$$USER_DRIVER' not found."; \
		exit 1; \
	fi


### Help
help: ## Show this help
	@sed -Ene 's/^([^ 	]+)( [^ 	]+)*:.*##/\1:\t/p' $(MAKEFILE_LIST) | column -t -s $$'\t'
