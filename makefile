module = LayersBlocks
directories = LayersBlocks
pre_formatter = isort
formatter = yapf -i --recursive --style='{based_on_style: pep8, indent_width: 4, column_limit: 110}'
partial_linter = flake8 --max-line-length 110 --max-complexity 10 --extend-ignore=C901,E203
full_linter_1 = flake8 --max-line-length 110 --max-complexity 10 --extend-ignore=E203
full_linter_2 = pylint --disable=E0401 --max-line-length=110 --generated-members=cv2
static_analyzer = mypy --ignore-missing-imports --strict
security_analyzer = bandit

partial_pre_commit:
	echo "pre commit..."
	$(pre_formatter) $(directories)
	$(formatter) $(directories)
	$(partial_linter) $(directories)

full_pre_commit:
	echo "pre commit..."
	$(pre_formatter) $(directories)
	$(formatter) $(directories)
	$(full_linter_1) $(directories)
	$(full_linter_2) $(directories)
	$(static_analyzer) $(directories)
	$(security_analizer) $(directories)

all: clean install

install:
	echo "installing..."
	python3 setuptools_script.py bdist_wheel
	python3 -m pip install dist/*.whl
	python3 -m pip install $(module)

clean:
	echo "cleaning..."
	pip3 uninstall -y $(module)
	rm -rf $(module).egg-info/
	rm -rf build/
	rm -rf dist/


