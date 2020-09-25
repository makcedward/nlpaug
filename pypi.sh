#!/bin/bash

python setup.py bdist_wheel --bdist-dir ~/temp/bdistwheel
python -m twine upload dist/* --verbose
