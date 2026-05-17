#!/bin/bash
set -euo pipefail

rm -rf dist
python -m pip install --upgrade --quiet build twine
python -m build
python -m twine upload dist/*
