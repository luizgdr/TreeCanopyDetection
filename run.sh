#!/bin/sh

if [ ! -d "venv" ]; then
	python -m venv && venv/bin/pip install -r requirements.txt || exit
fi

FILE="main.py"

if [ -f "$1" ]; then
	FILE="$1"
	shift
fi

venv/bin/python "$FILE" "$@"
