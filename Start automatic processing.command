#!/bin/zsh
cd -- "$(dirname -- "$0")"
"./.runtime/venv/bin/python" service.py install
read "?Press Return to close."
