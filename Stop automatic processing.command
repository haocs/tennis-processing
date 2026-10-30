#!/bin/zsh
cd -- "$(dirname -- "$0")"
"./.runtime/venv/bin/python" service.py stop
read "?Press Return to close."
