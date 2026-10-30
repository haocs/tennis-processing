#!/bin/zsh
cd -- "$(dirname -- "$0")"
"./.runtime/venv/bin/python" service.py status
read "?Press Return to close."
