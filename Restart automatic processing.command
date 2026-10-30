#!/bin/zsh
cd -- "$(dirname -- "$0")"
./tennisctl restart
read "?Press Return to close."
