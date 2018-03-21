#!/bin/bash
PYTHONPATH="."  python3 -m pytest localgraphclustering/tests/ "$@"
