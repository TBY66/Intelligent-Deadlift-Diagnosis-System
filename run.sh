#!/usr/bin/env bash
# Launch the Intelligent Deadlift Diagnosis System
cd "$(dirname "$0")"
/opt/anaconda3/bin/python app.py "$@"
