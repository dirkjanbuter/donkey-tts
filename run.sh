#!/bin/bash
source venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8979 --timeout-keep-alive 280 --workers 4
