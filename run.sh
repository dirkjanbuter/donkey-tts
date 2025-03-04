#!/bin/bash
source venv/bin/activate
python -c "import numba; numba.clear_cache()"
uvicorn main:app --host 127.0.0.1 --port 8979
