#!/bin/bash

text="$1"

if [ -z "$text" ]; then
    echo "Usage: $0 \"Your text here\""
    exit 1
fi

curl --connect-timeout 30 --max-time 0 -X POST -F "text=$text" -F "language=en" -F "speaker_id=yvonta" http://127.0.0.1:8979/tts_stream/ | mpg123 -q -

