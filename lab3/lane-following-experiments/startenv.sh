#!/usr/bin/env bash
if ! [[ -d bin ]]; then
  python3 -mvenv .
fi

if ! [[ -f apriltag.mov ]]; then
  echo "WARNING: no apriltags video found!"
fi

. bin/activate
pip install -r requirements.txt
