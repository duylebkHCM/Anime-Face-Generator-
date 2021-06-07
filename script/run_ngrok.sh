#!/usr/bin/env bash
wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip
unzip ngrok-stable-linux-amd64.zip

./ngrok http 8501 &

curl -s http://localhost:4040/api/tunnels | python -c \
    'import sys, json; print("Go to the following URL: " +json.load(sys.stdin)["tunnels"][0]["public_url"])'