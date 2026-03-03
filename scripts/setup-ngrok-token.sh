#!/usr/bin/env bash
# One-time setup: add your ngrok authtoken so tunnels work.
# Get your token: https://dashboard.ngrok.com/get-started/your-authtoken

set -e
NGROK="${NGROK:-$HOME/bin/ngrok}"
if [[ ! -x "$NGROK" ]]; then
  NGROK="ngrok"
fi

if [[ -z "$1" ]]; then
  echo "Usage: $0 YOUR_NGROK_AUTHTOKEN"
  echo "Get your token: https://dashboard.ngrok.com/get-started/your-authtoken"
  exit 1
fi

"$NGROK" config add-authtoken "$1"
echo "Done. You can run: $NGROK http 5001"
