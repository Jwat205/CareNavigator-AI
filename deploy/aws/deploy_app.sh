#!/bin/bash
# Ships the repo to the instance launch_instance.py created, builds the
# Docker image there, and starts the container with the same multi-worker
# setup we validated locally. Run from anywhere; paths are resolved relative
# to this script's own location.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
INFO_FILE="$HERE/.instance_info.json"

if [ ! -f "$INFO_FILE" ]; then
    echo "No $INFO_FILE found — run launch_instance.py first." >&2
    exit 1
fi

INFO_FILE_FOR_PY="$(cygpath -w "$INFO_FILE" 2>/dev/null || echo "$INFO_FILE")"
INSTANCE_IP="$(python3 -c "import json; print(json.load(open(r'$INFO_FILE_FOR_PY'))['public_ip'])")"
KEY_PATH="$(python3 -c "import json; print(json.load(open(r'$INFO_FILE_FOR_PY'))['key_path'])")"
SSH="ssh -i $KEY_PATH -o StrictHostKeyChecking=accept-new ec2-user@$INSTANCE_IP"
SCP="scp -i $KEY_PATH -o StrictHostKeyChecking=accept-new"

echo "Packaging repo (excluding .git, caches, prior benchmark results)..."
TARBALL="$(mktemp -u).tar.gz"
tar -czf "$TARBALL" \
    --exclude='.git' \
    --exclude='.claude' \
    --exclude='__pycache__' \
    --exclude='.pytest_cache' \
    --exclude='backend/benchmarks/benchmark_results_*' \
    --exclude='deploy/aws/*.pem' \
    -C "$REPO_ROOT" .

echo "Waiting for SSH to accept connections..."
for i in $(seq 1 30); do
    if $SSH -o ConnectTimeout=5 -o BatchMode=yes 'echo ok' >/dev/null 2>&1; then
        break
    fi
    sleep 5
done

echo "Uploading ($(du -h "$TARBALL" | cut -f1))..."
$SCP "$TARBALL" "ec2-user@$INSTANCE_IP:/tmp/app.tar.gz"
rm -f "$TARBALL"

echo "Building and starting the container on the instance..."
$SSH bash -s <<'REMOTE'
set -euo pipefail
rm -rf ~/app && mkdir ~/app
tar -xzf /tmp/app.tar.gz -C ~/app
cd ~/app
sudo docker build -t carenavigator-ai:loadtest .
sudo docker rm -f carenav-loadtest 2>/dev/null || true
sudo docker run -d --name carenav-loadtest -p 8000:8000 -p 8501:8501 carenavigator-ai:loadtest
echo "Waiting for the app to come up..."
for i in $(seq 1 30); do
    if curl -sf http://localhost:8000/health >/dev/null 2>&1; then
        echo "App is healthy."
        break
    fi
    sleep 5
done
sudo docker logs carenav-loadtest --tail 30
REMOTE

echo ""
echo "Done. Backend: http://$INSTANCE_IP:8000   Streamlit: http://$INSTANCE_IP:8501"
echo "Run the benchmark from YOUR machine (not the instance) against that backend URL, e.g.:"
echo "  cd \"$REPO_ROOT/backend/benchmarks\" && python benchmark.py --base-url http://$INSTANCE_IP:8000 --concurrent 1 10 50 100 500 --requests 500"
echo ""
echo "When you're done: python terminate_instance.py"
