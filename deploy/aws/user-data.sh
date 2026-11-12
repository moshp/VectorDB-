#!/bin/bash
# EC2 user data for Ubuntu 24.04: installs Docker, builds the Document Q&A app
# from GitHub and serves it on port 80. Progress: /var/log/cloud-init-output.log
set -euxo pipefail

apt-get update -y
apt-get install -y docker.io git
systemctl enable --now docker

git clone --depth 1 https://github.com/moshp/VectorDB- /opt/docqa
cd /opt/docqa
docker build -f deploy/Dockerfile -t docqa .
docker run -d --name docqa --restart unless-stopped -p 80:7860 docqa
