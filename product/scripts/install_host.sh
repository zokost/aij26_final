#!/usr/bin/env bash
set -euo pipefail
test "$(id -u)" -eq 0 || { echo 'Run with sudo bash install_host.sh'; exit 1; }
. /etc/os-release
test "$ID" = ubuntu || { echo 'This installer supports Ubuntu; see README for other Linux distributions.'; exit 1; }
test "$(uname -m)" = x86_64
nvidia-smi
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y --no-install-recommends ca-certificates curl gnupg python3 python3-pip python3-venv
if ! docker compose version >/dev/null 2>&1 || ! docker buildx version >/dev/null 2>&1; then
    install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
    chmod a+r /etc/apt/keyrings/docker.asc
    printf 'deb [arch=%s signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu %s stable\n' "$(dpkg --print-architecture)" "$VERSION_CODENAME" > /etc/apt/sources.list.d/docker.list
    apt-get update -qq
    apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
fi
if ! command -v nvidia-ctk >/dev/null 2>&1; then
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' > /etc/apt/sources.list.d/nvidia-container-toolkit.list
    apt-get update -qq
    apt-get install -y --no-install-recommends nvidia-container-toolkit
fi
if test -f /etc/docker/daemon.json; then
    cp -p /etc/docker/daemon.json "/etc/docker/daemon.json.bak.$(date -u +%Y%m%dT%H%M%SZ)"
fi
nvidia-ctk runtime configure --runtime=docker
systemctl enable docker
systemctl restart docker
# Isolate uv from distribution Python packages.
python3 -m venv /opt/aij-uv
/opt/aij-uv/bin/python -m pip install --disable-pip-version-check uv==0.13.5
ln -sfn /opt/aij-uv/bin/uv /usr/local/bin/uv
ln -sfn /opt/aij-uv/bin/uvx /usr/local/bin/uvx
docker --version
docker compose version
docker buildx version
nvidia-ctk --version
python3 --version
uv --version
nvidia-smi
