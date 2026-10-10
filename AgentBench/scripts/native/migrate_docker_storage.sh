#!/usr/bin/env bash
# Administrator-run migration for this host. Existing images/state are retained.
set -Eeuo pipefail
[[ ${EUID} -eq 0 ]] || { echo 'Run this script with sudo.' >&2; exit 1; }
docker_source=/data3/docker
containerd_source=/var/lib/containerd
destination=/data/docker-life-harness
[[ -d "$docker_source" && ! -L "$docker_source" ]]
[[ -d "$containerd_source" && ! -L "$containerd_source" ]]
[[ ! -e "$destination" ]]
[[ $(stat -c %d /data) != $(stat -c %d "$docker_source") ]]
[[ -z $(docker ps -q) ]] || { echo 'Running containers exist; stop and review before migrating.' >&2; exit 1; }
required_bytes=$(du -sb "$docker_source" "$containerd_source" | awk '{sum += $1} END {printf "%.0f",sum}')
available_bytes=$(df -B1 --output=avail /data | tail -n 1)
(( available_bytes > required_bytes + 10 * 1024 * 1024 * 1024 )) || {
  echo 'Insufficient destination space for a lossless migration.' >&2; exit 1;
}
echo "Moving Docker and containerd to $destination; no image/container deletion."
docker_moved=0
containerd_moved=0
rollback() {
  trap - ERR
  echo 'Migration failed; restoring original paths before restarting services.' >&2
  if (( containerd_moved )); then
    [[ ! -L "$containerd_source" ]] || unlink "$containerd_source"
    [[ -e "$containerd_source" ]] || mv "$destination/containerd" "$containerd_source"
  fi
  if (( docker_moved )); then
    [[ ! -L "$docker_source" ]] || unlink "$docker_source"
    [[ -e "$docker_source" ]] || mv "$destination/docker" "$docker_source"
  fi
  systemctl start containerd docker.socket docker
  exit 1
}
trap rollback ERR
systemctl stop docker.socket docker containerd
mkdir -m 0700 "$destination"
mv "$docker_source" "$destination/docker"
docker_moved=1
ln -s "$destination/docker" "$docker_source"
mv "$containerd_source" "$destination/containerd"
containerd_moved=1
ln -s "$destination/containerd" "$containerd_source"
sync
systemctl start containerd docker.socket docker
docker info --format 'Docker server {{.ServerVersion}}, configured root {{.DockerRootDir}}'
echo "Docker storage: $(readlink -f "$docker_source")"
echo "containerd storage: $(readlink -f "$containerd_source")"
df -h / /data
trap - ERR
echo 'Migration complete. Existing images and state were retained.'
