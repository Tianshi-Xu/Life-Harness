#!/usr/bin/env bash
# Diagnostic only. Never executes benchmark/model commands on the host.
set -euo pipefail
cd "$(dirname "$0")/../.."
bwrap="$PWD/.native/sysroot/usr/bin/bwrap"
rootfs="$PWD/.native/os-rootfs"
"$bwrap" --unshare-all --uid 0 --gid 0 --ro-bind "$rootfs" / --dev /dev --chdir /root \
  /bin/bash -c 'test ! -e /mnt/data2/xts/harness && test ! -e /proc/1/root && echo root_filesystem_isolated'
# Full OS benchmark readiness requires a private procfs, not host /proc.
"$bwrap" --unshare-all --uid 0 --gid 0 --ro-bind "$rootfs" / --dev /dev --proc /proc --chdir /root \
  /bin/bash -c 'test -d /proc/self && echo private_procfs_ready'
