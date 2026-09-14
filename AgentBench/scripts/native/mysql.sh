#!/usr/bin/env bash
# Foreground, dedicated loopback-only MySQL. Stop with SIGTERM/Ctrl-C.
set -euo pipefail
cd "$(dirname "$0")/../.."
agentbench_mysql_data=${AGENTBENCH_MYSQL_DATA:-/tmp/life-agentbench-native/mysql}
mkdir -p "$agentbench_mysql_data" .native/logs
chmod 700 "$agentbench_mysql_data"
export LD_LIBRARY_PATH="$PWD/.native/sysroot/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
args=(--no-defaults --user="$(id -un)" --basedir="$PWD/.native/sysroot/usr" --datadir="$agentbench_mysql_data")
if [[ ! -d "$agentbench_mysql_data/mysql" ]]; then
  .native/sysroot/usr/sbin/mysqld "${args[@]}" --initialize-insecure
fi
exec .native/sysroot/usr/sbin/mysqld "${args[@]}" \
  --bind-address=127.0.0.1 --port=13306 \
  --socket="$agentbench_mysql_data/mysql.sock" --pid-file="$agentbench_mysql_data/mysql.pid" \
  --mysqlx=OFF --default-time-zone=+00:00 --secure-file-priv=NULL --innodb-buffer-pool-size=128M \
  --log-error="$PWD/.native/logs/mysql.log"
