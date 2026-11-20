#!/usr/bin/env bash
# Install the cross toolchains and QEMU used by scripts/verify_arm.py (tested on Ubuntu 24.04).
# On failure the end of the apt log becomes a GitHub annotation, readable from the run page.
set -uo pipefail
packages="gcc-aarch64-linux-gnu libc6-dev-arm64-cross qemu-user-static gcc-arm-none-eabi libnewlib-arm-none-eabi"
. /etc/os-release && echo "::notice title=Runner OS::$PRETTY_NAME"
log=$(mktemp)
for args in "update" "install -y --no-install-recommends $packages"; do
  # shellcheck disable=SC2086  # $args is split into apt-get arguments on purpose
  if ! sudo apt-get $args > "$log" 2>&1; then
    echo "::error title=apt-get ${args%% *} failed::$(tail -n 15 "$log" | tr '\n' ' ' | cut -c1-1500)"
    exit 1
  fi
done
echo "Installed: $packages"
