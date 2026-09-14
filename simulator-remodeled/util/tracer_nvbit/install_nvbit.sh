#!/usr/bin/env bash
set -euo pipefail

BASH_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 1.7.6 fixes WARPSYNC.COLLECTIVE handling used by SM90 FA3 kernels.
NVBIT_VERSION=1.7.6
INSTALL_DIR="$BASH_ROOT/nvbit_release"

case "$(uname -m)" in
    x86_64)
        NVBIT_ARCH=x86_64
        ;;
    aarch64)
        NVBIT_ARCH=aarch64
        ;;
    *)
        echo "[NVBIT] Unsupported host architecture" >&2
        exit 1
        ;;
esac

sdk_matches() {
    [ -s "$1/core/libnvbit.a" ] &&
        [ -f "$1/core/nvbit.h" ] &&
        grep -Fxq "#define NVBIT_VERSION \"$NVBIT_VERSION\"" "$1/core/nvbit.h"
}

if sdk_matches "$INSTALL_DIR"; then
    echo "[NVBIT] $NVBIT_VERSION already installed"
    exit 0
fi

STAGING_DIR="$(mktemp -d "$BASH_ROOT/.nvbit-install.XXXXXX")"
trap 'rm -rf -- "$STAGING_DIR"' EXIT
NVBIT_FILE="nvbit-Linux-$NVBIT_ARCH-$NVBIT_VERSION.tar.bz2"
NVBIT_DIR="nvbit_release_$NVBIT_ARCH"
URL="https://github.com/NVlabs/NVBit/releases/download/v$NVBIT_VERSION/$NVBIT_FILE"

echo "[NVBIT] Installing $NVBIT_VERSION for $NVBIT_ARCH"
if command -v curl >/dev/null 2>&1; then
    curl -fL --retry 3 "$URL" -o "$STAGING_DIR/$NVBIT_FILE"
else
    wget -O "$STAGING_DIR/$NVBIT_FILE" "$URL"
fi
tar -xjf "$STAGING_DIR/$NVBIT_FILE" -C "$STAGING_DIR"

if ! sdk_matches "$STAGING_DIR/$NVBIT_DIR"; then
    echo "[NVBIT] Downloaded SDK is incomplete or has the wrong version" >&2
    exit 1
fi

# Release archives have old timestamps; make must rebuild against the new SDK.
touch "$STAGING_DIR/$NVBIT_DIR/core/nvbit.h" \
    "$STAGING_DIR/$NVBIT_DIR/core/libnvbit.a"

BACKUP_DIR=""
if [ -e "$INSTALL_DIR" ] || [ -L "$INSTALL_DIR" ]; then
    BACKUP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/nvbit-backup.XXXXXX")"
    mv "$INSTALL_DIR" "$BACKUP_DIR/nvbit_release"
    echo "[NVBIT] Previous SDK saved in $BACKUP_DIR/nvbit_release"
fi
if ! mv "$STAGING_DIR/$NVBIT_DIR" "$INSTALL_DIR"; then
    if [ -n "$BACKUP_DIR" ]; then
        mv "$BACKUP_DIR/nvbit_release" "$INSTALL_DIR"
    fi
    exit 1
fi

echo "[NVBIT] Installed $NVBIT_VERSION in $INSTALL_DIR"
