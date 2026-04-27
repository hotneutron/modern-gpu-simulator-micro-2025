export BASH_ROOT="$( cd "$( dirname "$BASH_SOURCE" )" && pwd )"

ARCH=$(uname -m)
if [ "$ARCH" = "aarch64" ]; then
    NVBIT_FILE="nvbit-Linux-aarch64-1.7.5.tar.bz2"
    NVBIT_DIR="nvbit_release_aarch64"
else
    NVBIT_FILE="nvbit-Linux-x86_64-1.7.5.tar.bz2"
    NVBIT_DIR="nvbit_release_x86_64"
fi

rm -rf $BASH_ROOT/nvbit_release
wget https://github.com/NVlabs/NVBit/releases/download/v1.7.5/$NVBIT_FILE
tar -xf $NVBIT_FILE -C $BASH_ROOT
rm $NVBIT_FILE
mv $BASH_ROOT/$NVBIT_DIR $BASH_ROOT/nvbit_release


