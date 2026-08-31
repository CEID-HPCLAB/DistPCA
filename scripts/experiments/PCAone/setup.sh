#!/bin/bash

source /opt/intel/oneapi/setvars.sh --force

export ONEAPI_COMPILER=/opt/intel/oneapi/compiler/latest

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd -- "${SCRIPT_DIR}/../../../" && pwd)"
PCAONE_DIR="${BASE_DIR}/PCAone"

if [ ! -d "${PCAONE_DIR}" ]; then

    echo "PCAone not found."
    echo "Cloning PCAone..."

    if ! git clone https://github.com/Zilong-Li/PCAone.git "${PCAONE_DIR}"; then
        echo "ERROR: git clone failed"
        exit 1
    fi

    echo "PCAone cloned successfully"

else

    echo "PCAone already exists"

fi

cd "${PCAONE_DIR}" || exit 1

TARGET_FILE="${PCAONE_DIR}/src/Data.cpp"

# To enable block-size configuration (# SNPs per block) via the --memory argument
sed -i 's/if (params\.memory > 1\.1 \* m)/if (params.memory > m)/' "$TARGET_FILE"
sed -i 's/blocksize = (unsigned int)ceil(/blocksize = (unsigned int)floor(/' "$TARGET_FILE"

if make -j4 \
    MKLROOT=/opt/intel/oneapi/mkl/latest \
    ONEAPI_COMPILER=/opt/intel/oneapi/compiler/latest
then
    echo "PCAone built successfully"
else
    echo "ERROR: PCAone build failed"
    exit 1
fi