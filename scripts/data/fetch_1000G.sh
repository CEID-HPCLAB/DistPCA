#!/bin/bash

set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd -- "${SCRIPT_DIR}/../" && pwd)"

URL="https://api.figshare.com/v2/articles/9208979/download"
RAW_ZIP="1000G.zip"
RAW_DATA="1000G_phase3_common_norel"
QC_DATA="1000G.qc"
PRUNED_DATA="${BASE_DIR}/experiments/1000G.qc.pruned"

if [[ -f "${PRUNED_DATA}.bed" && \
      -f "${PRUNED_DATA}.bim" && \
      -f "${PRUNED_DATA}.fam" ]]; then
    echo "Pruned dataset already exists. Exiting..."
    exit 0
fi

MAF=0.01
LD=1000
VAF=50
THRESHOLD=0.2

echo "Downloading 1000 Genomes dataset..."

wget -q -O "$RAW_ZIP" "$URL"
unzip -q "$RAW_ZIP"

rm -f "$RAW_DATA.fam2"
rm -f "1000G-phase3-common-norel.R" 

unzip -q "${RAW_DATA}.zip"

echo "Applying MAF filter..."

plink \
    --bfile "$RAW_DATA" \
    --maf $MAF \
    --make-bed \
    --out "$QC_DATA" \
    > /dev/null

echo "Performing LD pruning..."

plink \
    --bfile "$QC_DATA" \
    --indep-pairwise $LD $VAF $THRESHOLD \
    --out "$QC_DATA.prune" \
    > /dev/null

echo "Creating pruned dataset..."

plink \
    --bfile "$QC_DATA" \
    --extract "${QC_DATA}.prune.prune.in" \
    --make-bed \
    --out "$PRUNED_DATA" \
    > /dev/null

echo "Pruned dataset created at: ${PRUNED_DATA}"

rm -f "$RAW_ZIP" "${RAW_DATA}".{zip,bed,bim,fam,log,fam2}
rm -f "${QC_DATA}".* "${QC_DATA}.prune".*
rm -f "${PRUNED_DATA}.log"