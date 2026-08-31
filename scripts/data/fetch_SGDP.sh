#!/bin/bash

set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd -- "${SCRIPT_DIR}/../" && pwd)"

URL="https://sharehost.hms.harvard.edu/genetics/reich_lab/sgdp/variant_set/cteam_extended.v4.maf0.1perc"

RAW_DATA="${BASE_DIR}/experiments/cteam_extended.v4.maf0.1perc"
QC_DATA="${BASE_DIR}/experiments/sgdp.qc"
PRUNED_DATA="${BASE_DIR}/experiments/sgdp.qc.pruned"

MAF=0.01
LD=1000
VAF=50
THRESHOLD=0.2

if [[ -f "${PRUNED_DATA}.bed" &&
      -f "${PRUNED_DATA}.bim" &&
      -f "${PRUNED_DATA}.fam" ]]; then
    echo "Pruned SGDP dataset already exists. Exiting..."
    exit 0
fi

echo "Downloading SGDP dataset..."

wget -q -O "${RAW_DATA}.bed" "${URL}.bed"
wget -q -O "${RAW_DATA}.bim.zip" "${URL}.bim.zip"
wget -q -O "${RAW_DATA}.fam" "${URL}.fam"

unzip -q "${RAW_DATA}.bim.zip" -d "$(dirname "$RAW_DATA")"

rm -f "${RAW_DATA}.bim.zip"

echo "Applying MAF filter..."

plink \
    --bfile "$RAW_DATA" \
    --maf "$MAF" \
    --make-bed \
    --out "$QC_DATA" \
    > /dev/null

echo "Performing LD pruning..."

plink \
    --bfile "$QC_DATA" \
    --indep-pairwise "$LD" "$VAF" "$THRESHOLD" \
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
rm -f "${QC_DATA}".{bed,bim,fam,log,hh,nosex}
rm -f "${QC_DATA}.prune".*
rm -f "${PRUNED_DATA}".{log,hh,nosex}