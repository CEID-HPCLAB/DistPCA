#!/bin/bash

set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd -- "${SCRIPT_DIR}/../" && pwd)"

URL="https://reichdata.hms.harvard.edu/pub/datasets/humanOrigins/Harvard_HGDP-CEPH.tgz"

RAW_TGZ="${BASE_DIR}/experiments/Harvard_HGDP-CEPH.tgz"
RAW_DATA="${BASE_DIR}/experiments/hgdp"
RAW_DIR="${BASE_DIR}/experiments/Harvard_HGDP-CEPH"
QC_DATA="${BASE_DIR}/experiments/hgdp.qc"
PRUNED_DATA="${BASE_DIR}/experiments/hgdp.qc.pruned"

MAF=0.01
LD=1000
VAF=50
THRESHOLD=0.2

if [[ -f "${PRUNED_DATA}.bed" &&
      -f "${PRUNED_DATA}.bim" &&
      -f "${PRUNED_DATA}.fam" ]]; then
    echo "Pruned HGDP dataset already exists. Exiting..."
    exit 0
fi

echo "Downloading HGDP dataset..."

wget -q -O "$RAW_TGZ" "$URL"

tar -xzf "$RAW_TGZ" -C "${BASE_DIR}/experiments" > /dev/null 2>&1

rm -f "$RAW_TGZ"

plink \
    --file "$RAW_DIR/all_snp" \
    --make-bed \
    --out "$RAW_DATA" \
    > /dev/null

rm -rf "$RAW_DIR"

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

rm -f "${RAW_DATA}".{bed,bim,fam,log,hh}
rm -f "${QC_DATA}".{bed,bim,fam,log,hh}
rm -f "${QC_DATA}.prune".{prune.in,prune.out,log,hh}
rm -f "${PRUNED_DATA}".{log,hh}