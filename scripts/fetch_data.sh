#!/usr/bin/env bash
# Downloads the public test data into data/. Usage: scripts/fetch_data.sh [subject ...]  (default: sub-01)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/dicom data/ds000114
if [ ! -d data/dicom/dcm_qa_nih ]; then
  git clone -q --depth 1 https://github.com/neurolabusc/dcm_qa_nih data/dicom/dcm_qa_nih
  rm -rf data/dicom/dcm_qa_nih/.git
fi
# Oblique Siemens series (B1c): only the Orientation folder and its references
if [ ! -d data/dicom/dcm_qa ]; then
  git clone -q --depth 1 --filter=blob:none --sparse https://github.com/neurolabusc/dcm_qa data/dicom/dcm_qa
  (cd data/dicom/dcm_qa && git sparse-checkout set --no-cone 'In/Orientation/*' '/Ref/ax_*' '/Ref/cor_*' '/Ref/sag_*' '/Ref/fMRI_MB_*' && rm -rf .git)
fi
# Gantry-tilted head CT (B1c): GE folder and its references only
if [ ! -d data/dicom/dcm_qa_ct ]; then
  git clone -q --depth 1 --filter=blob:none --sparse https://github.com/neurolabusc/dcm_qa_ct data/dicom/dcm_qa_ct
  (cd data/dicom/dcm_qa_ct && git sparse-checkout set --no-cone 'In/GE/*' '/Ref/_2*' && rm -rf .git)
fi
B=https://s3.amazonaws.com/openneuro.org/ds000114
for f in task-fingerfootlips_events.tsv task-fingerfootlips_bold.json dataset_description.json; do
  [ -f data/ds000114/$f ] || curl -sfL -o data/ds000114/$f $B/$f
done
subs=("${@:-sub-01}")
for sub in "${subs[@]}"; do
  for ses in test retest; do
    for f in anat/${sub}_ses-${ses}_T1w.nii.gz func/${sub}_ses-${ses}_task-fingerfootlips_bold.nii.gz; do
      dst=data/ds000114/$sub/ses-$ses/$f
      mkdir -p "$(dirname "$dst")"
      [ -f "$dst" ] || curl -sfL -o "$dst" "$B/$sub/ses-$ses/$f"
    done
  done
done
echo "data ready: $(du -sh data | cut -f1)"
