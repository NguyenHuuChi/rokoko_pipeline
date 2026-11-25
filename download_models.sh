#!/usr/bin/env bash
# weights -> models/, WiLoR code + weights -> third_party/WiLoR (no MANO: see README).  Needs: hf (huggingface_hub), git
set -euo pipefail
cd "$(dirname "$0")"
hf download NguyenHuuChi/rokoko_pipeline_models --local-dir models
(cd models && sha256sum -c SHA256SUMS)
W=third_party/WiLoR
[ -d $W/.git ] || git clone https://github.com/rolpotamias/WiLoR.git $W
git -C $W checkout -q fcb911312a38fa8badd30d9656a167485d61b8f9
hf download rolpotamias/WiLoR pretrained_models/wilor_final.ckpt --repo-type space --local-dir $W
echo "now put MANO_RIGHT.pkl into $W/mano_data/ and run: python check_setup.py"
