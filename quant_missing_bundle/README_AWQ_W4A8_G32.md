AWQ W4A8 (group_size=32) missing quantization bundle

This folder is prepared for uploading missing quantization code that was not included in GitHub.

Included:
- smolvlm2_kv260/quant_pipeline/ (source only, cache removed)
- smolvlm2_kv260/scripts/run_quant_experiment.py
- smolvlm2_kv260/scripts/quantize_weights.py
- smolvlm2_kv260/scripts/check_hardware_export.py
- smolvlm2_kv260/scripts/replay_hardware_export.py
- smolvlm2_kv260/scripts/awq_lmhead_lowrank_eval.py
- smolvlm2_kv260/scripts/awq_mlp_channel_prune_eval.py
- smolvlm2_kv260/requirements-quant-pipeline.txt
- smolvlm2_kv260/hardware_export_manifest_samples/w4a8_awq_g32/
  - hardware_export_manifest.json
  - hardware_export_manifest.repaired.json

Not included:
- large binary exports (weights/meta/gamma/lmhead bins)
- local model folder (for example /Users/miracle/smolvlm_local)
- checkpoints under quant_experiments*

Recommended upload steps:
1) Copy this folder into your repo root (or keep it as is under HLS_1).
2) Commit as a dedicated patch:
   git add quant_missing_bundle
   git commit -m "Add missing AWQ W4A8 g32 quantization sources"
   git push

If you want these files merged back to their original paths in repo:
- remove ignore rules for quant_pipeline in .gitignore
- then move/copy quant_missing_bundle/smolvlm2_kv260/quant_pipeline back to smolvlm2_kv260/quant_pipeline
