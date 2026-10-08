# Third-party boundary

- Histoformer is not vendored. Fetch pinned commit `1f045f06c03551c31504d8042dbe6ff9b9569108` with `scripts/fetch_histoformer.sh`. The pinned upstream tree has no repository-level license file; this package does not assign it the project's Apache license.
- DA-CLIP/OpenCLIP is an external conditioner dependency. Use the matching checkpoint and source under their respective terms.
- The SMoE integration files in this package are distributed under the project license, subject to the external dependencies above.
