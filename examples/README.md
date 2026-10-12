# Examples

- `babs-ants-nidm-bids-study.yaml` -- a complete BABS configuration for this
  app. See "Running under BABS" in the top-level README for the two settings
  it cannot run without (`zip_foldernames` keyed by subject, and GPU flags).
- `dataset_description.json` -- a minimal BIDS `dataset_description.json` to
  copy when assembling a test dataset.

There is no sample imaging data here. To try the app, use any BIDS dataset
with T1w images and run one subject:

```bash
./run_container.sh ants-nidm_bidsapp.sif /path/to/bids /path/to/output 01
```

`--participant-label` takes one subject per run; run subjects as separate jobs.
