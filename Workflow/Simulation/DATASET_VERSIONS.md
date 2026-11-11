# Dataset versions and first-episode review

Create a fresh dataset version whenever route/altitude ranges, appearance,
dynamics, scene model, camera calibration or collection code change. Keep old
versions for traceability and exclude unsuitable episodes through an explicit
training index rather than silently modifying original labels/images.

## First version

From the repository root:

```bash
python3 Workflow/Simulation/prepare_dataset_version.py --version campaign_v1
```

Default location:

```text
data/scene_20261025_200034/isaac_sim/campaigns/campaign_v1/
  config.json                    read-only settings snapshot
  version.json                   hashes, code revision and version lineage
  calibration_snapshot.json      copy of the candidate transform
  source_snapshot/               source/config snapshot; no model-weight copy
  pilot_test/
    README.md
    run_pilot.sh                 collects the first three native episodes
    runs/                       new folder per attempt; initially empty
    reviews/review_template.json
  collection/                   reserved for a later full campaign
```

`pilot_test` means development inspection. Its seeds start at 90000000 and are
separate from the campaign's train/validation/final-test allocations. Reviewing
these episodes makes them unsuitable as an untouched final test set.

The first version snapshots the current large profile: route X/Y dimensions
0.6–2 m, altitude 1.0–1.6 m, duration 30–60 s, initial roll/pitch ±5°, heading,
physics variation and RGB augmentation. The configuration allows broader
coverage than the old pilot; review is required before considering it useful
for training. Existing historical pilots are not relabeled as this version.

The setup command creates folders/provenance only. It starts no simulation and
does not create training samples. Existing version directories are refused.

## Run only the first few episodes

Transfer code and the new version folder to the GPU host first. Its raw assets
must also exist there; relative asset paths support the existing original or
transferred reconstruction layout. Do not transfer Python environments.

Start the pose renderer as described in [START_HERE.md](START_HERE.md).
In a second remote terminal, from the repository root, deactivate Conda and run:

```bash
bash data/scene_20261025_200034/isaac_sim/campaigns/campaign_v1/pilot_test/run_pilot.sh
```

The wrapper checks the configuration, source/model/calibration hashes and
running renderer identity, then invokes native `--checks --randomize` for
three full episodes. It saves each attempt under `pilot_test/runs/run_*`.
It does not start the million-pair campaign. The Gaussian model is referenced,
not duplicated into the version. Snapshots made from uncommitted code explicitly
record that state and preserve the source files.

Run the normal dataset visualization against the resulting run directory:

```bash
python3 Workflow/Simulation/visualize_dataset.py PATH_TO_PILOT_RUN
python3 Workflow/Simulation/plot_collected_trajectories.py PATH_TO_PILOT_RUN
```

Passing audits means data consistency, not acceptable scene geometry. Review
all pilot episodes for camera/attitude consistency, view coverage, artifacts,
trajectory/altitude coverage, augmentation and expert tracking. At matching
recorded camera poses, compare against measured RGB/depth. Note weak surfaces
or viewpoints and decide whether to restrict the route, improve reconstruction,
fix calibration, or exclude episodes. The scene has no verified room colliders.

Copy the review template to a run-specific JSON file and record the decision.
Keep review notes outside the immutable original run. Bulk collection should
remain unstarted while this data-quality review is pending.

## Corrected version

Edit a working configuration outside the version folder, then create a new
version, recording its parent:

```bash
python3 Workflow/Simulation/prepare_dataset_version.py \
  --version campaign_v2 \
  --config Workflow/Simulation/configs/large_collection.json \
  --parent data/scene_20261025_200034/isaac_sim/campaigns/campaign_v1
```

The same pilot seeds permit controlled development comparisons between versions;
allocate distinct seed ranges for actual additional training campaigns using
`collect_campaign.py --seed-offset ...`. Do not mix versions automatically.

After an acceptable review, the large campaign runner can use
`--config <version>/config.json --output <version>/collection`. Archive complete
versions to lab/cloud storage when needed, verify the backup, and keep manifests,
code snapshots and review records with the images/labels. No Google Drive
upload is performed by these tools.
