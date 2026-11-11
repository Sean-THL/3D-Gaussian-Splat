# Large native Isaac collection campaigns

First create a frozen version and review three episodes using
[DATASET_VERSIONS.md](DATASET_VERSIONS.md). Bulk collection remains unstarted
until those images/poses and the intended ranges have been reviewed.

`collect_campaign.py` plans and resumes native PhysX collection in sequential
batches, audits every saved pair, and builds separate train/validation/test
episode indexes. It does not train a student or run parallel Isaac instances.

## Default campaign

| Split | Episodes | Seeds |
| --- | ---: | --- |
| Train | 1,000 | 100000–100999 |
| Validation | 100 | 10000000–10000099 |
| Test | 100 | 20000000–20000099 |

Nominal route duration is 30 seconds, randomized to 30–60 seconds; with two
settling seconds at 20 Hz this gives 640–1,240 pairs per episode. Expected
campaign size is approximately **1,128,000 image/action pairs**, with an estimated
range of 768,000–1,488,000 depending on sampled duration and step rounding.

The large profile in [configs/large_collection.json](configs/large_collection.json)
adds the following variations without changing the ordinary preview profile:

- Route scale 1–3.333: the same figure-eight aspect ratio, X/Y spans 0.6–2 m.
- Duration scale 1–2: different speeds/accelerations while retaining the 30 s minimum.
- Altitude 1.0–1.6 m (Z offset −0.2 to +0.4 m around the nominal 1.2 m).
- Initial roll/pitch offsets each ±5°; heading and initial yaw also vary.
- Existing horizontal route shifts, initial errors and actual mass/response changes.
  Orientation then evolves through the plant, rather than arbitrary camera rotations.
- RGB brightness, contrast, saturation, gamma, channel gains, mild blur and noise.

Photometric settings are sampled per episode; noise is drawn per frame. A separate
seeded generator leaves the physical randomization sequence unaffected. Actual
settings and image encoding are recorded in `dataset_metadata.json`. Images
still correspond to the same achieved pose and expert decision timestamp.

This changes route geometry and image appearance. It does **not** move scene
objects, create collision geometry, physically relight Gaussians, or identify
the real drone. Large-route and altitude rendered coverage must be reviewed. The UI permits
altitudes up to 8 m, but this campaign does not sample to 8 m without verified
reconstructed coverage. The native
tracking/solver gates and structural audits reject failed batches but do not
prove obstacle clearance or visual fidelity.

## Storage and duration

This profile stores JPEG RGB at quality 95 with no chroma subsampling, saving
space at the cost of lossy image compression. Other profiles retain PNG by
default. Seventeen pilot views with the new augmentation averaged about 84 KB
per JPEG, with a maximum of 106 KB. That suggests roughly 95 GB of image data at
the expected sample count, plus labels/logs and filesystem overhead; future
viewpoints may compress differently. Measure the first batch on the actual host.

The planner conservatively reserves 260 KB per sample, maximum duration, and
15% margin: about **414 GiB free** for the default full campaign. It checks total
remaining estimated capacity before execution and capacity for each next batch.
The estimate can be adjusted with `--estimated-bytes-per-sample` after measurement.
It is not a disk quota and does not account for other projects consuming space.

Previous native RGB pilots took about 0.09 wall seconds per pair. A million
pairs can therefore take around **a day**, plus batch startup/audits; this is an
estimate, not a benchmark of the new large-route campaign. No large campaign
has been GPU-validated or collected until its actual run reports show success.

## Run on ROB-DASC-06U

First transfer the latest code, including the new runner, augmentation module
and large profile. A Git pull only supplies them after they have been committed
and pushed; generated assets and environments are transferred separately.

Terminal A: start or reuse the Gaussian renderer:

```bash
cd ~/Seans_Project/DASC/3D-Gaussian-Splat
PATH="$HOME/anaconda3/bin:$PATH" SIMULATION_ENV="$HOME/anaconda3/envs/3dgrut" \
  bash Workflow/Simulation/run.sh \
  --config Workflow/Simulation/configs/quadrotor_nmpc.json \
  --pose-render-service --port 8766
```

Do not start another renderer if port 8766 is already serving the correct scene.
The collector verifies renderer checkpoint, alignment and resolution before
collection and pins renderer identity for future resume operations.

Terminal B: prepare and inspect the plan (no GPU collection yet):

```bash
cd ~/Seans_Project/DASC/3D-Gaussian-Splat
campaign_root=data/scene_20261025_200034/isaac_sim/campaigns/large_v1
"$HOME/anaconda3/envs/3dgrut/bin/python" Workflow/Simulation/collect_campaign.py \
  --output "$campaign_root"
```

Start the same plan, detached from SSH:

```bash
nohup "$HOME/anaconda3/envs/3dgrut/bin/python" -u \
  Workflow/Simulation/collect_campaign.py \
  --output "$campaign_root" --execute \
  > "$campaign_root/collector.log" 2>&1 < /dev/null &
```

The renderer must also remain alive (for example, in a persistent terminal/tmux
session). Detaching only the collector does not detach a foreground renderer.
For a managed unattended deployment, start both in persistent sessions and
record their PIDs; do not start duplicate services or kill unrelated processes.

The parent uses the project environment for NumPy/Pillow audits; native physics
uses Isaac's bundled Python through `run_native.sh`.

Monitor from any later SSH connection:

```bash
tail -f "$campaign_root/collector.log"
```

`campaign.json` stores progress. Individual native stdout/stderr logs are under
`batches/<split_batch>/attempt_NNN.log`. Each batch contains at most 25 episodes
by default. `--checks` verifies nominal hover/rate/floor response for every native
process; actual randomized episodes then pass tracking/solver checks and all
image/action audits before entering a split index.

## Resume and failure handling

Rerun the exact same command with the same campaign folder and `--execute`.
Completed audited batches are skipped; an interrupted/failed batch starts a new
attempt with the same seeds. Failed attempt data remain available for diagnosis
and never enter train/validation/test indexes. A failure stops the campaign;
inspect its log before resuming so a persistent failure is not repeatedly retried.

A lock prevents simultaneous collectors using the same folder. A fingerprint
rejects changed config, counts or Python source when resuming. Do not update
collection code midway through a campaign. Use a new folder for changed settings.

For more data in a separate campaign, set larger counts and a distinct seed
offset, e.g. `--seed-offset 1000000`; record previous seed allocations to avoid
duplicates across campaigns. Do not silently change the counts of an existing
campaign. Extremely large requests can require another disk or host.

## Output

```text
large_v1/
  config.json
  campaign.json
  collector.log
  train_episodes.json
  validation_episodes.json
  test_episodes.json
  batches/
    train_00001/
      attempt_001.log
      attempt_001/
        report.json
        episode_0001/
          rgb/000000.jpg
          rgb_actions.jsonl
          dataset_metadata.json
          dataset_audit.json
          episode_parameters.json
          telemetry.csv
        ...
```

Split indexes include only audited batches and store episode paths relative to
the campaign root so downloading a campaign preserves its data references.
Campaign progress/log paths refer to the collection host. Original per-run
configurations and source fingerprints are retained by the native backend.
Overview plots/videos are generated afterward; no inspection MP4 is produced
for every episode during high-volume collection.

The student loader/network and training loop remain the next stage. Dataset
quantity alone does not establish closed-loop policy performance.
