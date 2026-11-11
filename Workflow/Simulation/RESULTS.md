# Browse, download and visualize results

Native Isaac collection files live on ROB-DASC-06U at:

```text
/home/tk/Seans_Project/DASC/3D-Gaussian-Splat/data/scene_20261025_200034/isaac_sim/
```

## Browse from Ubuntu Files

Open Files on your local laptop, press Ctrl+L, and enter:

```text
sftp://tk@141.212.76.61/home/tk/Seans_Project/DASC/3D-Gaussian-Splat/data/scene_20261025_200034/isaac_sim/
```

Enter the remote account password in the file manager. Each `pilot_*`, `batch_*`
or `native_*` folder is a collection run. Open `report.json` first to check its
status; `episode_*` folders contain images, labels and flight records.

## Browse from a terminal

```bash
ssh tk@141.212.76.61
cd ~/Seans_Project/DASC/3D-Gaussian-Splat/data/scene_20261025_200034/isaac_sim
ls -lt
```

To inspect the existing pilot:

```bash
cat pilot_20261110T164842Z/report.json
ls pilot_20261110T164842Z/episode_0001
```

Type `exit` when finished.

## Download one completed run to the local project

Run this on ROB-DASC-04, not inside SSH. Change the run name for future batches:

```bash
cd /home/robin/Seans_Project/DASC/Gaussian_Splat
run_name=pilot_20261110T164842Z
mkdir -p "data/scene_20261025_200034/isaac_sim/$run_name"
rsync -avh --progress \
  "tk@141.212.76.61:/home/tk/Seans_Project/DASC/3D-Gaussian-Splat/data/scene_20261025_200034/isaac_sim/$run_name/" \
  "data/scene_20261025_200034/isaac_sim/$run_name/"
```

This copies files without deleting the remote run. Running it again transfers
new/changed files rather than copying everything again. To fetch several runs,
repeat with each run name. Check storage before copying large campaigns.

## Generate plots and videos locally

With the complete run downloaded and the visualization dependencies installed:

```bash
python3 Workflow/Simulation/visualize_dataset.py \
  "data/scene_20261025_200034/isaac_sim/$run_name"
python3 Workflow/Simulation/plot_collected_trajectories.py \
  "data/scene_20261025_200034/isaac_sim/$run_name"
```

The first command audits episodes and creates a `visualization/` folder inside
each episode, containing `rollout.mp4`, overview plots and an RGB contact sheet.
The second creates `randomized_trajectories.png`, `.pdf` and `.svg` at the run
root. Both use actual saved data; they do not rerun physics or GPU rendering.
Videos/plots generated locally are not automatically uploaded to the remote host.

For every future batch: collect into a new run folder, check its report, audit
the episodes, download the complete run, generate visualizations, then review
the results before adding them to a training split. Keep the run configuration,
report and provenance together with the RGB and labels.

The confirmed pilot backend is `isaac_physx_forces_and_torques`: rigid-body
physics with thrust and rate-controller torques. The NMPC prediction model
remains a nominal thrust/body-rate approximation. Neither is an independently
identified model of the flight hardware.
