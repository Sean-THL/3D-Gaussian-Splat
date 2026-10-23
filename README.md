# 3D Gaussian Splat

ROS bag data preparation, Gaussian reconstruction comparisons, and a workflow toward Isaac Sim integration.

## Project layout

- `Workflow/Data/`: reusable ROS bag extractor, custom message definitions, tests, and usage documentation.
- `Workflow/Calibration/`: planned synchronization and camera/body calibration.
- `Workflow/Reconstruction/`: planned baseline and optimized reconstruction experiments.
- `Workflow/Evaluation/`: planned image/geometry comparisons and resource measurements.
- `Workflow/IsaacSim/`: planned visual assets, collision geometry, and simulation validation.
- `AnySplat/`: existing AnySplat repository, tracked as a submodule pinned to its current commit.
- `diagnostics/`: reconstruction quality notes and small comparison figures.

Start with [Workflow/Data/README.md](Workflow/Data/README.md).

## Clone with AnySplat

```bash
git clone --recurse-submodules REPOSITORY_URL
```

For an existing clone:

```bash
git submodule update --init --recursive
```

Commit code changes made inside `AnySplat/` to that repository first, then commit the updated submodule pointer here. The submodule keeps its original history and remote.

## Data and environments

Git tracks scripts, configuration, documentation, message definitions, and small diagnostic figures. ROS bags, extracted datasets, videos, Gaussian PLY files, model weights, and Python environments remain local and are excluded by `.gitignore`. A Git clone does not contain those assets. Copy the required recording separately into `Workflow/Data/rosbags/BAG_NAME/`, then run the extractor.

Create a new Python environment on each computer; do not copy `.venv` between machines. The Data extractor has its own requirements and does not require ROS or a GPU.
