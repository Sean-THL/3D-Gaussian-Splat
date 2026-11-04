# Evaluation

Planned inputs: reconstructions, shared held-out images, calibrated reference poses, and valid reference depth.

Planned outputs: held-out RGB metrics, depth/pose checks, runtime and peak memory, comparison figures, and limitations. Identify methods that receive additional depth or pose inputs.

Shared dataset selection now produces `frames.csv` and reference `splits.json`; every resize/crop variant inherits the same frame identities and labels. See [the dataset comparison policy](../DATASET_WORKFLOW.md#5-comparison-policy).

These lists are not automatically enforced by external trainers. The existing NVIDIA training launcher uses NVIDIA's own split logic; registration losses/order changes may produce different real splits. Audit or adapt each method's loader before claiming the same held-out-view comparison. Merely writing a reference split does not prevent a model from seeing those images.

General cross-method metrics, evaluation-pose conversion, and backend-specific split adapters remain planned. The scene-specific native 3DGUT depth check below is implemented. Record whether known camera calibration, measured depth, Vicon, or test-image features were used; methods receiving different information need to be labeled accordingly.

## October 25 native 3DGUT depth check

`depth_check.py` evaluates the transferred October 25 checkpoint against recorded aligned depth. It uses the native 3DGUT renderer and original distorted COLMAP cameras at 1280×720, matching the recorded depth grid. It does not retrain or modify the scene. The default inputs are the scene's transferred training reference, the local native RGB dataset, and `oct25_shared_native` timestamp/depth associations.

```bash
source /home/robin/tools/3dgrut/.venv/bin/activate
MAX_JOBS=1 python Workflow/Evaluation/depth_check.py
```

CUDA must be visible to the process. In a sandbox that hides NVIDIA devices, the GPU command needs host execution. This environment's activation script sets its installed CUDA toolkit and Slang compiler paths; invoking its Python executable without activation does not set those compiler paths.

Use `--prepare-only` with NumPy/Pillow to inspect associations and fit scale without CUDA, or `--limit 1` for an installation check. Each invocation creates a new timestamped output directory under the scene's `evaluation/`, with `summary.json`, `per_frame.csv`, metric depth/opacity/mask arrays, and comparison images. The default completed run evaluates 25 of 37 appearance-test views that have a depth match within 40 ms.

Scale is fitted from sparse COLMAP-to-measured-depth ratios in eligible **training** views and held fixed on the evaluation views. This is an estimated metric scale from the same capture, not independent calibration. The renderer supplies accumulated weighted ray-hit distance; divide by accumulated opacity, convert normalized ray range to camera Z, and apply the scale. Metrics exclude invalid sensor depth, measured depth beyond 8 m, and rendered opacity below 0.95; coverage is reported relative to valid in-range measured pixels. No error-based filtering is applied.

The shared reference split is not assumed to match by filename: source timestamps identify frames, and sorted original COLMAP image names determine every-eighth-view evaluation. COLMAP used all views to estimate poses. Timing residuals, assumed sensor depth units, thin objects and depth discontinuities remain limitations. Expected Gaussian depth is not a certified collision surface.

CPU convention and scale-split tests:

```bash
python3 -m unittest Workflow.Evaluation.test_depth_check -v
```
