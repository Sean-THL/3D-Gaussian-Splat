# Evaluation

Planned inputs: reconstructions, shared held-out images, calibrated reference poses, and valid reference depth.

Planned outputs: held-out RGB metrics, depth/pose checks, runtime and peak memory, comparison figures, and limitations. Identify methods that receive additional depth or pose inputs.

Shared dataset selection now produces `frames.csv` and reference `splits.json`; every resize/crop variant inherits the same frame identities and labels. See [the dataset comparison policy](../DATASET_WORKFLOW.md#5-comparison-policy).

These lists are not automatically enforced by external trainers. The existing NVIDIA training launcher uses NVIDIA's own split logic; registration losses/order changes may produce different real splits. Audit or adapt each method's loader before claiming the same held-out-view comparison. Merely writing a reference split does not prevent a model from seeing those images.

Metrics, evaluation-pose conversion, and backend-specific split adapters are not implemented yet. Record whether known camera calibration, measured depth, Vicon, or test-image features were used; methods receiving different information need to be labeled accordingly.
