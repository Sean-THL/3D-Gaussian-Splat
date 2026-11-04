"""Resolve a copied training tree or the same original tree on the training host."""
from pathlib import Path


def resolve_training_asset(root, requested):
    requested = Path(requested).expanduser()
    path = requested if requested.is_absolute() else Path(root)/requested
    if path.is_file():
        return path
    parts = requested.parts
    if 'training_reference' in parts:
        suffix = parts[parts.index('training_reference')+1:]
        if suffix and suffix[0] == 'Workflow':
            original = Path(root).joinpath(*suffix)
            if original.is_file():
                return original
    raise FileNotFoundError(f'Training asset missing: {path}. Transfer it or update the simulator configuration.')
