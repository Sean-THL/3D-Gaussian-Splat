#!/usr/bin/env python3
"""Inspect scene assets or create descriptive relative symlinks without moving data."""
import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def contained(path, parent):
    if not path.is_relative_to(parent):
        raise ValueError(f'Path outside expected root: {path}')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registry', type=Path, default=Path(__file__).with_name('scene_oct25_assets.json'))
    parser.add_argument('--create-links', action='store_true', help='Create named symlinks; never overwrite an existing destination')
    parser.add_argument('--verify-hashes', action='store_true', help='Verify known asset identities, including the two final PLY copies')
    args = parser.parse_args()
    registry = json.loads(args.registry.read_text())
    organized = contained(Path(os.path.abspath(ROOT/registry['organized_root'])), ROOT)
    rows = []
    for asset in registry['assets']:
        source = contained(Path(os.path.abspath(ROOT/asset['path'])), ROOT)
        alias = contained(Path(os.path.abspath(organized/asset['alias'])), organized)
        present = source.exists()
        row = dict(id=asset['id'], source=asset['path'], alias=str(alias.relative_to(ROOT)), present=present)
        if present and args.verify_hashes and 'sha256' in asset:
            digest = hashlib.sha256()
            with source.open('rb') as stream:
                for chunk in iter(lambda: stream.read(1024*1024), b''):
                    digest.update(chunk)
            row['sha256'] = digest.hexdigest()
            if row['sha256'] != asset['sha256']:
                raise ValueError(f"Unexpected asset hash: {asset['id']}")
        if args.create_links and present:
            alias.parent.mkdir(parents=True, exist_ok=True)
            if alias.is_symlink():
                if alias.resolve() != source.resolve():
                    raise FileExistsError(f'Refusing to replace existing link: {alias}')
            elif alias.exists():
                raise FileExistsError(f'Refusing to replace existing file/directory: {alias}')
            else:
                alias.symlink_to(os.path.relpath(source, alias.parent), target_is_directory=source.is_dir())
            row['link_valid'] = alias.resolve() == source.resolve()
        print(('FOUND  ' if present else 'MISSING')+' '+asset['alias'])
        rows.append(row)
    if args.create_links:
        organized.mkdir(parents=True, exist_ok=True)
        report = dict(scene_id=registry['scene_id'], registry=str(args.registry),
                      missing=sum(not row['present'] for row in rows), assets=rows)
        (organized/'asset_inventory.json').write_text(json.dumps(report, indent=2)+'\n')
        lines = ['# October 25 scene: named asset shortcuts', '',
                 'These are relative symlinks to existing data. Originals were not moved or renamed.',
                 'Run `python3 Workflow/organize_scene.py --create-links --verify-hashes` from the repository root to refresh.', '',
                 'COLMAP input was 296 PNG frames extracted from the rosbag; no MP4 was its input.',
                 'The final PLY is an export. The active renderer loads the matching checkpoint.', '',
                 '| Asset | Named shortcut | Meaning |', '| --- | --- | --- |']
        for asset, row in zip(registry['assets'], rows):
            label = f"[{asset['alias']}]({asset['alias']})" if row['present'] else f"{asset['alias']} (missing on this machine)"
            lines.append(f"| {asset['name']} | {label} | {asset['description']} |")
        (organized/'README.md').write_text('\n'.join(lines)+'\n')
    missing = sum(not row['present'] for row in rows)
    print(f'\n{len(rows)-missing}/{len(rows)} assets present; {missing} missing on this machine.')
    return 1 if missing else 0


if __name__ == '__main__':
    raise SystemExit(main())
