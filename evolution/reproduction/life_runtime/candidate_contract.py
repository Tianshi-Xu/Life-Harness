"""Shared AST-only bundle contract; never imports generated code in controller."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from evolution.life_optimizer_v25.bundle import filter_operations

def validate_bundle(document, domain, records, snapshot, workspace):
    normalized, rejected = filter_operations(document, domain, records, snapshot, workspace)
    if rejected:
        raise ValueError('Validated proposal changed: operations no longer all valid')
    for operation in normalized['operations']:
        if operation['op'] != 'delete':
            operation['artifact_path'] = Path(workspace) / operation['artifact']
    return normalized
