import json
from pathlib import Path

import yaml


def available_models(path='config.yaml'):
    """Model variants whose files are present."""
    path = Path(path); cfg = yaml.safe_load(path.read_text())
    return [k for k, d in cfg['model_variants'].items() if (path.parent / d / 'fusion.json').exists()]


def load_config(path='config.yaml', model=None):
    """Load config.yaml + chosen model variant's metadata (feature order, labels, fusion weight)."""
    path = Path(path)
    cfg = yaml.safe_load(path.read_text())
    root = path.parent
    cfg['active_model'] = model or cfg.get('active_model', 'lite')
    if model is None and not (root / cfg['model_variants'].get(cfg['active_model'], '') / 'fusion.json').exists():
        avail = available_models(path)                    # default missing -> pehla available model
        if avail:
            cfg['active_model'] = avail[0]
    if cfg['active_model'] not in cfg['model_variants']:
        raise KeyError(f"Model '{cfg['active_model']}' is not in the config: {list(cfg['model_variants'])}")
    cfg['models_dir'] = str((root / cfg['model_variants'][cfg['active_model']]).resolve())
    cfg['demo_stream'] = str((root / cfg['demo_stream']).resolve())
    meta_path = Path(cfg['models_dir']) / 'fusion.json'
    if not meta_path.exists():
        raise FileNotFoundError(f'{meta_path} not found - download the model files from Colab and put them in this folder')
    meta = json.loads(meta_path.read_text())
    cfg['features'] = meta['features']
    cfg['labels'] = meta['labels']
    cfg['fusion_w'] = float(meta['w'])
    cfg['context_k'] = int(meta.get('K', cfg.get('context_k', 10)))
    cfg['model_info'] = {k: meta[k] for k in ('variant', 'max_depth', 'max_trees') if k in meta}
    thr = cfg.get('attack_threshold')
    cfg['attack_threshold'] = thr.get(cfg['active_model']) if isinstance(thr, dict) else thr
    return cfg
