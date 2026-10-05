# Trimmed from fschmid56/EfficientAT's helpers/utils.py (MIT License).
# Only NAME_TO_WIDTH is kept -- the original file's `models/mn/model.py` needs it, but the
# original also loads `metadata/class_labels_indices.csv` (AudioSet's 527-class label file)
# as an import-time side effect, purely for a `labels`/`ids`/`classes_num` this project
# never uses. Vendoring that CSV just to satisfy an unused import wasn't worth it, so this
# file keeps only the one function actually needed.


def NAME_TO_WIDTH(name):
    mn_map = {
        'mn01': 0.1,
        'mn02': 0.2,
        'mn04': 0.4,
        'mn05': 0.5,
        'mn06': 0.6,
        'mn08': 0.8,
        'mn10': 1.0,
        'mn12': 1.2,
        'mn14': 1.4,
        'mn16': 1.6,
        'mn20': 2.0,
        'mn30': 3.0,
        'mn40': 4.0,
    }

    dymn_map = {
        'dymn04': 0.4,
        'dymn10': 1.0,
        'dymn20': 2.0
    }

    try:
        if name.startswith('dymn'):
            w = dymn_map[name[:6]]
        else:
            w = mn_map[name[:4]]
    except Exception:
        w = 1.0

    return w
