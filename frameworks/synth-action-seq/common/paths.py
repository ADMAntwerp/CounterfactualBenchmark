from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
SRC_DIR = Path(__file__).parent.parent.resolve()
MODELS_DIR = SRC_DIR / 'models'
RESULTS_DIR = SRC_DIR / 'results'
GERMAN_DIR = MODELS_DIR / 'german'
ADULT_DIR = MODELS_DIR / 'adult'
FANNIEMAE_DIR = MODELS_DIR / 'fanniemae'
QUICKDRAW_DIR = MODELS_DIR / 'quickdraw'


def get_ckpt_path(model_name, ckpt_filename):
    model_dir = MODELS_DIR / model_name
    if not model_dir.exists():
        raise ValueError('Model directory does not exist')
    ckpt_path = model_dir / ckpt_filename
    if not ckpt_path.exists():
        raise ValueError('Specified checkpoint file does not exist')
    return str(ckpt_path)
