"""Colab setup for the Parkinson's FoG pipeline — paste this whole file into ONE
Colab cell and run it first.

WHY THIS EXISTS
    In Colab you import the SELF-CONTAINED ``fog_allinone`` module — NOT the
    deployed ``fog`` package. That package lives under ``pi/`` and is the
    torch-free core for the laptop host; it is not on Colab's path, so
    ``from fog.config import ...`` dies with ``ModuleNotFoundError: No module
    named 'fog'``. ``colab/fog_interpret.py`` already imports the right way
    (``from fog_allinone import ...``). This file makes that just work and hands
    your notebook the same names.

WHAT IT DOES
    1. puts ``fog_allinone`` on the path (an uploaded copy, a local clone, or by
       cloning REPO_URL),
    2. fetches the Daphnet Freezing-of-Gait dataset if it isn't already here and
       finds the S*R*.txt folder (UCI nests it under ``dataset_fog_release/``),
    3. imports the symbols your notebook needs and prints a sanity line.

AFTER THIS CELL
    res = run_all(DATA_DIR, sensor=SENSOR, epochs=25, run_optuna_flag=False)

Zero-config route: in the Colab Files panel, upload ``colab/fog_allinone.py``,
then run this cell — no REPO_URL needed. Runs as a plain script too:
``python colab/colab_setup.py``.
"""
import glob
import os
import subprocess
import sys
import urllib.request
import zipfile

# ------------------------------------------------------------------ knobs ----
REPO_URL    = "https://github.com/USERNAME/scenario2_pd.git"  # set only if cloning
SENSOR      = "ankle"          # ankle | thigh | trunk  (Daphnet sensor site)
DAPHNET_URL = "https://archive.ics.uci.edu/static/public/245/daphnet+freezing+of+gait.zip"
CLONE_DIR   = "scenario2_pd"   # where to clone the repo, if cloning
DATA_OUT    = "daphnet"        # where to unzip the dataset


# -------------------------------------------- 1. make fog_allinone importable
def _find_colab_dir():
    """Return the absolute dir holding fog_allinone.py, or None."""
    for c in (".", "colab", f"{CLONE_DIR}/colab", f"/content/{CLONE_DIR}/colab"):
        if os.path.isfile(os.path.join(c, "fog_allinone.py")):
            return os.path.abspath(c)
    return None


_dir = _find_colab_dir()
if _dir is None and REPO_URL and "USERNAME" not in REPO_URL and not os.path.isdir(CLONE_DIR):
    print(f"· cloning {REPO_URL}")
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, CLONE_DIR], check=True)
    _dir = _find_colab_dir()
if _dir is None:
    raise SystemExit(
        "Can't find fog_allinone.py. Easiest fix: in the Colab Files panel upload "
        "colab/fog_allinone.py, then re-run this cell. (Or set REPO_URL above to clone.)")
if _dir not in sys.path:
    sys.path.insert(0, _dir)
print(f"· fog_allinone path: {_dir}")


# -------------------------------------------------- 2. get the Daphnet data --
def _find_data_dir():
    """Return the dir directly holding S*R*.txt (searched recursively), or None."""
    hits = glob.glob("**/S*R*.txt", recursive=True)
    return os.path.dirname(hits[0]) if hits else None


DATA_DIR = _find_data_dir()
if DATA_DIR is None:
    _zip = "daphnet.zip"
    if not os.path.exists(_zip):
        print("· downloading Daphnet (~21 MB) ...")
        urllib.request.urlretrieve(DAPHNET_URL, _zip)
    print("· unzipping ...")
    with zipfile.ZipFile(_zip) as z:
        z.extractall(DATA_OUT)
    DATA_DIR = _find_data_dir()
if DATA_DIR is None:
    raise SystemExit(
        "Downloaded Daphnet but found no S*R*.txt. Unzip it manually and set "
        "DATA_DIR to the folder holding S01R01.txt.")
print(f"· Daphnet data dir: {DATA_DIR}  "
      f"({len(glob.glob(os.path.join(DATA_DIR, 'S*R*.txt')))} runs)")


# -------------------------------------------------- 3. the imports you need --
def _import_fog_allinone(max_tries=4):
    """Import fog_allinone, pip-installing any missing heavy dep and retrying."""
    for _ in range(max_tries):
        try:
            import fog_allinone as fa
            return fa
        except ModuleNotFoundError as e:
            if e.name in (None, "fog_allinone"):
                raise  # path problem, not a dep — surface it
            print(f"· installing missing dep: {e.name}")
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", e.name], check=True)
    import fog_allinone as fa  # last attempt; raises if still broken
    return fa


_fa = _import_fog_allinone()
SAMPLE_RATE = _fa.SAMPLE_RATE
WINDOW_SIZE = _fa.WINDOW_SIZE
WINDOW_HOP  = _fa.WINDOW_HOP
sens_spec   = _fa.sens_spec
FoGNet      = _fa.FoGNet
run_all     = _fa.run_all

print(f"· imports OK — SAMPLE_RATE={SAMPLE_RATE}  WINDOW_SIZE={WINDOW_SIZE}  "
      f"WINDOW_HOP={WINDOW_HOP}  FoGNet={FoGNet.__name__}  sens_spec={sens_spec.__name__}")
print("\nready →  res = run_all(DATA_DIR, sensor=SENSOR, epochs=25, run_optuna_flag=False)")

# -------------------------------------------------- 4. (optional) run it -----
# Uncomment to train + LOSO-evaluate right away (a few minutes on a Colab GPU):
# res = run_all(DATA_DIR, sensor=SENSOR, epochs=25, run_optuna_flag=False)

if __name__ == "__main__":
    pass  # importing/printing above already did the setup
