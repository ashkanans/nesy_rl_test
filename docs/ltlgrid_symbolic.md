# Training the DT on the symbolic LTL-grid datasets (Susanna's benchmark)

## 1. Data
Download `Dataset_Symbolic.zip` (Drive folder `1vBQvduY6Vnc7rrQoShhDYi_72Nc3YBfD`, 22 MB) and unzip it:

    unzip Dataset_Symbolic.zip -d <data_dir>
    # -> <data_dir>/Dataset/Symbolic/{distance,three_value}/task{1..10}/dataset_{medium_expert,medium,medium_replay}.h5

## 2. Setup
    pip install -r requirements-core.txt -r requirements-env.txt   # needs torch, h5py, flloat

## 3. Train + evaluate everything (resumable)
    python scripts/ltlgrid_train_all.py --data_dir <data_dir> --out_dir runs/ltlgrid_dt

One run = one (reward, task, dataset, seed): train the DT on the offline data, then evaluate it by playing
120 episodes (12 start cells x 10) in the grid. Results: `runs/ltlgrid_dt/results.csv` (success rate,
failure rate, return, final losses, time). Finished runs are skipped on restart. The GPU is used if available.

Useful options: `--tasks 1 7`, `--rewards distance`, `--datasets medium_expert`, `--seeds 0 1 2`,
`--epochs 30` (default), `--eval_episodes 120`.

With the DFA logic loss: add `--logic_alpha 1.0 --grounder supervised` (`oracle` = exact symbols,
`weak` = learned from rewards only).

Single run: `python scripts/train_dt.py --env ltlgrid --ltlgrid_h5_path <file.h5> --context_len 50 --epochs 30`
(task and reward type are read from the path).

## 4. What was checked
All 60 files (10 tasks x 3 splits x 2 rewards, 120,960 trajectories) replay exactly through our reward machine
(`logic/ltlgrid_dfa.py`, `envs/ltlgrid_env.py`): every reward, every episode end, and `final_observation`.
Tests: `pytest tests/test_ltlgrid_*.py` (set `LTLGRID_DATA_DIR` to `<data_dir>` for the real-data tests).
