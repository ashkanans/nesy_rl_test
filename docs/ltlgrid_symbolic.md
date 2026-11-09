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

With the DFA logic loss: add `--logic_alphas 0 0.1 0.5 0.9 --grounder supervised` (`oracle` = exact
symbols, `weak` = learned from rewards only). The loss is `(1 - alpha) * imitation + alpha * logic`, so
alpha is in [0, 1] and 0 is the baseline; every alpha is one run per seed, stored under `alpha<a>/`.
`--logic_weighting additive` gives the older `imitation + alpha * logic`. Draw the alpha-sweep figures
(one per reward, split and task, plus one averaged over tasks) with
`python scripts/ltlgrid_alpha_figures.py --results <out_dir> [<out_dir> ...] --out_dir <figures_dir>`.

Single run: `python scripts/train_dt.py --env ltlgrid --ltlgrid_h5_path <file.h5> --context_len 50 --epochs 30`
(task and reward type are read from the path).

## 4. What was checked
All 60 files (10 tasks x 3 splits x 2 rewards, 120,960 trajectories) replay exactly through our reward machine
(`logic/ltlgrid_dfa.py`, `envs/ltlgrid_env.py`): every reward, every episode end, and `final_observation`.
Tests: `pytest tests/test_ltlgrid_*.py` (set `LTLGRID_DATA_DIR` to `<data_dir>` for the real-data tests).

## 5. 7x7 grid with layouts (Grid7 "fixed50")
Data: `datasets_fixed50.tar.gz` from Susanna (7x7 grid, 50 fixed object layouts with 2 copies of each object, 5,000
trajectories per file, plus `test_<split>_fixed50.npz` with 500 fixed test episodes). Unpack it so that you get
`<data_dir>/Dataset/<distance|three_value>/task<N>/dataset_<split>_fixed50.h5`. The files store each game's
7x7 layout and the agent positions, no observations.

- State token = `layout * 49 + x * 7 + y` (`envs/ltlgrid7_env.py`), so the 4x4 token/sequence code is reused.
- The DT reads the map: `datasets/ltlgrid7_dataset.py` gives every state token its symbolic map (5 planes of
  7x7: pickaxe, lava, door, gem, agent) and `models/dt_model.py` encodes it with a small MLP (`state_features`).
- The logic loss is layout aware (`planning/ltlgrid7_logic.py`): the belief is over (cell, DFA state), the grounder
  table of the game's layout gives the symbols. Grounders (oracle / supervised / weak) are those of 4x4.
- The test episodes are the 500 of the npz: `reset(seed)` picks episode `seed % 500`, so 500 rollouts visit each once.

Run: `python scripts/ltlgrid_train_all.py --grid 7 --data_dir <data_dir> --out_dir <out> --rewards distance
--tasks 1 --datasets medium_replay --logic_alphas 0 0.5 0.9 --seeds 0 1 2` (defaults for `--grid 7`: context 20,
500 test games). Her protocol (success rate and CDR over the 500 episodes, seeds 0-2) is in `LEGGIMI_fixed50.md`;
we report success and failure rate, and our budget is `--epochs` passes over all windows, not her 200 random windows.

Checks: every one of the 60 files replays exactly (moves, rewards, terminal flags; 300,000 games, 8.6M steps).
Tests: `LTLGRID7_DATA_DIR=<data_dir> pytest tests/test_ltlgrid7.py`.
