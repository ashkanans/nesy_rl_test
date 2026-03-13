from pathlib import Path

from train_cb import _find_latest_checkpoint


def test_find_latest_checkpoint_empty_dir(tmp_path):
    epoch, path = _find_latest_checkpoint(str(tmp_path))
    assert epoch is None
    assert path is None


def test_find_latest_checkpoint_picks_highest_epoch(tmp_path):
    (tmp_path / "cb_state_0.pt").write_text("x")
    (tmp_path / "cb_state_12.pt").write_text("x")
    (tmp_path / "cb_state_3.pt").write_text("x")
    (tmp_path / "ignore.txt").write_text("x")
    (tmp_path / "cb_state_bad.pt").write_text("x")

    epoch, path = _find_latest_checkpoint(str(tmp_path))
    assert epoch == 12
    assert path == str(Path(tmp_path) / "cb_state_12.pt")
