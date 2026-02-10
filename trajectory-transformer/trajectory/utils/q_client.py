import json
import socket
from typing import Optional

import numpy as np


class QClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 5555, timeout: float = 10.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self._sock: Optional[socket.socket] = None
        self._file = None

    def connect(self):
        if self._sock is not None:
            return
        self._sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        self._file = self._sock.makefile("rwb")

    def close(self):
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
        self._file = None
        self._sock = None

    def query(self, obs: np.ndarray, acts: np.ndarray) -> np.ndarray:
        self.connect()
        payload = {
            "obs": np.asarray(obs, dtype=np.float32).tolist(),
            "acts": np.asarray(acts, dtype=np.float32).tolist(),
        }
        self._file.write((json.dumps(payload) + "\n").encode("utf-8"))
        self._file.flush()
        line = self._file.readline()
        if not line:
            raise RuntimeError("Q server closed the connection")
        resp = json.loads(line.decode("utf-8"))
        if "error" in resp:
            raise RuntimeError(resp["error"])
        return np.asarray(resp["q"], dtype=np.float32)

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
