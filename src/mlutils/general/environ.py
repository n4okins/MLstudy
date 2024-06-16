import os
from pathlib import Path


def load_env(root: str | Path) -> dict:
    env = dict()
    env_file = Path(root) / ".env"
    with env_file.open("r") as f:
        for line in f:
            key, value = line.strip().split("=")
            env[key] = value

    os.environ.update(env)
    return env