import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


def get_repo_name() -> str:
    return f"{HfApi().whoami()['name']}/paroquant-pair-selection"


def format_settings(
    **settings: object,
) -> str:
    # 닿지 않는 축은 None으로 받아 빼므로, 경로에 돌리지 않은 설정이 찍히지 않음.
    return ",".join(f"{name}={value:g}" if isinstance(value, float) else f"{name}={value}" for name, value in settings.items() if value is not None)


def make_rotation_path(
    model_name: str,
    layer: int,
    linear_name: str,
    num_bits: int,
    group_size: int,
    num_samples: int,
    sequence_len: int,
    alpha_mode: str | None,
    schedule: str | None,
    num_rotations: int | None,
    num_steps: int,
    learning_rate: float,
) -> str:
    settings = format_settings(layer=layer, linear_name=linear_name, num_bits=num_bits, group_size=group_size, num_samples=num_samples, sequence_len=sequence_len, alpha_mode=alpha_mode, schedule=schedule, num_rotations=num_rotations, num_steps=num_steps, learning_rate=learning_rate)
    return f"rotation/{model_name}/{settings}"


def make_connectivity_path(
    model_name: str,
    layer: int,
    linear_name: str,
    group_size: int,
    num_samples: int,
    sequence_len: int,
    num_rotations: int,
) -> str:
    settings = format_settings(layer=layer, linear_name=linear_name, group_size=group_size, num_samples=num_samples, sequence_len=sequence_len, num_rotations=num_rotations)
    return f"connectivity/{model_name}/{settings}"


def make_pair_score_path(
    model_name: str,
    layer: int,
    linear_name: str,
    num_bits: int,
    group_size: int,
    num_samples: int,
    sequence_len: int,
    num_rotations: int,
    num_steps: int,
    learning_rate: float,
) -> str:
    settings = format_settings(layer=layer, linear_name=linear_name, num_bits=num_bits, group_size=group_size, num_samples=num_samples, sequence_len=sequence_len, num_rotations=num_rotations, num_steps=num_steps, learning_rate=learning_rate)
    return f"pair_score/{model_name}/{settings}"


def load_rows(
    path: str,
) -> list[dict]:
    return json.loads(Path(hf_hub_download(get_repo_name(), f"{path}.json")).read_text())


def get_pending_jobs(
    entry_name: str,
    jobs: list[tuple[str, str, int]],
) -> list[tuple[str, str, int]]:
    # 설정 하나가 파일 하나이고, 그 안의 행이 arm과 seed마다 하나씩이라 이미 있는 행만 뺌.
    files = set(HfApi().list_repo_files(get_repo_name()))
    done = set()
    for path in dict.fromkeys(path for path, _, _ in jobs):
        if f"{path}.json" in files:
            done |= {(path, row["arm"], row["seed"]) for row in load_rows(path)}
    pending = [job for job in jobs if job not in done]
    print(f"{entry_name} pending {len(pending)}/{len(jobs)}", *[f"{path} arm={arm} seed={seed}" for path, arm, seed in pending], sep="\n  ")
    return pending


def save_rows(
    path: str,
    rows: list[dict],
) -> None:
    # 같은 설정의 기존 행과 합쳐 파일 하나를 한 커밋으로 올림.
    api = HfApi()
    repo_name = get_repo_name()
    api.create_repo(repo_name, exist_ok=True)
    merged = {(row["arm"], row["seed"]): row for row in load_rows(path)} if api.file_exists(repo_name, f"{path}.json") else {}
    merged |= {(row["arm"], row["seed"]): row for row in rows}
    api.upload_file(path_or_fileobj=json.dumps(sorted(merged.values(), key=lambda row: (row["arm"], row["seed"])), indent=2).encode(), path_in_repo=f"{path}.json", repo_id=repo_name, commit_message=f"Save {path}.json")
    print(f"saved {path}.json")
