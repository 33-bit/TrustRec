"""Mirror a local folder of JSON Lines parts to a Hugging Face dataset repository."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

STATE_FILE = ".hf_sync_state.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _token(token_env: str) -> str:
    for name in (token_env, "HF_TOKEN"):
        value = os.environ.get(name)
        if value:
            return value
    raise RuntimeError(f"set {token_env} or HF_TOKEN to a Hugging Face write token")


@dataclass
class HFSync:
    """Upload changed files under ``local_dir`` to ``repo_id/prefix`` and download them back.

    The repository must be private unless ``allow_public`` is set, because the
    files hold review excerpts. Each push is one commit with only the files whose
    SHA-256 changed since the last push.
    """

    repo_id: str
    local_dir: Path
    prefix: str
    token_env: str = "HF_KEY"
    allow_public: bool = False
    api: Any = None

    def _api(self) -> Any:
        if self.api is None:
            try:
                from huggingface_hub import HfApi
            except ModuleNotFoundError as error:
                raise RuntimeError("install huggingface_hub to sync with Hugging Face") from error
            self.api = HfApi(token=_token(self.token_env))
        return self.api

    def check_visibility(self) -> bool:
        info = self._api().dataset_info(self.repo_id)
        private = bool(getattr(info, "private", False))
        if not private and not self.allow_public:
            raise RuntimeError(
                f"{self.repo_id} is public; make it private or pass allow_public to upload"
            )
        return private

    def _state(self) -> dict[str, str]:
        path = self.local_dir / STATE_FILE
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}

    def push(self) -> dict[str, Any]:
        from huggingface_hub import CommitOperationAdd

        self.check_visibility()
        state = self._state()
        changed: dict[str, str] = {}
        for path in sorted(self.local_dir.glob("*.jsonl")):
            digest = _sha256(path)
            if state.get(path.name) != digest:
                changed[path.name] = digest
        if not changed:
            return {"uploaded": [], "commit": None}
        operations = [
            CommitOperationAdd(
                path_in_repo=f"{self.prefix}/{name}", path_or_fileobj=str(self.local_dir / name)
            )
            for name in changed
        ]
        commit = self._api().create_commit(
            repo_id=self.repo_id,
            repo_type="dataset",
            operations=operations,
            commit_message=f"Sync {len(operations)} file(s) under {self.prefix}",
        )
        state.update(changed)
        (self.local_dir / STATE_FILE).write_text(json.dumps(state, indent=2), encoding="utf-8")
        return {"uploaded": sorted(changed), "commit": getattr(commit, "oid", None)}

    def pull(self, overwrite: bool = False) -> dict[str, Any]:
        """Download remote parts that are missing locally (or all with ``overwrite``)."""

        from huggingface_hub import hf_hub_download

        api = self._api()
        self.local_dir.mkdir(parents=True, exist_ok=True)
        remote = [
            name
            for name in api.list_repo_files(self.repo_id, repo_type="dataset")
            if name.startswith(f"{self.prefix}/") and name.endswith(".jsonl")
        ]
        downloaded = []
        state = self._state()
        for name in sorted(remote):
            target = self.local_dir / Path(name).name
            if target.exists() and not overwrite:
                continue
            cached = hf_hub_download(
                self.repo_id, name, repo_type="dataset", token=_token(self.token_env)
            )
            target.write_bytes(Path(cached).read_bytes())
            state[target.name] = _sha256(target)
            downloaded.append(target.name)
        (self.local_dir / STATE_FILE).write_text(json.dumps(state, indent=2), encoding="utf-8")
        return {"downloaded": downloaded, "remote_files": len(remote)}
