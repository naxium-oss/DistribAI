"""
HuggingFace Hub Integration for DistribAI
Provides seamless model and dataset loading from HuggingFace Hub,
with automatic caching and local fallback.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

try:
    from huggingface_hub import (
        HfApi,
        hf_hub_download,
        snapshot_download,
    )
    from huggingface_hub import (
        login as hf_login,
    )

    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

logger = logging.getLogger(__name__)


class HuggingFaceIntegration:
    def __init__(self, cache_dir: str | None = None, token: str | None = None):
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/huggingface")
        self.token = token or os.getenv("HF_TOKEN")
        self._authenticated = False
        if not HF_AVAILABLE:
            logger.warning("huggingface_hub not installed. HF integration disabled.")
            return
        if self.token:
            try:
                hf_login(token=self.token)
                self._authenticated = True
                logger.info("Authenticated with HuggingFace Hub")
            except Exception as e:
                logger.warning(f"HF authentication failed: {e}")

    def is_available(self) -> bool:
        return HF_AVAILABLE

    def download_model(
        self,
        model_id: str,
        filename: str | None = None,
        revision: str | None = None,
        local_files_only: bool = False,
    ) -> str:
        """
        Download a model from HuggingFace Hub.
        Args:
            model_id: HuggingFace model ID (e.g., 'bert-base-uncased')
            filename: Specific file to download (None for entire repo)
            revision: Git revision to download
            local_files_only: Only use cached files
        Returns:
            Local path to downloaded model
        """
        if not HF_AVAILABLE:
            raise RuntimeError("huggingface_hub not installed")
        if filename:
            local_path = hf_hub_download(
                repo_id=model_id,
                filename=filename,
                revision=revision,
                cache_dir=self.cache_dir,
                local_files_only=local_files_only,
                token=self.token,
            )
            return local_path
        else:
            local_dir = snapshot_download(
                repo_id=model_id,
                revision=revision,
                cache_dir=self.cache_dir,
                local_files_only=local_files_only,
                token=self.token,
            )
            return local_dir

    def download_dataset(
        self,
        dataset_id: str,
        revision: str | None = None,
    ) -> str:
        """
        Download a dataset from HuggingFace Hub.
        Args:
            dataset_id: HuggingFace dataset ID
            revision: Git revision to download
        Returns:
            Local path to downloaded dataset
        """
        if not HF_AVAILABLE:
            raise RuntimeError("huggingface_hub not installed")
        local_dir = snapshot_download(
            repo_id=dataset_id,
            repo_type="dataset",
            revision=revision,
            cache_dir=self.cache_dir,
            token=self.token,
        )
        return local_dir

    def upload_checkpoint(
        self,
        local_path: str,
        repo_id: str,
        path_in_repo: str = "",
        commit_message: str = "Upload checkpoint",
    ) -> str:
        """
        Upload a checkpoint to HuggingFace Hub.
        Args:
            local_path: Local file or directory to upload
            repo_id: Target repository ID
            path_in_repo: Path within repository
            commit_message: Git commit message
        Returns:
            URL of uploaded file
        """
        if not HF_AVAILABLE:
            raise RuntimeError("huggingface_hub not installed")
        if not self._authenticated:
            raise RuntimeError("HF authentication required for upload")
        api = HfApi()
        if os.path.isfile(local_path):
            url = api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=path_in_repo or os.path.basename(local_path),
                repo_id=repo_id,
                commit_message=commit_message,
            )
        else:
            url = api.upload_folder(
                folder_path=local_path,
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                commit_message=commit_message,
            )
        logger.info(f"Uploaded to HuggingFace Hub: {url}")
        return url

    def create_repo(
        self,
        repo_id: str,
        repo_type: str = "model",
        private: bool = False,
    ) -> str:
        if not HF_AVAILABLE:
            raise RuntimeError("huggingface_hub not installed")
        if not self._authenticated:
            raise RuntimeError("HF authentication required")
        api = HfApi()
        repo_url = api.create_repo(
            repo_id=repo_id,
            repo_type=repo_type,
            private=private,
            exist_ok=True,
        )
        return repo_url


def get_model_from_hf(
    model_name: str,
    cache_dir: str | None = None,
    trust_remote_code: bool = True,
    allow_external: bool | None = None,
) -> Any:
    """
    Load a model from HuggingFace Hub using transformers.

    Supports arbitrary custom architectures via ``trust_remote_code`` (gated by
    ``DISTRIBAI_ALLOW_EXTERNAL_ARCH`` unless ``allow_external`` is set).
    """
    from worker.src.compute.external_arch import load_external_architecture

    try:
        return load_external_architecture(
            model_name,
            trust_remote_code=trust_remote_code,
            cache_dir=cache_dir,
            allow=allow_external,
        )
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        raise

def get_tokenizer_from_hf(
    tokenizer_name: str,
    cache_dir: str | None = None,
):
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            cache_dir=cache_dir,
        )
        return tokenizer
    except ImportError:
        raise RuntimeError("transformers library not installed")


class ModelCache:
    def __init__(self, cache_dir: str | None = None):
        self.cache_dir = Path(cache_dir or os.path.expanduser("~/.cache/distribai/models"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_cached_path(self, model_id: str) -> Path | None:
        model_path = self.cache_dir / model_id.replace("/", "--")
        if model_path.exists():
            return model_path
        return None

    def add_to_cache(self, model_id: str, local_path: str) -> Path:
        target = self.cache_dir / model_id.replace("/", "--")
        target.parent.mkdir(parents=True, exist_ok=True)
        import shutil

        if os.path.isdir(local_path):
            shutil.copytree(local_path, target, dirs_exist_ok=True)
        else:
            shutil.copy2(local_path, target)
        return target

    def clear_cache(self):
        import shutil

        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_cache_size(self) -> int:
        total = 0
        for path in self.cache_dir.rglob("*"):
            if path.is_file():
                total += path.stat().st_size
        return total
