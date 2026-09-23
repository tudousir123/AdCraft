"""Startup seeding for trusted provider catalog entries and installation defaults."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from threading import Lock

from app.core.config import PROJECT_ROOT, Settings
from app.persistence.provider_model_repository import (
    ModelDefaultRecord,
    ProviderModelRepository,
)
from app.services.provider_credentials import (
    DotenvCredentialStore,
    ProviderConnectionService,
    ProviderCredentialRegistry,
)
from app.services.provider_model_catalog import (
    ProviderModelCatalogService,
    trusted_manifest_for,
)


_BOOTSTRAP_LOCK = Lock()
_ARK_MINI_TEXT_MODEL_REF = "volcengine_ark:doubao-seed-2-0-mini-260428"
_ARK_PRO_TEXT_MODEL_REF = "volcengine_ark:doubao-seed-2-1-pro-260628"
_FALLBACK_IMAGE_MODEL_REF = "volcengine_ark:doubao-seedream-5-0-lite-260128"
_ARK_VIDEO_MODEL_REF = "volcengine_ark:doubao-seedance-2-0-fast-260128"
_MINIMAX_H3_VIDEO_MODEL_REF = "minimax:minimax/h3"


@dataclass(frozen=True)
class ProviderModelBootstrapResult:
    seeded_providers: tuple[str, ...]
    seeded_defaults: tuple[str, ...]


class ProviderModelBootstrapService:
    """Seed missing model policy without replacing existing catalog or defaults."""

    def __init__(self, settings: Settings, repository: ProviderModelRepository) -> None:
        self._settings = settings
        self._repository = repository

    def bootstrap(self, *, now: str) -> ProviderModelBootstrapResult:
        with _BOOTSTRAP_LOCK:
            return self._bootstrap(now=now)

    def _bootstrap(self, *, now: str) -> ProviderModelBootstrapResult:
        catalog = ProviderModelCatalogService(self._repository)
        catalog.ensure_no_retired_defaults()
        registry = ProviderCredentialRegistry()
        connection_service = ProviderConnectionService(
            registry=registry,
            dotenv_store=DotenvCredentialStore(
                PROJECT_ROOT,
                allowed_fields={
                    field
                    for provider_id in registry.provider_ids
                    for binding in registry.get(provider_id).bindings.values()
                    for field in (
                        binding.dotenv_field,
                        binding.endpoint_dotenv_field,
                    )
                    if field is not None
                },
            ),
            metadata_repository=self._repository,
            settings_loader=lambda: self._settings,
        )
        connection_service.synchronize_metadata(updated_at=now)
        catalog.reconcile_retired_models(now=now)
        seeded_providers: list[str] = []
        for provider_id in (
            "siliconflow",
            "volcengine_ark",
            "tianpuyue",
            "openrouter",
            "minimax",
            "fake",
        ):
            had_models = bool(self._repository.list_models(provider_id=provider_id))
            catalog.reconcile_trusted_models(provider_id, now=now)
            if not had_models:
                seeded_providers.append(provider_id)

        existing = catalog.get_default_records()
        candidates = {
            key: model_ref
            for key, model_ref in self._recognized_defaults().items()
            if key not in existing
        }
        valid_candidates: dict[str, str] = {}
        for key, model_ref in candidates.items():
            if self._model_is_available(model_ref):
                valid_candidates[key] = model_ref
        if (
            "video" not in existing
            and "video" not in valid_candidates
            and self._model_is_available(_ARK_VIDEO_MODEL_REF)
        ):
            valid_candidates["video"] = _ARK_VIDEO_MODEL_REF
        migrated_defaults: dict[str, str] = {}
        try:
            ark_pro = catalog.get_model(_ARK_PRO_TEXT_MODEL_REF)
        except ValueError:
            ark_pro = None
        if ark_pro is not None and ark_pro.availability == "available":
            migrated_defaults = {
                key: _ARK_PRO_TEXT_MODEL_REF
                for key in ("agent", "text")
                if existing.get(key) is not None
                and existing[key].model_ref == _ARK_MINI_TEXT_MODEL_REF
            }
        default_updates = {
            **migrated_defaults,
            **valid_candidates,
            **self._video_default_updates(existing),
        }
        if default_updates:
            catalog.set_defaults(default_updates, now=now)
        return ProviderModelBootstrapResult(
            seeded_providers=tuple(seeded_providers),
            seeded_defaults=tuple(valid_candidates),
        )

    def _recognized_defaults(self) -> dict[str, str]:
        text_ref = "fake:deterministic-text"
        if self._settings.agent_runtime_mode != "fake":
            text_ref = (
                "siliconflow:zai-org/GLM-5.2"
                if self._settings.siliconflow_api_key
                else _ARK_PRO_TEXT_MODEL_REF
            )
        if self._settings.media_mode == "mock":
            return {
                "agent": text_ref,
                "text": text_ref,
                "image": "fake:deterministic-image",
                "video": "fake:deterministic-video",
                "audio": "fake:deterministic-audio",
            }
        return {
            "agent": text_ref,
            "text": text_ref,
            "image": self._image_default_model_ref(),
            "video": self._video_default_model_ref(),
            "audio": "tianpuyue:TemPolor-i3",
        }

    def _image_default_model_ref(self) -> str:
        """Follow the configured image model to its trusted entry, with a fallback.

        Only entries served by the image credential group's provider qualify, so
        an IMAGE_GENERATION_MODEL naming another provider's model (or any
        unrecognized name) keeps the historical default.
        """

        configured = self._settings.image_generation_model.strip()
        manifest = trusted_manifest_for("image", configured)
        if manifest is not None and manifest.provider_id == "volcengine_ark":
            return manifest.model_ref
        return _FALLBACK_IMAGE_MODEL_REF

    def _video_default_model_ref(self) -> str:
        """Prefer MiniMax H3 once its gateway credentials are configured."""

        if self._minimax_video_credentials_ready():
            return _MINIMAX_H3_VIDEO_MODEL_REF
        return _ARK_VIDEO_MODEL_REF

    def _minimax_video_credentials_ready(self) -> bool:
        return bool(
            (self._settings.minimax_api_key or "").strip()
            and (self._settings.minimax_base_url or "").strip()
        )

    def _video_default_updates(self, existing: Mapping[str, ModelDefaultRecord]) -> dict[str, str]:
        """Keep the video default on H3 while it stays available; fall back otherwise."""

        current = existing.get("video")
        if current is None:
            return {}
        h3_available = self._model_is_available(_MINIMAX_H3_VIDEO_MODEL_REF)
        if current.model_ref == _MINIMAX_H3_VIDEO_MODEL_REF:
            if h3_available or not self._model_is_available(_ARK_VIDEO_MODEL_REF):
                return {}
            return {"video": _ARK_VIDEO_MODEL_REF}
        if current.model_ref == _ARK_VIDEO_MODEL_REF and h3_available:
            return {"video": _MINIMAX_H3_VIDEO_MODEL_REF}
        return {}

    def _model_is_available(self, model_ref: str) -> bool:
        try:
            model = self._repository.get_model(model_ref)
        except ValueError:
            return False
        return model.availability == "available"
