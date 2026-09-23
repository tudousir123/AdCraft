"""Catalog and default-policy coverage for the gateway GPT Image 2.5 Flare entry."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pytest

from app.core.config import Settings
from app.persistence.database import create_v2_database
from app.persistence.provider_model_repository import ProviderModelRepository
from app.persistence.schema import upgrade_v2_schema
from app.schemas.provider_models import ProviderAdapterProfileV1
from app.services.provider_model_bootstrap import ProviderModelBootstrapService
from app.services.provider_model_catalog import (
    ProviderModelCatalogService,
    trusted_image_size_enumeration,
)

FLARE_MODEL_REF = "volcengine_ark:gpt-image-2.5-flare"
FLARE_SIZE_TABLE = {
    "1:1": "1024x1024",
    "16:9": "1536x1024",
    "9:16": "1024x1536",
}


@pytest.fixture
def repository(v2_media_data_dir: Path) -> Iterator[ProviderModelRepository]:
    database = create_v2_database(v2_media_data_dir)
    try:
        upgrade_v2_schema(database)
        yield ProviderModelRepository(database)
    finally:
        database.dispose()


def _catalog(
    repository: ProviderModelRepository,
    *,
    image_credentials_configured: bool,
) -> ProviderModelCatalogService:
    def capability_available(provider_id: str, capability: str) -> bool:
        return (
            image_credentials_configured
            and provider_id == "volcengine_ark"
            and capability == "image"
        )

    return ProviderModelCatalogService(repository, capability_available=capability_available)


def test_flare_entry_is_not_selectable_without_image_credentials(
    repository: ProviderModelRepository,
) -> None:
    catalog = _catalog(repository, image_credentials_configured=False)
    catalog.sync("volcengine_ark", now="2026-09-24T00:00:00+00:00")

    model = catalog.get_model(FLARE_MODEL_REF)
    assert model.availability == "unavailable"
    assert model.unavailable_reason == "provider_credentials_missing"
    selectable = {item.model_ref for item in catalog.list_models(capability="image")}
    assert FLARE_MODEL_REF not in selectable


def test_flare_entry_is_selectable_with_image_credentials(
    repository: ProviderModelRepository,
) -> None:
    catalog = _catalog(repository, image_credentials_configured=True)
    catalog.sync("volcengine_ark", now="2026-09-24T00:00:00+00:00")

    selectable = {item.model_ref for item in catalog.list_models(capability="image")}
    assert FLARE_MODEL_REF in selectable

    metadata = catalog.get_model(FLARE_MODEL_REF).capability_metadata
    assert metadata["supported_aspect_ratios"] == ["1:1", "16:9", "9:16"]
    assert metadata["supported_sizes_by_aspect_ratio"] == FLARE_SIZE_TABLE
    assert metadata["pixel_bounds"] == [1024, 1536]
    assert metadata["reference_limits"] == {"image": 4, "video": 0, "audio": 0}
    assert metadata["max_references"] == 4

    profile = ProviderAdapterProfileV1.model_validate(metadata["adapter_profile"])
    assert profile.transport_kind == "ark_image_native"
    assert profile.conformance_status == "compatible"


def test_flare_declares_closed_size_table_while_seedream_stays_open() -> None:
    assert trusted_image_size_enumeration("gpt-image-2.5-flare") == FLARE_SIZE_TABLE
    assert trusted_image_size_enumeration("doubao-seedream-5-0-lite-260128") is None
    assert trusted_image_size_enumeration("not-a-trusted-model") is None


def _bootstrap_settings(
    v2_media_data_dir: Path,
    *,
    image_model: str,
    image_credentials: bool,
) -> Settings:
    return Settings(
        media_data_dir=v2_media_data_dir,
        media_mode="real",
        image_generation_model=image_model,
        image_generation_api_key="gateway-key" if image_credentials else None,
        image_generation_endpoint=(
            "https://gateway.example.com/v1/images/generations"
            if image_credentials
            else None
        ),
    )


def test_image_default_follows_configured_flare_model(
    repository: ProviderModelRepository,
    v2_media_data_dir: Path,
) -> None:
    settings = _bootstrap_settings(
        v2_media_data_dir,
        image_model="gpt-image-2.5-flare",
        image_credentials=True,
    )

    ProviderModelBootstrapService(settings, repository).bootstrap(
        now="2026-09-24T00:00:00+00:00"
    )

    assert repository.get_defaults()["image"].model_ref == FLARE_MODEL_REF


def test_image_default_falls_back_when_model_name_is_unrecognized(
    repository: ProviderModelRepository,
    v2_media_data_dir: Path,
) -> None:
    settings = _bootstrap_settings(
        v2_media_data_dir,
        image_model="not-a-trusted-model",
        image_credentials=True,
    )

    ProviderModelBootstrapService(settings, repository).bootstrap(
        now="2026-09-24T00:00:00+00:00"
    )

    assert (
        repository.get_defaults()["image"].model_ref
        == "volcengine_ark:doubao-seedream-5-0-lite-260128"
    )


def test_image_default_ignores_models_outside_the_image_credential_group(
    repository: ProviderModelRepository,
    v2_media_data_dir: Path,
) -> None:
    settings = _bootstrap_settings(
        v2_media_data_dir,
        image_model="openai/gpt-image-2",
        image_credentials=True,
    )

    ProviderModelBootstrapService(settings, repository).bootstrap(
        now="2026-09-24T00:00:00+00:00"
    )

    assert (
        repository.get_defaults()["image"].model_ref
        == "volcengine_ark:doubao-seedream-5-0-lite-260128"
    )


def test_image_default_keeps_existing_default_when_flare_is_unavailable(
    repository: ProviderModelRepository,
    v2_media_data_dir: Path,
) -> None:
    settings = _bootstrap_settings(
        v2_media_data_dir,
        image_model="gpt-image-2.5-flare",
        image_credentials=False,
    )

    ProviderModelBootstrapService(settings, repository).bootstrap(
        now="2026-09-24T00:00:00+00:00"
    )

    assert "image" not in repository.get_defaults()
