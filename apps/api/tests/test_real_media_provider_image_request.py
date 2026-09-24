"""Image request-construction coverage: WAF User-Agent, sizes, and response shapes."""

from __future__ import annotations

import base64
import json
from dataclasses import replace
from pathlib import Path
from typing import Any
from urllib import request as urllib_request

import pytest

from app.core.config import Settings
from app.tools.media_provider_protocol import (
    PROVIDER_HTTP_USER_AGENT,
    MediaConfigurationError,
)
from app.tools.real_media_provider import RealMediaProvider

FLARE_MODEL_ID = "gpt-image-2.5-flare"
PNG_BYTES = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + b"\x00" * 24


class _FakeResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self, *_args: object) -> bytes:
        return self._body

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *_args: object) -> bool:
        return False


class _RecordedTransport:
    def __init__(self, responses: list[bytes]) -> None:
        self._responses = list(responses)
        self.requests: list[urllib_request.Request] = []

    def __call__(self, request: urllib_request.Request, **_kwargs: object) -> _FakeResponse:
        self.requests.append(request)
        return _FakeResponse(self._responses.pop(0))

    @property
    def submit_request(self) -> urllib_request.Request:
        return self.requests[0]

    def submitted_payload(self) -> dict[str, Any]:
        return json.loads(self.submit_request.data.decode("utf-8"))


@pytest.fixture
def media_data_dir(tmp_path: Path) -> Path:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir


def _settings(media_data_dir: Path, *, image_model: str = FLARE_MODEL_ID) -> Settings:
    return Settings(
        media_data_dir=media_data_dir,
        media_mode="real",
        skip_audio_agents=True,
        image_generation_api_key="gateway-key",
        image_generation_endpoint="https://gateway.example.com/v1/images/generations",
        image_generation_model=image_model,
        video_generation_api_key="video-key",
        video_generation_endpoint="https://gateway.example.com/v1/contents/generations/tasks",
    )


def _flare_request(**overrides: Any) -> dict[str, Any]:
    request: dict[str, Any] = {
        "prompt": "A minimal product shot on a clean backdrop.",
        "slot_type": "image",
        "slot_id": "image-node-1",
        "semantic_type": "image",
        "provider_id": "volcengine_ark",
        "provider_model_id": FLARE_MODEL_ID,
    }
    request.update(overrides)
    return request


def test_flare_url_response_downloads_artifact_with_gateway_user_agent(
    monkeypatch: pytest.MonkeyPatch,
    media_data_dir: Path,
) -> None:
    artifact = b"https-only-artifact"
    transport = _RecordedTransport(
        [
            json.dumps({"data": [{"url": "https://cdn.example.com/artifact.png"}]}).encode(),
            artifact,
        ]
    )
    monkeypatch.setattr(urllib_request, "urlopen", transport)

    output = RealMediaProvider(_settings(media_data_dir)).generate_v2_canonical_image(
        _flare_request(aspect_ratio="16:9"),
        "wf-flare-url",
    )

    assert len(transport.requests) == 2
    submit = transport.submit_request
    assert submit.full_url == "https://gateway.example.com/v1/images/generations"
    assert submit.get_header("User-agent") == PROVIDER_HTTP_USER_AGENT
    assert not PROVIDER_HTTP_USER_AGENT.lower().startswith("python-urllib")
    assert submit.get_header("Authorization") == "Bearer gateway-key"
    assert submit.get_header("Content-type") == "application/json"

    download = transport.requests[1]
    assert download.full_url == "https://cdn.example.com/artifact.png"
    assert download.get_header("User-agent") == PROVIDER_HTTP_USER_AGENT

    payload = transport.submitted_payload()
    assert payload["model"] == FLARE_MODEL_ID
    assert payload["size"] == "1536x1024"

    asset = output["assets"][0]
    assert asset["model"] == FLARE_MODEL_ID
    assert asset["remote_url"] == "https://cdn.example.com/artifact.png"
    assert asset["download_status"] == "downloaded"
    persisted = media_data_dir / str(asset["local_path"])
    assert persisted.read_bytes() == artifact


def test_flare_reference_image_rides_existing_image_field_as_data_url(
    monkeypatch: pytest.MonkeyPatch,
    media_data_dir: Path,
) -> None:
    transport = _RecordedTransport(
        [
            json.dumps({"data": [{"url": "https://cdn.example.com/ref-result.png"}]}).encode(),
            b"reference-artifact",
        ]
    )
    monkeypatch.setattr(urllib_request, "urlopen", transport)
    data_url = "data:image/png;base64," + base64.b64encode(PNG_BYTES).decode("utf-8")

    RealMediaProvider(_settings(media_data_dir)).generate_v2_canonical_image(
        _flare_request(
            aspect_ratio="1:1",
            reference_assets=[{"asset_id": "ref-1", "provider_input_value": data_url}],
            submitted_reference_asset_ids=["ref-1"],
        ),
        "wf-flare-ref",
    )

    payload = transport.submitted_payload()
    assert payload["image"] == data_url
    assert payload["size"] == "1024x1024"


def test_flare_b64_response_is_decoded_and_persisted(
    monkeypatch: pytest.MonkeyPatch,
    media_data_dir: Path,
) -> None:
    encoded = base64.b64encode(b"b64-decoded-artifact").decode("utf-8")
    transport = _RecordedTransport(
        [json.dumps({"data": [{"b64_json": encoded}]}).encode()],
    )
    monkeypatch.setattr(urllib_request, "urlopen", transport)

    output = RealMediaProvider(_settings(media_data_dir)).generate_v2_canonical_image(
        _flare_request(aspect_ratio="9:16"),
        "wf-flare-b64",
    )

    assert len(transport.requests) == 1
    payload = transport.submitted_payload()
    assert payload["size"] == "1024x1536"

    asset = output["assets"][0]
    assert asset["remote_url"] is None
    assert asset["download_status"] == "decoded_base64"
    persisted = media_data_dir / str(asset["local_path"])
    assert persisted.read_bytes() == b"b64-decoded-artifact"


def test_flare_clamps_stale_size_onto_enumerated_table(
    monkeypatch: pytest.MonkeyPatch,
    media_data_dir: Path,
) -> None:
    transport = _RecordedTransport(
        [
            json.dumps({"data": [{"url": "https://cdn.example.com/clamped.png"}]}).encode(),
            b"clamped-artifact",
        ]
    )
    monkeypatch.setattr(urllib_request, "urlopen", transport)

    RealMediaProvider(_settings(media_data_dir)).generate_v2_canonical_image(
        _flare_request(size="2048x2048", aspect_ratio="9:16"),
        "wf-flare-clamp",
    )

    assert transport.submitted_payload()["size"] == "1024x1536"


def test_non_enumerated_model_keeps_seedream_size_normalization(
    monkeypatch: pytest.MonkeyPatch,
    media_data_dir: Path,
) -> None:
    transport = _RecordedTransport(
        [
            json.dumps({"data": [{"url": "https://cdn.example.com/seedream.png"}]}).encode(),
            b"seedream-artifact",
        ]
    )
    monkeypatch.setattr(urllib_request, "urlopen", transport)

    RealMediaProvider(
        _settings(media_data_dir, image_model="doubao-seedream-5-0-lite-260128")
    ).generate_v2_canonical_image(
        _flare_request(
            provider_model_id=None,
            size="2048x2048",
        ),
        "wf-seedream",
    )

    payload = transport.submitted_payload()
    assert payload["model"] == "doubao-seedream-5-0-lite-260128"
    assert payload["size"] == "2048x2048"


def test_enumerated_image_model_bypasses_seedream_size_floor_at_construction(
    media_data_dir: Path,
) -> None:
    flare = replace(_settings(media_data_dir), image_generation_size="1024x1024")
    RealMediaProvider(flare)

    seedream = replace(
        _settings(media_data_dir, image_model="doubao-seedream-5-0-lite-260128"),
        image_generation_size="1024x1024",
    )
    with pytest.raises(MediaConfigurationError):
        RealMediaProvider(seedream)
