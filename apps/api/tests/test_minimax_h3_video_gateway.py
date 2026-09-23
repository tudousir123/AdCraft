"""MiniMax H3 gateway seam tests: transport HTTP shape, catalog entry, registry wiring."""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Callable, Mapping
from uuid import uuid4

import pytest

from app.core.config import Settings
from app.persistence.database import create_v2_database
from app.persistence.provider_model_repository import ProviderModelRepository
from app.persistence.schema import upgrade_v2_schema
from app.schemas.provider_models import ProviderAdapterProfileV1
from app.services.provider_adapter_registry import build_trusted_provider_adapter_registry
from app.services.provider_credentials import ProviderHttpResponse
from app.services.provider_model_bootstrap import ProviderModelBootstrapService
from app.services.provider_model_catalog import (
    ProviderModelCatalogService,
    _MINIMAX_H3_VIDEO_PROFILE,
)
from app.services.provider_native_adapters import (
    CanonicalProviderReference,
    CanonicalProviderRequest,
    MiniMaxGatewayHttpError,
    MiniMaxVideoAdapter,
    MiniMaxVideoTransport,
)
from app.services.v2_provider_executor import (
    _decode_native_result_value,
    _gateway_error_passthrough,
    _native_status_provider_error,
)
from app.services.v2_provider_reference_input_delivery import (
    CANVAS_PROTOCOL_REFERENCE_DELIVERY_MODES,
)

H3_MODEL_REF = "minimax:minimax/h3"
GATEWAY_BASE_URL = "https://gateway.example.com/v1"
SAMPLE_MP4 = b"\x00\x00\x00\x18ftypisom" + b"\x00" * 32
FIRST_FRAME_DATA_URL = "data:image/png;base64,aWRlbnRpY2Fs" + "A" * 64


def _gateway_settings() -> Settings:
    return Settings(
        minimax_api_key="test-minimax-key",
        minimax_base_url=GATEWAY_BASE_URL,
        media_data_dir=Path("data"),
    )


def _h3_profile() -> ProviderAdapterProfileV1:
    return ProviderAdapterProfileV1.model_validate(dict(_MINIMAX_H3_VIDEO_PROFILE))


def _h3_request(
    parameters: Mapping[str, object] | None = None,
    references: tuple[CanonicalProviderReference, ...] = (),
) -> CanonicalProviderRequest:
    return CanonicalProviderRequest(
        model_ref=H3_MODEL_REF,
        provider_model_id="minimax/h3",
        capability="video",
        prompt="A cinematic product shot.",
        parameters=(
            dict(parameters)
            if parameters is not None
            else {
                "duration_seconds": 4,
                "aspect_ratio": "16:9",
                "resolution": "720p",
                "generate_audio": True,
            }
        ),
        references=references,
    )


def _resolution() -> dict[str, str]:
    profile = _h3_profile()
    return {
        "model_ref": profile.model_ref,
        "adapter_id": profile.adapter_id,
        "transport_kind": profile.transport_kind,
        "adapter_revision": profile.adapter_revision,
        "capability_revision": profile.capability_revision,
    }


def _first_frame_reference() -> CanonicalProviderReference:
    return CanonicalProviderReference(
        reference_id="asset_1",
        role="storyboard",
        input_type="data_url",
        value=FIRST_FRAME_DATA_URL,
    )


class FakeGatewayHttp:
    """Injectable HTTP seam recording every gateway call."""

    def __init__(self) -> None:
        self.routes: dict[tuple[str, str], Callable[[dict[str, Any]], ProviderHttpResponse]] = {}
        self.calls: list[dict[str, Any]] = []

    def get(
        self,
        *,
        url: str,
        headers: dict[str, str],
        timeout_seconds: float,
        max_response_bytes: int,
    ) -> ProviderHttpResponse:
        call = {"method": "GET", "url": url, "headers": dict(headers)}
        self.calls.append(call)
        handler = self.routes.get(("GET", url))
        assert handler is not None, f"unexpected GET {url}"
        return handler(call)

    def post_json(
        self,
        *,
        url: str,
        headers: dict[str, str],
        payload: dict[str, object],
        timeout_seconds: float,
        max_response_bytes: int,
    ) -> ProviderHttpResponse:
        call = {
            "method": "POST",
            "url": url,
            "headers": dict(headers),
            "payload": dict(payload),
        }
        self.calls.append(call)
        handler = self.routes.get(("POST", url))
        assert handler is not None, f"unexpected POST {url}"
        return handler(call)

    def route_json(self, method: str, url: str, status_code: int, body: Mapping[str, object]) -> None:
        encoded = json.dumps(dict(body)).encode("utf-8")

        def _handler(call: dict[str, Any]) -> ProviderHttpResponse:
            return ProviderHttpResponse(status_code=status_code, body=encoded)

        self.routes[(method, url)] = _handler

    def route_bytes(self, method: str, url: str, status_code: int, body: bytes) -> None:
        def _handler(call: dict[str, Any]) -> ProviderHttpResponse:
            return ProviderHttpResponse(status_code=status_code, body=body)

        self.routes[(method, url)] = _handler

    def calls_for(self, method: str, url_suffix: str) -> list[dict[str, Any]]:
        return [
            call for call in self.calls if call["method"] == method and call["url"].endswith(url_suffix)
        ]


@dataclass
class AdapterHarness:
    adapter: MiniMaxVideoAdapter
    http: FakeGatewayHttp = field(repr=False)


def _h3_adapter() -> AdapterHarness:
    http = FakeGatewayHttp()
    adapter = MiniMaxVideoAdapter(
        _h3_profile(),
        transport=MiniMaxVideoTransport(_gateway_settings(), http_transport=http),
    )
    return AdapterHarness(adapter=adapter, http=http)


def _route_task_lifecycle(http: FakeGatewayHttp, *, task_id: str = "vid_123") -> None:
    http.route_json("POST", f"{GATEWAY_BASE_URL}/videos", 200, {"id": task_id, "status": "queued"})
    http.route_json(
        "GET",
        f"{GATEWAY_BASE_URL}/videos/{task_id}",
        200,
        {"id": task_id, "status": "completed"},
    )
    http.route_bytes("GET", f"{GATEWAY_BASE_URL}/videos/{task_id}/content", 200, SAMPLE_MP4)


class TestTransportRequestShape:
    def test_t2v_submit_sends_only_gateway_fields(self) -> None:
        harness = _h3_adapter()
        _route_task_lifecycle(harness.http)
        compiled = harness.adapter.compile(_h3_request(), _resolution())
        submission = harness.adapter.submit(compiled)

        assert submission.provider_task_id == "vid_123"
        post = harness.http.calls_for("POST", "/videos")[0]
        assert post["url"] == f"{GATEWAY_BASE_URL}/videos"
        assert post["headers"]["Authorization"] == "Bearer test-minimax-key"
        assert post["payload"] == {
            "model": "minimax/h3",
            "prompt": "A cinematic product shot.",
            "seconds": 4,
            "size": "1344x768",
        }

    def test_i2v_submit_carries_first_frame_data_url(self) -> None:
        harness = _h3_adapter()
        _route_task_lifecycle(harness.http)
        compiled = harness.adapter.compile(
            _h3_request(references=(_first_frame_reference(),)),
            _resolution(),
        )
        harness.adapter.submit(compiled)

        payload = harness.http.calls_for("POST", "/videos")[0]["payload"]
        assert payload == {
            "model": "minimax/h3",
            "prompt": "A cinematic product shot.",
            "seconds": 4,
            "size": "1344x768",
            "input_reference": FIRST_FRAME_DATA_URL,
        }

    @pytest.mark.parametrize(
        ("aspect_ratio", "size"),
        [
            ("21:9", "1536x672"),
            ("16:9", "1344x768"),
            ("4:3", "1024x768"),
            ("1:1", "768x768"),
            ("3:4", "768x1024"),
            ("9:16", "768x1344"),
        ],
    )
    def test_six_canvas_ratios_map_to_fixed_sizes(self, aspect_ratio: str, size: str) -> None:
        harness = _h3_adapter()
        _route_task_lifecycle(harness.http)
        compiled = harness.adapter.compile(
            _h3_request(parameters={"duration_seconds": 15, "aspect_ratio": aspect_ratio}),
            _resolution(),
        )
        harness.adapter.submit(compiled)

        payload = harness.http.calls_for("POST", "/videos")[0]["payload"]
        assert payload["size"] == size
        assert payload["seconds"] == 15

    def test_missing_duration_falls_back_to_default(self) -> None:
        harness = _h3_adapter()
        _route_task_lifecycle(harness.http)
        compiled = harness.adapter.compile(
            _h3_request(parameters={"aspect_ratio": "1:1"}),
            _resolution(),
        )
        harness.adapter.submit(compiled)

        payload = harness.http.calls_for("POST", "/videos")[0]["payload"]
        assert payload["seconds"] == 5
        assert payload["size"] == "768x768"

    def test_unsupported_ratio_is_rejected_before_submit(self) -> None:
        harness = _h3_adapter()
        with pytest.raises(ValueError, match="model_parameter_incompatible"):
            harness.adapter.compile(
                _h3_request(parameters={"aspect_ratio": "5:4"}),
                _resolution(),
            )
        assert harness.http.calls == []

    def test_second_reference_is_rejected(self) -> None:
        harness = _h3_adapter()
        request = _h3_request(references=(_first_frame_reference(), _first_frame_reference()))
        with pytest.raises(ValueError, match="reference_count_exceeded"):
            harness.adapter.compile(request, _resolution())


class TestTransportStatusAndDownload:
    def test_poll_lowercases_gateway_status(self) -> None:
        harness = _h3_adapter()
        http = harness.http
        http.route_json("POST", f"{GATEWAY_BASE_URL}/videos", 200, {"id": "vid_1", "status": "queued"})
        http.route_json(
            "GET",
            f"{GATEWAY_BASE_URL}/videos/vid_1",
            200,
            {"id": "vid_1", "status": "IN_PROGRESS"},
        )
        compiled = harness.adapter.compile(_h3_request(), _resolution())
        submission = harness.adapter.submit(compiled)
        status = harness.adapter.poll(submission)

        assert status.state == "in_progress"
        poll_call = http.calls_for("GET", "/videos/vid_1")[0]
        assert poll_call["url"] == f"{GATEWAY_BASE_URL}/videos/vid_1"
        assert poll_call["headers"]["Authorization"] == "Bearer test-minimax-key"

    def test_poll_failure_carries_gateway_error_code_and_message(self) -> None:
        harness = _h3_adapter()
        http = harness.http
        http.route_json("POST", f"{GATEWAY_BASE_URL}/videos", 200, {"id": "vid_2", "status": "queued"})
        http.route_json(
            "GET",
            f"{GATEWAY_BASE_URL}/videos/vid_2",
            200,
            {
                "id": "vid_2",
                "status": "FAILED",
                "error": {"code": "content_policy_violation", "message": "Prompt was rejected."},
            },
        )
        submission = harness.adapter.submit(harness.adapter.compile(_h3_request(), _resolution()))
        status = harness.adapter.poll(submission)

        assert status.state == "failed"
        assert status.raw["error_code"] == "content_policy_violation"
        assert status.raw["message"] == "Prompt was rejected."

    def test_download_inlines_authenticated_content_as_data_url(self) -> None:
        harness = _h3_adapter()
        _route_task_lifecycle(harness.http)
        submission = harness.adapter.submit(harness.adapter.compile(_h3_request(), _resolution()))
        status = harness.adapter.poll(submission)
        artifact = harness.adapter.download(status)
        result = harness.adapter.normalize(artifact)

        assert result.media_type == "video"
        assert result.value.startswith("data:video/mp4;base64,")
        decoded = base64.b64decode(result.value.split(";base64,", 1)[1])
        assert decoded == SAMPLE_MP4

        content_calls = harness.http.calls_for("GET", "/content")
        assert len(content_calls) == 1
        assert content_calls[0]["url"] == f"{GATEWAY_BASE_URL}/videos/vid_123/content"
        assert content_calls[0]["headers"]["Authorization"] == "Bearer test-minimax-key"
        # No unauthenticated content URL may leak into the artifact payload.
        assert "/content" not in result.value
        assert artifact.raw == {"mime_type": "video/mp4", "decoded_size": len(SAMPLE_MP4)}

        # The executor ingestion leg decodes the inlined data URL into asset bytes.
        assert _decode_native_result_value(result.value) == SAMPLE_MP4

    def test_download_rejects_non_mp4_payload(self) -> None:
        harness = _h3_adapter()
        http = harness.http
        http.route_json("POST", f"{GATEWAY_BASE_URL}/videos", 200, {"id": "vid_3", "status": "queued"})
        http.route_json(
            "GET", f"{GATEWAY_BASE_URL}/videos/vid_3", 200, {"id": "vid_3", "status": "completed"}
        )
        http.route_bytes("GET", f"{GATEWAY_BASE_URL}/videos/vid_3/content", 200, b"not-a-video")
        submission = harness.adapter.submit(harness.adapter.compile(_h3_request(), _resolution()))
        status = harness.adapter.poll(submission)

        with pytest.raises(ValueError, match="provider_response_contract_invalid"):
            harness.adapter.download(status)

    def test_download_rejects_oversized_content(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            "app.services.provider_native_adapters._MINIMAX_VIDEO_MAX_BYTES",
            1024,
        )
        harness = _h3_adapter()
        http = harness.http
        http.route_json("POST", f"{GATEWAY_BASE_URL}/videos", 200, {"id": "vid_4", "status": "queued"})
        http.route_json(
            "GET", f"{GATEWAY_BASE_URL}/videos/vid_4", 200, {"id": "vid_4", "status": "completed"}
        )
        http.route_bytes(
            "GET", f"{GATEWAY_BASE_URL}/videos/vid_4/content", 200, SAMPLE_MP4 + b"\x00" * 4096
        )
        submission = harness.adapter.submit(harness.adapter.compile(_h3_request(), _resolution()))
        status = harness.adapter.poll(submission)

        with pytest.raises(ValueError, match="provider_response_contract_invalid"):
            harness.adapter.download(status)

    def test_non_image_data_url_reference_is_rejected(self) -> None:
        reference = CanonicalProviderReference(
            reference_id="asset_2",
            role="storyboard",
            input_type="data_url",
            value="data:video/mp4;base64,AAAA",
        )
        harness = _h3_adapter()
        with pytest.raises(ValueError, match="provider_reference_input_invalid"):
            harness.adapter.compile(_h3_request(references=(reference,)), _resolution())


class TestTransportHttpErrors:
    def test_submit_http_error_passes_gateway_code_and_message(self) -> None:
        harness = _h3_adapter()
        harness.http.route_json(
            "POST",
            f"{GATEWAY_BASE_URL}/videos",
            400,
            {"error": {"code": "invalid_request_error", "message": "Unknown parameter: n."}},
        )
        compiled = harness.adapter.compile(_h3_request(), _resolution())

        with pytest.raises(MiniMaxGatewayHttpError) as excinfo:
            harness.adapter.submit(compiled)

        assert str(excinfo.value) == "provider_request_failed"
        assert excinfo.value.status_code == 400
        assert excinfo.value.gateway_code == "invalid_request_error"
        assert excinfo.value.gateway_message == "Unknown parameter: n."

    def test_poll_http_error_without_error_body_uses_status_code(self) -> None:
        harness = _h3_adapter()
        http = harness.http
        http.route_json("POST", f"{GATEWAY_BASE_URL}/videos", 200, {"id": "vid_5", "status": "queued"})
        http.route_json("GET", f"{GATEWAY_BASE_URL}/videos/vid_5", 401, {})
        submission = harness.adapter.submit(harness.adapter.compile(_h3_request(), _resolution()))

        with pytest.raises(MiniMaxGatewayHttpError) as excinfo:
            harness.adapter.poll(submission)

        assert excinfo.value.status_code == 401
        assert excinfo.value.gateway_code == "http_401"
        assert "401" in excinfo.value.gateway_message

    def test_missing_credentials_fail_fast(self) -> None:
        transport = MiniMaxVideoTransport(
            Settings(minimax_api_key=None, minimax_base_url=None),
        )
        with pytest.raises(ValueError, match="provider_configuration_missing"):
            transport.submit({"model": "minimax/h3", "prompt": "x", "seconds": 4, "size": "768x768"})

    def test_invalid_base_url_is_rejected(self) -> None:
        transport = MiniMaxVideoTransport(
            Settings(minimax_api_key="k", minimax_base_url="gateway.example.com/v1"),
        )
        with pytest.raises(ValueError, match="provider_base_url_invalid"):
            transport.poll("vid_1")

    def test_executor_surfaces_gateway_error_passthrough(self) -> None:
        error = MiniMaxGatewayHttpError(
            status_code=400,
            code="invalid_request_error",
            message="Unknown parameter: n.",
        )
        message, passthrough = _gateway_error_passthrough(error)

        assert message == "Provider gateway error invalid_request_error: Unknown parameter: n."
        assert passthrough == {"code": "invalid_request_error", "message": "Unknown parameter: n."}

    def test_executor_surfaces_poll_status_error(self) -> None:
        message, passthrough = _native_status_provider_error(
            {"error_code": "content_policy_violation", "message": "Prompt was rejected."}
        )

        assert message == "Provider gateway error content_policy_violation: Prompt was rejected."
        assert passthrough["code"] == "content_policy_violation"

    def test_executor_passthrough_ignores_plain_value_errors(self) -> None:
        assert _gateway_error_passthrough(ValueError("provider_request_failed")) == (None, {})


def _catalog_service(
    tmp_path: Path,
    *,
    credentials_ready: bool,
) -> tuple[ProviderModelCatalogService, ProviderModelRepository]:
    database = _new_database(tmp_path)
    repository = ProviderModelRepository(database)
    catalog = ProviderModelCatalogService(
        repository,
        capability_available=(lambda provider_id, capability: credentials_ready),
    )
    catalog.reconcile_trusted_models("minimax", now="2026-09-24T00:00:00+00:00")
    return catalog, repository


def _new_database(tmp_path: Path):
    data_dir = tmp_path / f"db-{uuid4().hex}"
    (data_dir / "v2").mkdir(parents=True)
    database = create_v2_database(data_dir)
    upgrade_v2_schema(database)
    return database


class TestCatalogSeam:
    def test_h3_is_listed_when_credentials_ready(self, tmp_path: Path) -> None:
        catalog, _ = _catalog_service(tmp_path, credentials_ready=True)

        models = catalog.list_models(capability="video")
        refs = {model.model_ref for model in models}

        assert H3_MODEL_REF in refs

    def test_h3_is_unselectable_without_credentials(self, tmp_path: Path) -> None:
        catalog, _ = _catalog_service(tmp_path, credentials_ready=False)

        listed = catalog.list_models(capability="video")
        assert H3_MODEL_REF not in {model.model_ref for model in listed}

        h3 = next(
            model
            for model in catalog.list_models(capability="video", include_unavailable=True)
            if model.model_ref == H3_MODEL_REF
        )
        assert h3.availability == "unavailable"
        assert h3.unavailable_reason == "provider_credentials_missing"

    def test_hailuo_entries_stay_unselectable_even_with_credentials(
        self,
        tmp_path: Path,
    ) -> None:
        catalog, _ = _catalog_service(tmp_path, credentials_ready=True)

        listed = catalog.list_models(capability="video")
        refs = {model.model_ref for model in listed}

        assert "minimax:MiniMax-Hailuo-2.3" not in refs
        assert "minimax:MiniMax-Hailuo-2.3-Fast" not in refs
        assert "minimax:MiniMax-Hailuo-02" not in refs
        assert H3_MODEL_REF in refs

    def test_h3_capability_metadata_and_conformance(self, tmp_path: Path) -> None:
        catalog, _ = _catalog_service(tmp_path, credentials_ready=True)

        h3 = next(
            model
            for model in catalog.list_models(capability="video", include_unavailable=True)
            if model.model_ref == H3_MODEL_REF
        )
        metadata = h3.capability_metadata

        assert h3.provider_model_id == "minimax/h3"
        assert h3.display_name == "MiniMax H3"
        assert metadata["provider_protocol"] == "minimax_video_generation"
        assert metadata["supported_aspect_ratios"] == ["21:9", "16:9", "4:3", "1:1", "3:4", "9:16"]
        assert metadata["duration_range_seconds"] == [4, 15]
        assert metadata["max_references"] == 1
        assert metadata["reference_limits"] == {"image": 1, "video": 0, "audio": 0}
        assert metadata["supports_provider_idempotency_token"] is False

        profile = ProviderAdapterProfileV1.model_validate(metadata["adapter_profile"])
        assert profile.conformance_status == "compatible"
        assert profile.transport_kind == "minimax_video_native"
        assert profile.reference_policy.max_images == 1
        assert profile.supports_provider_idempotency is False

        descriptors = {
            descriptor.name: descriptor
            for descriptor in profile.parameter_matrix.descriptors
        }
        assert descriptors["duration_seconds"].minimum == 4
        assert descriptors["duration_seconds"].maximum == 15
        assert descriptors["aspect_ratio"].allowed_values == (
            "21:9",
            "16:9",
            "4:3",
            "1:1",
            "3:4",
            "9:16",
        )

    def test_minimax_protocol_supports_first_frame_delivery(self) -> None:
        modes = CANVAS_PROTOCOL_REFERENCE_DELIVERY_MODES["minimax_video_generation"]

        assert "data_url" in modes
        assert "image_url" in modes
        assert "provider_file_id" not in modes


class TestRegistrySeam:
    def test_h3_resolves_to_adapter_with_real_transport(self, tmp_path: Path) -> None:
        catalog, _ = _catalog_service(tmp_path, credentials_ready=True)
        registry = build_trusted_provider_adapter_registry(
            catalog.list_models(include_unavailable=True),
            settings=_gateway_settings(),
        )

        resolved = registry.resolve(H3_MODEL_REF, "video")

        assert resolved.profile.model_ref == H3_MODEL_REF
        assert resolved.profile.conformance_status == "compatible"
        assert isinstance(resolved.adapter, MiniMaxVideoAdapter)
        assert isinstance(resolved.adapter._transport, MiniMaxVideoTransport)  # noqa: SLF001

    def test_hailuo_resolution_is_rejected(self, tmp_path: Path) -> None:
        catalog, _ = _catalog_service(tmp_path, credentials_ready=True)
        registry = build_trusted_provider_adapter_registry(
            catalog.list_models(include_unavailable=True),
            settings=_gateway_settings(),
        )

        with pytest.raises(ValueError, match="model_conformance_required"):
            registry.resolve("minimax:MiniMax-Hailuo-2.3", "video")

    def test_h3_without_settings_has_no_transport(self, tmp_path: Path) -> None:
        catalog, _ = _catalog_service(tmp_path, credentials_ready=True)
        registry = build_trusted_provider_adapter_registry(
            catalog.list_models(include_unavailable=True),
            settings=None,
        )

        resolved = registry.resolve(H3_MODEL_REF, "video")

        assert resolved.adapter._transport is None  # noqa: SLF001
        with pytest.raises(ValueError, match="provider_transport_unavailable"):
            resolved.adapter.submit(
                resolved.adapter.compile(_h3_request(), _resolution())
            )


def _bootstrap_defaults(tmp_path: Path, settings: Settings) -> dict[str, str]:
    database = _new_database(tmp_path)
    repository = ProviderModelRepository(database)
    ProviderModelBootstrapService(settings, repository).bootstrap(
        now="2026-09-24T00:00:00+00:00"
    )
    defaults = {
        key: record.model_ref for key, record in repository.get_defaults().items()
    }
    database.dispose()
    return defaults


class TestVideoDefaultPolicy:
    def test_h3_becomes_video_default_when_credentials_ready(self, tmp_path: Path) -> None:
        defaults = _bootstrap_defaults(
            tmp_path,
            Settings(
                media_mode="real",
                minimax_api_key="test-minimax-key",
                minimax_base_url=GATEWAY_BASE_URL,
            ),
        )

        assert defaults.get("video") == H3_MODEL_REF

    def test_ark_stays_video_default_without_minimax_credentials(
        self,
        tmp_path: Path,
    ) -> None:
        defaults = _bootstrap_defaults(
            tmp_path,
            Settings(media_mode="real", video_generation_api_key="ark-key"),
        )

        assert defaults.get("video") == "volcengine_ark:doubao-seedance-2-0-fast-260128"

    def test_minimax_credentials_win_over_ark(self, tmp_path: Path) -> None:
        defaults = _bootstrap_defaults(
            tmp_path,
            Settings(
                media_mode="real",
                video_generation_api_key="ark-key",
                minimax_api_key="test-minimax-key",
                minimax_base_url=GATEWAY_BASE_URL,
            ),
        )

        assert defaults.get("video") == H3_MODEL_REF

    def test_h3_default_falls_back_to_ark_when_credentials_removed(
        self,
        tmp_path: Path,
    ) -> None:
        database = _new_database(tmp_path)
        repository = ProviderModelRepository(database)
        ProviderModelBootstrapService(
            Settings(
                media_mode="real",
                minimax_api_key="test-minimax-key",
                minimax_base_url=GATEWAY_BASE_URL,
            ),
            repository,
        ).bootstrap(now="2026-09-24T00:00:00+00:00")
        assert (
            repository.get_defaults()["video"].model_ref == H3_MODEL_REF
        )

        ProviderModelBootstrapService(
            Settings(media_mode="real", video_generation_api_key="ark-key"),
            repository,
        ).bootstrap(now="2026-09-24T01:00:00+00:00")

        assert (
            repository.get_defaults()["video"].model_ref
            == "volcengine_ark:doubao-seedance-2-0-fast-260128"
        )
        database.dispose()
