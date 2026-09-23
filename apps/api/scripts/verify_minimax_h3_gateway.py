# ruff: noqa: E402
"""Drive the MiniMax H3 adapter→transport chain against a real gateway.

Submits one text-to-video and one first-frame image-to-video 4-second job,
polls until completion, downloads the authorized content, verifies MP4 bytes,
and writes both clips into --output-dir.  Each run spends real gateway quota
(about two 4-second videos); only run this with authorized credentials.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import struct
import sys
import time
import zlib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import Settings
from app.schemas.provider_models import ProviderAdapterProfileV1
from app.services.provider_model_catalog import _MINIMAX_H3_VIDEO_PROFILE
from app.services.provider_native_adapters import (
    CanonicalProviderReference,
    CanonicalProviderRequest,
    MiniMaxGatewayHttpError,
    MiniMaxVideoAdapter,
    MiniMaxVideoTransport,
)

MINIMAX_H3_MODEL_REF = "minimax:minimax/h3"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-key", default=None, help="Defaults to MINIMAX_API_KEY.")
    parser.add_argument("--base-url", default=None, help="Defaults to MINIMAX_BASE_URL.")
    parser.add_argument("--aspect-ratio", default="16:9")
    parser.add_argument("--output-dir", type=Path, default=Path("runtime-data/minimax-h3-verify"))
    parser.add_argument("--poll-interval-seconds", type=float, default=5.0)
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    return parser.parse_args(argv)


def _tiny_png_data_url() -> str:
    """Build an 8x8 solid PNG first frame without external dependencies."""

    width = height = 8
    raw = b"".join(b"\x00" + b"\x46\x8c\x2f" * width for _ in range(height))

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
        )

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )
    return f"data:image/png;base64,{base64.b64encode(png).decode('ascii')}"


def _run_generation(
    adapter: MiniMaxVideoAdapter,
    *,
    prompt: str,
    references: tuple[CanonicalProviderReference, ...],
    aspect_ratio: str,
    poll_interval_seconds: float,
    timeout_seconds: float,
) -> bytes:
    request = CanonicalProviderRequest(
        model_ref=MINIMAX_H3_MODEL_REF,
        provider_model_id="minimax/h3",
        capability="video",
        prompt=prompt,
        parameters={
            "duration_seconds": 4,
            "aspect_ratio": aspect_ratio,
            "resolution": "720p",
            "generate_audio": True,
        },
        references=references,
    )
    resolution = {
        "model_ref": adapter.active_profile.model_ref,
        "adapter_id": adapter.active_profile.adapter_id,
        "transport_kind": adapter.active_profile.transport_kind,
        "adapter_revision": adapter.active_profile.adapter_revision,
        "capability_revision": adapter.active_profile.capability_revision,
    }
    compiled = adapter.compile(request, resolution)
    print(f"payload: {json.dumps(compiled.payload, ensure_ascii=False)[:300]}")
    submission = adapter.submit(compiled)
    print(f"submitted task: {submission.provider_task_id}")

    deadline = time.monotonic() + timeout_seconds
    while True:
        status = adapter.poll(submission)
        print(f"poll: {status.state}")
        if status.state in {"completed", "succeeded", "success"}:
            break
        if status.state in {"failed", "error", "cancelled"}:
            raise SystemExit(
                f"generation failed: {status.raw.get('error_code')} {status.raw.get('message')}"
            )
        if time.monotonic() > deadline:
            raise SystemExit(f"poll timed out after {timeout_seconds}s (task {submission.provider_task_id})")
        time.sleep(poll_interval_seconds)

    result = adapter.normalize(adapter.download(status))
    content = base64.b64decode(result.value.split(";base64,", 1)[1])
    if len(content) < 12 or content[4:8] != b"ftyp":
        raise SystemExit("downloaded content is not a valid MP4")
    return content


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    settings = Settings.from_env()
    api_key = args.api_key or settings.minimax_api_key
    base_url = args.base_url or settings.minimax_base_url
    if not api_key or not base_url:
        print("MINIMAX_API_KEY and MINIMAX_BASE_URL (or --api-key/--base-url) are required.")
        return 2
    gateway_settings = Settings(minimax_api_key=api_key, minimax_base_url=base_url)
    profile = ProviderAdapterProfileV1.model_validate(dict(_MINIMAX_H3_VIDEO_PROFILE))
    adapter = MiniMaxVideoAdapter(
        profile,
        transport=MiniMaxVideoTransport(gateway_settings),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    jobs = (
        (
            "t2v",
            "A cinematic slow push-in on a glass of iced lemon tea, condensation details.",
            (),
        ),
        (
            "i2v",
            "Animate this still into a gentle parallax camera move with soft light shifts.",
            (
                CanonicalProviderReference(
                    reference_id="first-frame",
                    role="storyboard",
                    input_type="data_url",
                    value=_tiny_png_data_url(),
                ),
            ),
        ),
    )
    for name, prompt, references in jobs:
        print(f"--- {name} ---")
        try:
            content = _run_generation(
                adapter,
                prompt=prompt,
                references=references,
                aspect_ratio=args.aspect_ratio,
                poll_interval_seconds=args.poll_interval_seconds,
                timeout_seconds=args.timeout_seconds,
            )
        except MiniMaxGatewayHttpError as error:
            print(f"gateway error {error.status_code} {error.gateway_code}: {error.gateway_message}")
            return 1
        output_path = args.output_dir / f"{name}-{int(time.time())}.mp4"
        output_path.write_bytes(content)
        print(
            f"{name}: {len(content)} bytes, sha256={hashlib.sha256(content).hexdigest()}, "
            f"saved to {output_path}"
        )
    print("verification complete: both clips are valid MP4 files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
