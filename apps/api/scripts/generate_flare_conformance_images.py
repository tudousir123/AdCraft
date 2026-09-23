# ruff: noqa: E402
"""Drive two real GPT Image 2.5 Flare generations through the canonical chain.

Runs inside the API container against a configured OpenAI-compatible image
gateway (IMAGE_GENERATION_API_KEY/ENDPOINT/MODEL) and persists both outputs:
one bare text-to-image request and one carrying a product reference image as a
data URL on the existing ``image`` field. Each request is billed by the gateway
at generation cost, so this script is for authorized conformance runs only.
"""

import argparse
import base64
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import Settings
from app.tools.real_media_provider import RealMediaProvider

FLARE_MODEL_ID = "gpt-image-2.5-flare"
# 1x1 transparent PNG; real runs should pass --reference-image with a product shot.
_FALLBACK_REFERENCE_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-image", type=Path, default=None)
    return parser.parse_args(argv)


def _base_request(slot_id: str, **overrides: object) -> dict[str, object]:
    request: dict[str, object] = {
        "prompt": "A glass water bottle on a bright minimal kitchen counter, "
        "soft daylight, crisp commercial product photography.",
        "slot_type": "image",
        "slot_id": slot_id,
        "semantic_type": "image",
        "provider_id": "volcengine_ark",
        "provider_model_id": FLARE_MODEL_ID,
        "aspect_ratio": "16:9",
    }
    request.update(overrides)
    return request


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    settings = Settings.from_env()
    if not settings.image_generation_api_key or not settings.image_generation_endpoint:
        print("IMAGE_GENERATION_API_KEY and IMAGE_GENERATION_ENDPOINT must be configured.")
        return 2
    if settings.image_generation_model != FLARE_MODEL_ID:
        print(
            f"IMAGE_GENERATION_MODEL is {settings.image_generation_model!r}, "
            f"expected {FLARE_MODEL_ID!r}."
        )
        return 2

    reference_bytes = (
        args.reference_image.read_bytes() if args.reference_image else _FALLBACK_REFERENCE_PNG
    )
    data_url = "data:image/png;base64," + base64.b64encode(reference_bytes).decode("utf-8")

    provider = RealMediaProvider(settings)
    workflow_id = f"flare-conformance-{settings.image_generation_model}"
    requests = [
        ("bare-text-to-image", _base_request("flare-bare")),
        (
            "product-reference-image",
            _base_request(
                "flare-reference",
                aspect_ratio="1:1",
                reference_assets=[{"asset_id": "ref-1", "provider_input_value": data_url}],
                submitted_reference_asset_ids=["ref-1"],
            ),
        ),
    ]

    failures: list[str] = []
    for label, request in requests:
        output = provider.generate_v2_canonical_image(request, workflow_id)
        asset = output["assets"][0]
        local_path = settings.media_data_dir / str(asset.get("local_path") or "")
        size = local_path.stat().st_size if local_path.exists() else 0
        print(
            f"[{label}] model={asset.get('model')} "
            f"status={asset.get('status')} download={asset.get('download_status')} "
            f"bytes={size} path={asset.get('local_path')} url={asset.get('remote_url')}"
        )
        if size <= 0:
            failures.append(label)

    if failures:
        print(f"failed: {', '.join(failures)}")
        return 1
    print("both generations persisted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
