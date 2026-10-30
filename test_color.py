import pytest
from color import export_profile, verify_color


def hdr():
    return {"transfer": 18, "primaries": 9, "matrix": 9, "range": 1, "pixel_format": "yuv420p10le"}


def test_hdr_export_keeps_hlg_and_10bit_without_tone_mapping():
    filters, args, info = export_profile(hdr())
    assert "tonemap" not in filters and "p010le" in filters
    assert args[args.index("-color_trc") + 1] == "arib-std-b67"
    assert args[args.index("-profile:v") + 1] == "main10"
    assert info["tone_mapped"] is False
    assert "DOVI_METADATA" in filters
    verify_color(hdr(), hdr())


@pytest.mark.parametrize("key,value", [("transfer", 1), ("primaries", 1), ("matrix", 1), ("range", 2), ("pixel_format", "yuv420p")])
def test_color_verification_rejects_hdr_regression(key, value):
    with pytest.raises(RuntimeError):
        verify_color(hdr(), dict(hdr(), **{key: value}))


def test_unverified_hdr_color_tags_fail_instead_of_guessing():
    with pytest.raises(ValueError):
        export_profile(dict(hdr(), primaries=2))


def test_sdr_remains_sdr():
    filters, args, info = export_profile({"transfer": 1})
    assert filters == "format=yuv420p" and "libx264" in args
    assert not info["tone_mapped"]
