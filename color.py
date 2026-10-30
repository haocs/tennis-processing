"""Keep HDR exports in their source transfer function and color space."""


def export_profile(meta):
    hdr = meta["transfer"] in (16, 18)
    if hdr:
        if meta.get("primaries") != 9 or meta.get("matrix") != 9:
            raise ValueError("HDR export requires verified BT.2020 source color tags")
        transfer = "arib-std-b67" if meta["transfer"] == 18 else "smpte2084"
        # Old Dolby Vision dynamic metadata describes the original encoded image;
        # it must not be attached to a newly encoded HDR base-layer export.
        filters = "sidedata=mode=delete:type=DOVI_RPU_BUFFER,sidedata=mode=delete:type=DOVI_METADATA,format=p010le"
        args = ["-c:v", "hevc_videotoolbox", "-profile:v", "main10", "-q:v", "70", "-tag:v", "hvc1",
                "-color_primaries", "bt2020", "-color_trc", transfer, "-colorspace", "bt2020nc",
                "-color_range", "pc" if meta.get("range") == 2 else "tv"]
        description = {"version": 2, "codec": "HEVC Main10", "color": "HLG BT.2020" if meta["transfer"] == 18 else "PQ BT.2020",
                       "tone_mapped": False, "lossy": True, "encoder": "hevc_videotoolbox", "quality": 70,
                       "note": "Preserves the HDR base layer's color space and transfer function; Dolby Vision dynamic metadata is not retained. Display/player HDR handling may still differ from the original."}
    else:
        filters = "format=yuv420p"
        args = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18"]
        description = {"version": 2, "codec": "H.264", "color": "Source SDR", "tone_mapped": False, "lossy": True, "crf": 18}
    return filters, args, description


def verify_color(source, output):
    if source["transfer"] in (16, 18):
        if any(source[key] != output[key] for key in ("transfer", "primaries", "matrix", "range")):
            raise RuntimeError("HDR color metadata changed during export")
        if output.get("pixel_format") not in {"yuv420p10le", "p010le"}:
            raise RuntimeError("HDR export lost 10-bit precision")
