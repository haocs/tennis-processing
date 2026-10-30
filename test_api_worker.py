import hashlib
import json
from pathlib import Path
import pytest
from api_worker import Agent, LeaseLost


def bare(tmp_path):
    agent=object.__new__(Agent);agent.config={'storage':{'originals':{'path':str(tmp_path)}}};agent.assignment={'attempt':{'lease_expires_at':0}};agent.failed=None;agent.deadline=0
    return agent


def test_storage_rejects_cross_platform_escape_and_symlink(tmp_path):
    a=bare(tmp_path)
    for path in ('../secret','C:/secret','/secret',r'..\secret'):
        with pytest.raises(ValueError): a.ref({'storage_id':'originals','path':path})
    assert a.ref({'storage_id':'originals','path':'2026/video.mov'})==tmp_path/'2026/video.mov'
    external=tmp_path.parent/'external';external.mkdir(exist_ok=True);(tmp_path/'alias').symlink_to(external)
    with pytest.raises(ValueError): a.ref({'storage_id':'originals','path':'alias/video.mov'})


def test_expired_lease_stops_execution(tmp_path):
    with pytest.raises(LeaseLost): bare(tmp_path).guard()


def test_original_metadata_sidecar_contains_source_tags(tmp_path):
    from pipeline import original_metadata
    class Codec: name='hevc';width=1920;height=1080;color_trc=18
    class Stream: codec_context=Codec();type='video';index=0;metadata={'rotate':'90'};time_base='1/600';duration=600
    class Container: metadata={'com.apple.quicktime.creationdate':'2026-10-30T12:00:00-0700','com.apple.quicktime.location.ISO6709':'+37.1-122.1+12.0/','make':'Apple'};streams=[Stream()]
    meta=original_metadata(Container())
    assert meta['container_tags']['make']=='Apple'
    assert meta['streams'][0]['metadata']['rotate']=='90'
    assert meta['streams'][0]['color_trc']==18


def test_rotation_and_binary_source_side_data_are_preserved():
    import base64
    from pipeline import original_metadata
    class Side(bytes): type='DISPLAYMATRIX'
    class Frame: rotation=90;side_data={'matrix':Side(b'original-matrix')}
    class Container: metadata={};streams=[]
    metadata=original_metadata(Container(),Frame())
    assert metadata['rotation_degrees']==90
    assert base64.b64decode(metadata['first_frame_side_data'][0]['base64'])==b'original-matrix'
