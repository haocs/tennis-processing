import json
from pathlib import Path
import subprocess
import sys

import pytest

import publication
from playback import build_playback
from pipeline import location_metadata, run_media, probe
from storage_config import check_library_storage
import worker


def test_iphone_location_is_extracted_without_guessing():
    assert location_metadata({'com.apple.quicktime.location.ISO6709':'+12.3456-045.6789+001.000/'})=={'latitude':12.3456,'longitude':-45.6789,'altitude_m':1.0}
    assert location_metadata({'com.apple.quicktime.location.ISO6709':'+12.3456-045.6789/'})=={'latitude':12.3456,'longitude':-45.6789}
    assert location_metadata({}) is None
    assert location_metadata({'com.apple.quicktime.location.ISO6709':'+999.0-122.0/'}) is None


def test_nested_private_and_shared_storage_is_rejected(tmp_path):
    private=tmp_path/'private';private.mkdir()
    library=private/'library';library.mkdir();(library/'.tennis-library-id').write_text('id')
    with pytest.raises(RuntimeError):
        check_library_storage({'nas_root':str(private),'library_root':str(library)})
    with pytest.raises(RuntimeError):
        check_library_storage({'nas_root':str(library),'library_root':str(private)})


def test_split_publication_and_crash_recovery(tmp_path,monkeypatch):
    local=tmp_path/'result';(local/'Clips').mkdir(parents=True)
    (local/'Practice.mp4').write_bytes(b'joined')
    (local/'Clips/clip-001.mp4').write_bytes(b'clip')
    (local/'Review.html').write_text('review')
    (local/'edit-timeline.json').write_text(json.dumps({'source':'PrivateFolder/s.mov'}))
    digest='a'*64
    playback={'duration_seconds':5,'alignment':'joined-video-keyframe-pts','clips':[{'index':1,'file':'Clips/clip-001.mp4','original_start_seconds':2,'original_end_seconds':7,'duration_seconds':5,'playback_start_seconds':0,'playback_end_seconds':5}]}
    report={'source':'PrivateFolder/s.mov','source_sha256':digest,'source_metadata':{'duration':10,'width':100,'height':100,'transfer':1,'recorded_at':'2026-10-01T12:00:00-0700'},'playback':playback,'kept_seconds':5,'removed_seconds':5,'analysis':{'detector':'test'}}
    report['output_checksums']={str(p.relative_to(local)):worker.sha256(p) for p in local.rglob('*') if p.is_file()}
    worker.atomic_json(local/'session.json',report)
    monkeypatch.setattr(publication,'poster',lambda session,dest:dest.write_bytes(b'poster'))
    root=tmp_path/'shared';(root/'media'/digest).mkdir(parents=True)
    (root/'media'/digest/'partial.mp4').write_bytes(b'interrupted')
    media=publication.publish_library(local,root,digest=digest)
    manifest=root/'metadata/sessions'/f'{digest}.json'
    assert publication.validate_library(media,digest)
    assert worker.validate_publication(media,digest)
    assert not (media/'session.json').exists()
    assert list(root.glob('.orphan-*'))
    assert 'PrivateFolder' not in manifest.read_text()
    assert 'PrivateFolder' not in (root/'metadata/sessions'/digest/'edit-timeline.json').read_text()
    public=json.loads(manifest.read_text())
    assert public['clips'][0]['id']==publication.clip_id(digest,2,7)
    assert publication.publish_library(local,root,digest=digest)==media
    (media/'Practice.mp4').write_bytes(b'broken')
    assert not publication.validate_library(media,digest)


def test_chapters_follow_encoded_packets_with_audio_rounding(tmp_path):
    (tmp_path/'Clips').mkdir()
    log=tmp_path/'encode.log'
    intervals=[[2,3.17],[6,6.83]]
    for i,(a,b) in enumerate(intervals,1):
        run_media(['-f','lavfi','-i','testsrc2=size=160x90:rate=30','-f','lavfi','-i','sine=frequency=440:sample_rate=48000','-t',str(b-a),'-c:v','libx264','-preset','ultrafast','-pix_fmt','yuv420p','-c:a','aac',str(tmp_path/f'Clips/clip-{i:03d}.mp4')],log)
    concat=tmp_path/'concat.txt';concat.write_text("file 'Clips/clip-001.mp4'\nfile 'Clips/clip-002.mp4'\n")
    run_media(['-f','concat','-safe','1','-i',str(concat),'-c','copy',str(tmp_path/'Practice.mp4')],log)
    playback=build_playback(tmp_path,intervals)
    assert playback['clips'][0]['playback_start_seconds']==0
    assert 1.1<playback['clips'][1]['playback_start_seconds']<1.3
    assert playback['clips'][1]['playback_start_seconds']!=6
    assert playback['clips'][-1]['playback_end_seconds']==pytest.approx(probe(tmp_path/'Practice.mp4')['duration'])


def test_unavailable_shared_output_does_not_create_folders_or_process(tmp_path,monkeypatch):
    private=tmp_path/'private';private.mkdir();(private/'new.mov').write_bytes(b'original')
    shared=tmp_path/'missing-shared';state=tmp_path/'state'
    def unexpected(*args):raise AssertionError('Must not process while shared destination is unavailable')
    monkeypatch.setattr(worker,'process_source',unexpected)
    with pytest.raises(RuntimeError):
        worker.run_once({'input_root':str(private),'library_root':str(shared),'state_dir':str(state),'settle_seconds':0})
    assert not shared.exists() and (private/'new.mov').read_bytes()==b'original'
    assert json.loads((state/'status.json').read_text())['storage']=='unavailable'


def test_public_status_removes_private_storage_paths(tmp_path):
    from monitor import public_snapshot
    snapshot={'pid':12,'current_video':'/private/Originals/practice.mov','files':[{'path':'/private/Originals/practice.mov','output':'/private/processed/a','error':'failed /private/file'}],'scan_error':'/private/file','mirror_error':None}
    clean=public_snapshot(snapshot)
    assert '/private' not in json.dumps(clean) and 'pid' not in clean
    assert clean['current_video']=='practice.mov'
    assert snapshot['files'][0]['path'].startswith('/private')
