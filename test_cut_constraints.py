import copy
import hashlib
import json
from pathlib import Path
import pytest
from cut_constraints import apply,load,validate,version_id

SHA='a'*64


def payload(commands=None):
    base={'id':SHA,'published_at':1234.125,'processing':{'attempt_id':'b'*32},'source':{'sha256':SHA,'duration_seconds':30},'clips':[{'id':'base','original_start_seconds':10,'original_end_seconds':20}]}
    return {'schema_version':1,'compiler_version':'cut-constraints-compiler-v1','source_sha256':SHA,'base_attempt_id':'b'*32,'base_video_version_id':version_id(SHA,base['published_at']),'base_manifest':base,
        'selected_evidence_revisions':[{'annotation_id':'review','version':2,'kind':'cut_feedback','context':{'publication':base['published_at']}}],
        'confirmed_commands':commands or {'keep_intervals_ms':[],'exclude_intervals_ms':[command(15000,15001)],'override_clip_bounds_ms':[]}}


def command(start,end):return dict(start_ms=start,end_ms=end,source_annotation_id='review',source_revision=2)


def test_hard_exclusion_is_never_bridged_by_detector_merge_gap():
    p=payload();validate(p,SHA,30)
    assert apply([[10,20]],p)==[[10,15],[15.001,20]]
    p['confirmed_commands']['exclude_intervals_ms']=[command(0,30000)]
    assert apply([[10,20]],p)==[]


def test_override_targets_base_cut_even_when_detector_boundaries_changed():
    p=payload({'keep_intervals_ms':[],'exclude_intervals_ms':[],'override_clip_bounds_ms':[dict(clip_id='base',original_start_ms=12000,original_end_ms=18000,source_annotation_id='review',source_revision=2)]})
    validate(p,SHA,30)
    assert apply([[8,22]],p)==[[8,10],[12,18],[20,22]]
    p['confirmed_commands']['override_clip_bounds_ms'][0].update(original_start_ms=8000,original_end_ms=22000)
    assert apply([[0,30]],p)==[[0,8],[8,22],[22,30]]


def test_keep_commands_work_without_detector_activity():
    p=payload({'keep_intervals_ms':[command(1000,2000)],'exclude_intervals_ms':[],'override_clip_bounds_ms':[]});validate(p,SHA,30)
    assert apply([],p)==[[1,2]]


@pytest.mark.parametrize('change',['source','attempt','version','bool','nan','conflict'])
def test_invalid_source_bound_or_conflicting_snapshots_reject(change):
    p=payload()
    if change=='source':p['source_sha256']='c'*64
    elif change=='attempt':p['base_attempt_id']='c'*32
    elif change=='version':p['base_video_version_id']='v_'+'0'*32
    elif change=='bool':p['confirmed_commands']['exclude_intervals_ms'][0]['start_ms']=True
    elif change=='nan':p['confirmed_commands']['exclude_intervals_ms'][0]['end_ms']=float('nan')
    else:p['confirmed_commands']['keep_intervals_ms']=[command(14000,16000)]
    with pytest.raises(ValueError):validate(p,SHA,30)


def test_snapshot_hash_symlink_bounds_and_local_retry(tmp_path):
    raw=json.dumps(payload(),sort_keys=True,separators=(',',':')).encode();digest=hashlib.sha256(raw).hexdigest()
    relative=f'snapshots/{SHA}/{digest}.json';source=tmp_path/relative;source.parent.mkdir(parents=True);source.write_bytes(raw)
    ref={'storage_id':'processing','path':relative,'sha256':digest,'bytes':len(raw)};local=tmp_path/'local/snapshot.json'
    assert load(tmp_path,ref,local,SHA,30)['base_attempt_id']=='b'*32
    source.unlink();assert load(tmp_path,ref,local,SHA,30)['source_sha256']==SHA
    with pytest.raises(ValueError):load(tmp_path,dict(ref,path='../escape'),local,SHA,30)
    with pytest.raises(ValueError):load(tmp_path,dict(ref,sha256='f'*64,path=f'snapshots/{SHA}/'+('f'*64)+'.json'),local,SHA,30)
    source.symlink_to(local)
    with pytest.raises(ValueError,match='symlink'):load(tmp_path,ref,tmp_path/'new-local.json',SHA,30)


def test_upgraded_worker_refreshes_profile_without_registering_again(tmp_path,monkeypatch):
    import api_worker
    root=tmp_path/'state';root.mkdir();(root/'worker.json').write_text(json.dumps({'worker_id':'existing','worker_token':'fixture-only','sequence':7}))
    calls=[]
    def call(self,path,body=None,**kwargs):calls.append((path,body));return {'version':8}
    monkeypatch.setattr(api_worker.API,'call',call)
    worker=api_worker.Agent({'state_dir':str(root),'name':'Test','storage':{},'processor':{}})
    assert calls[0][0]=='/workers/existing/profile' and 'cut-constraints-v1' in calls[0][1]['capabilities']
    assert worker.state['worker_id']=='existing' and worker.sequence==8


def test_old_server_profile_absence_preserves_worker_identity(tmp_path,monkeypatch):
    import api_worker
    root=tmp_path/'state';root.mkdir();(root/'worker.json').write_text(json.dumps({'worker_id':'existing','worker_token':'fixture-only'}))
    def old(*args,**kwargs):raise api_worker.APIError(404,'old server')
    monkeypatch.setattr(api_worker.API,'call',old)
    worker=api_worker.Agent({'state_dir':str(root),'name':'Test','storage':{},'processor':{}})
    assert worker.state['worker_id']=='existing'


def test_empty_result_retains_metadata_without_encoding_media(tmp_path,monkeypatch):
    import pipeline
    def no_media(*args,**kwargs):raise AssertionError("Empty candidate must not encode media")
    monkeypatch.setattr(pipeline,"run_media",no_media)
    monkeypatch.setattr(pipeline,"build_playback",no_media)
    monkeypatch.setattr(pipeline,"export_profile",lambda meta:("",[],{"fixture":True}))
    meta={"duration":30,"fps":30,"width":160,"height":90,"transfer":1,"recorded_at":"2026-01-01T00:00:00Z","original_metadata":{"container_tags":{"make":"TestCamera"}}}
    out=tmp_path/"output"
    result=pipeline.render_session(tmp_path/"source.mov",out,{},SHA,"source.mov",[],{"detector":"synthetic"},meta=meta)
    assert result["clips"]==0 and result["playback"]["duration_seconds"]==0
    assert result["source_metadata"]["original_metadata"]==meta["original_metadata"]
    assert not (out/"Practice.mp4").exists() and (out/"edit-timeline.json").is_file()
    assert 'No retained clips' in (out/"Review.html").read_text()
