import copy
import hashlib
import json
from pathlib import Path
import pytest
from cut_constraints import apply,export_fps_for_source,load,min_frame_duration_ms,validate,validate_parameter_overrides,version_id

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


def test_nonlocal_overrides_rejected_for_disjoint_own_clip_or_overlapping_other_clip():
    p=payload({'keep_intervals_ms':[],'exclude_intervals_ms':[],'override_clip_bounds_ms':[dict(clip_id='base',original_start_ms=2000,original_end_ms=5000,source_annotation_id='review',source_revision=2)]})
    with pytest.raises(ValueError,match='override_does_not_overlap_base_clip'):
        validate(p,SHA,30)
    with pytest.raises(ValueError,match='override_does_not_overlap_base_clip'):
        apply([[10,20]],p)
    # Touching endpoint only (5000..10000 against 10000..20000) is not an overlap.
    p['confirmed_commands']['override_clip_bounds_ms'][0].update(original_start_ms=5000,original_end_ms=10000)
    with pytest.raises(ValueError,match='override_does_not_overlap_base_clip'):
        validate(p,SHA,30)
    # Add a second base clip [22s, 28s]; overriding 'base' into [10s, 22.001s] overlaps the second clip.
    p['base_manifest']['clips'].append({'id':'second','original_start_seconds':22,'original_end_seconds':28})
    p['confirmed_commands']['override_clip_bounds_ms'][0].update(original_start_ms=10000,original_end_ms=22001)
    with pytest.raises(ValueError,match='override_overlaps_other_base_clip'):
        validate(p,SHA,30)
    with pytest.raises(ValueError,match='override_overlaps_other_base_clip'):
        apply([[10,20],[22,28]],p)
    # Touching the boundary of the next base clip without crossing into it is valid.
    p['confirmed_commands']['override_clip_bounds_ms'][0].update(original_start_ms=10000,original_end_ms=22000)
    validate(p,SHA,30)
    assert apply([[10,20],[22,28]],p)==[[10,22],[22,28]]


@pytest.mark.parametrize('bad_overrides',[
    'not-a-dict',
    {'unknown_param':1},
    {'padding_before':True},
    {'padding_after':-0.1},
    {'merge_gap':float('nan')},
    {'swing_speed':float('inf')},
    {'arm_reach':10001},
    {'target_region':[0.1,0.2,0.9]},
    {'target_region':[False,0.2,0.9,0.8]},
    {'target_region':[0.8,0.2,0.4,0.8]},
    {'target_region':[0.1,0.8,0.9,0.2]},
    {'target_region':[-0.1,0.2,0.9,0.8]},
    {'target_region':[0.1,0.2,1.1,0.8]},
])
def test_parameter_overrides_validated_independently(bad_overrides):
    p=payload()
    p['parameter_overrides']=bad_overrides
    with pytest.raises(ValueError,match='invalid_parameter_overrides'):
        validate(p,SHA,30)


def test_valid_parameter_overrides_accepted():
    p=payload()
    p['parameter_overrides']={'target_region':[0.05,0.1,0.95,0.9],'padding_before':1.5,'padding_after':2,'merge_gap':0}
    validate(p,SHA,30)
    assert validate_parameter_overrides(p['parameter_overrides'])==p['parameter_overrides']


def test_subframe_detector_remainder_omitted_with_findings_and_confirmed_subframe_fails():
    assert export_fps_for_source(29.97)==30
    assert export_fps_for_source(59.94)==60
    assert round(min_frame_duration_ms(export_fps=30),3)==33.333
    assert round(min_frame_duration_ms(export_fps=60),3)==16.667
    # Unforced 20ms detector remainder after hard exclude [10000, 19980] on [10, 20]:
    p=payload({'keep_intervals_ms':[],'exclude_intervals_ms':[command(10000,19980)],'override_clip_bounds_ms':[]})
    validate(p,SHA,30,export_fps=30)
    # Default pure apply retains exact ms remainder for backward compatibility:
    assert apply([[10,20]],p)==[[19.98,20.0]]
    # With 30fps export (frame=33.333ms), the 20ms unforced detector remainder is omitted, never restoring the hard exclude:
    findings30={}
    assert apply([[10,20]],p,export_fps=30,findings_out=findings30)==[]
    assert findings30=={
        'export_fps':30,
        'min_frame_duration_ms':33.333,
        'omitted_subframe_count':1,
        'omitted_subframe_remainders_ms':[{'start_ms':19980,'end_ms':20000,'duration_ms':20,'reason':'subframe_detector_remainder_omitted'}],
    }
    # With 60fps export (frame=16.667ms), 20ms is >= 1 export frame so it is kept, whereas a 10ms remainder is omitted:
    findings60={}
    assert apply([[10,20]],p,export_fps=60,findings_out=findings60)==[[19.98,20.0]]
    assert findings60['omitted_subframe_count']==0
    p10=payload({'keep_intervals_ms':[],'exclude_intervals_ms':[command(10000,19990)],'override_clip_bounds_ms':[]})
    findings60_drop={}
    assert apply([[10,20]],p10,export_fps=60,findings_out=findings60_drop)==[]
    assert findings60_drop['omitted_subframe_count']==1
    # Explicitly confirmed keep or override shorter than 1 export frame must fail clearly:
    bad_keep=payload({'keep_intervals_ms':[command(5000,5020)],'exclude_intervals_ms':[],'override_clip_bounds_ms':[]})
    with pytest.raises(ValueError,match='subframe_confirmed_interval'):
        validate(bad_keep,SHA,30,export_fps=30)
    with pytest.raises(ValueError,match='subframe_confirmed_interval'):
        apply([[10,20]],bad_keep,export_fps=30)
    bad_override=payload({'keep_intervals_ms':[],'exclude_intervals_ms':[],'override_clip_bounds_ms':[dict(clip_id='base',original_start_ms=15000,original_end_ms=15020,source_annotation_id='review',source_revision=2)]})
    with pytest.raises(ValueError,match='subframe_confirmed_interval'):
        validate(bad_override,SHA,30,export_fps=30)
    with pytest.raises(ValueError,match='subframe_confirmed_interval'):
        apply([[10,20]],bad_override,export_fps=30)
