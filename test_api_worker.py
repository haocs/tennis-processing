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


def test_smb_timestamp_rounding_does_not_reject_unchanged_source():
    from types import SimpleNamespace
    from api_worker import matches_signature
    expected={'bytes':100,'modified_ns':1_000_000_123}
    assert matches_signature(SimpleNamespace(st_size=100,st_mtime_ns=1_000_000_100),expected)
    assert not matches_signature(SimpleNamespace(st_size=101,st_mtime_ns=1_000_000_100),expected)
    assert not matches_signature(SimpleNamespace(st_size=100,st_mtime_ns=1_010_000_100),expected)


def test_background_worker_stays_alive_and_recovers_from_network_failure(monkeypatch):
    import api_worker
    calls=[];sleeps=[]
    class Fake:
        def __init__(self,config):pass
        def run(self,once):
            calls.append(once)
            if len(calls)==1:raise OSError('network unavailable')
    monkeypatch.setattr(api_worker,'Agent',Fake)
    api_worker.run_supervised({},sleep=sleeps.append)
    assert calls==[False,False] and sleeps==[15]
    calls.clear()
    with pytest.raises(OSError):api_worker.run_supervised({},once=True,sleep=sleeps.append)


def test_mac_storage_reconnects_configured_mount_and_checks_identity(tmp_path,monkeypatch):
    import api_worker
    a=bare(tmp_path)
    a.config['storage']['originals'].update(smb_url='//user@nas/share/originals',identity='expected',marker='.storage-id')
    calls=[]
    def mount(command,**kwargs):
        calls.append(command)
        (tmp_path/'.storage-id').write_text('expected')
    monkeypatch.setattr(api_worker.sys,'platform','darwin')
    monkeypatch.setattr(api_worker.subprocess,'run',mount)
    assert a.ref({'storage_id':'originals','path':'video.mov'})==tmp_path/'video.mov'
    assert calls==[['/sbin/mount_smbfs','-N','//user@nas/share/originals',str(tmp_path)]]
    a.ref({'storage_id':'originals','path':'video.mov'})
    assert len(calls)==1
    (tmp_path/'.storage-id').write_text('different')
    with pytest.raises(OSError,match='identity'):a.ref({'storage_id':'originals','path':'video.mov'})


def test_profile_refresh_retries_after_404_and_503_on_heartbeat_preserving_active_task(tmp_path,monkeypatch):
    import api_worker
    root=tmp_path/'state';root.mkdir()
    active={'task':{'id':'task-1'},'attempt':{'id':'att-1','lease_token':'lease-1','lease_expires_at':200}}
    (root/'worker.json').write_text(json.dumps({'worker_id':'existing','worker_token':'fixture-only','sequence':10,'active':active,'event_sequence':4}))
    calls=[]
    mode={'profile':'404'}
    def call(self,path,body=None,bootstrap=False,**kwargs):
        assert not bootstrap and path!='/workers/register'
        calls.append((path,body))
        if path=='/workers/existing/profile':
            if mode['profile']=='404':raise api_worker.APIError(404,'old server')
            if mode['profile']=='503':raise api_worker.APIError(503,'unavailable')
            if mode['profile']=='401':raise api_worker.APIError(401,'unauthorized')
            return {'version':12}
        if path=='/workers/existing/heartbeat':
            return {'lease_expires_at':300,'server_time':100,'cancel':False}
        raise AssertionError(path)
    monkeypatch.setattr(api_worker.API,'call',call)
    cfg={'state_dir':str(root),'name':'Test','storage':{'originals':{'path':str(tmp_path)}},'processor':{}}
    worker=api_worker.Agent(cfg)
    assert worker.state['worker_id']=='existing' and worker.sequence==10 and not worker.profile_synced
    assert worker.state['active']['task']['id']=='task-1'
    # Transient 503 during heartbeat profile retry keeps worker_id/sequence and still sends capabilities on heartbeat:
    mode['profile']='503'
    worker.assignment=worker.state['active']
    worker.heartbeat()
    assert not worker.profile_synced and worker.sequence==11
    hb_body=calls[-1][1]
    assert 'cut-constraints-v1' in hb_body['capabilities'] and hb_body['accessible_storage_ids']==['originals']
    assert hb_body['active']=={'attempt_id':'att-1','lease_token':'lease-1'}
    assert hb_body['local_state']=={'task_id':'task-1','attempt_id':'att-1','acknowledged_sequence':4}
    # Once server upgrade completes, next heartbeat syncs profile and advances sequence monotonically:
    mode['profile']='ok'
    worker.heartbeat()
    assert worker.profile_synced and worker.sequence==13 and worker.state['sequence']==13
    assert worker.assignment['attempt']['lease_expires_at']==300
    # Auth 401/403 on profile sync must raise APIError and never bootstrap a new identity:
    mode['profile']='401'
    with pytest.raises(api_worker.APIError) as exc_info:
        api_worker.Agent(cfg)
    assert exc_info.value.code==401
    assert json.loads((root/'worker.json').read_text())['worker_id']=='existing'


def test_supervised_daemon_retries_transient_5xx_and_429_but_not_auth_401_403(monkeypatch):
    import api_worker
    errors=[api_worker.APIError(502,'bad gateway'),api_worker.APIError(503,'unavailable'),api_worker.APIError(429,'rate limit'),None]
    sleeps=[]
    class TransientAgent:
        def __init__(self,config):
            err=errors.pop(0)
            if err is not None:raise err
        def run(self,once):return
    monkeypatch.setattr(api_worker,'Agent',TransientAgent)
    api_worker.run_supervised({},sleep=sleeps.append)
    assert sleeps==[15,15,15]
    for code in (401,403,400):
        sleeps.clear()
        class AuthFailAgent:
            def __init__(self,config):raise api_worker.APIError(code,'auth failed')
            def run(self,once):return
        monkeypatch.setattr(api_worker,'Agent',AuthFailAgent)
        with pytest.raises(api_worker.APIError) as exc_info:
            api_worker.run_supervised({},sleep=sleeps.append)
        assert exc_info.value.code==code and sleeps==[]


def test_worker_rejects_subframe_confirmed_keep_before_ffmpeg(tmp_path,monkeypatch):
    import api_worker,pipeline
    from cut_constraints import version_id
    roots={name:{'path':str(tmp_path/name)} for name in ('originals','processing')}
    for r in roots.values():Path(r['path']).mkdir()
    source=Path(roots['originals']['path'])/'clip.mov';source.write_bytes(b'synthetic-video-bytes')
    sha=hashlib.sha256(source.read_bytes()).hexdigest();stat=source.stat()
    base={'id':sha,'published_at':1000.0,'processing':{'attempt_id':'b'*32},'source':{'sha256':sha,'duration_seconds':30},'clips':[{'id':'base','original_start_seconds':10,'original_end_seconds':20}]}
    snap={'schema_version':1,'compiler_version':'cut-constraints-compiler-v1','source_sha256':sha,'base_attempt_id':'b'*32,'base_video_version_id':version_id(sha,1000.0),'base_manifest':base,
          'selected_evidence_revisions':[{'annotation_id':'rev','version':1,'kind':'cut_feedback','context':{'publication':1000.0}}],
          'confirmed_commands':{'keep_intervals_ms':[{'start_ms':5000,'end_ms':5020,'source_annotation_id':'rev','source_revision':1}],'exclude_intervals_ms':[],'override_clip_bounds_ms':[]}}
    raw=json.dumps(snap,sort_keys=True,separators=(',',':')).encode();snap_sha=hashlib.sha256(raw).hexdigest()
    snap_rel=f'snapshots/{sha}/{snap_sha}.json';snap_path=Path(roots['processing']['path'])/snap_rel;snap_path.parent.mkdir(parents=True);snap_path.write_bytes(raw)
    calls=[]
    def call(self,path,body=None,**kwargs):
        calls.append((path,body))
        if path.endswith('/register'):return {'worker_id':'w1','worker_token':'tok'}
        if path.endswith('/heartbeat'):return {'lease_expires_at':9999999999,'server_time':100,'cancel':False}
        if path.endswith('/events'):return {'acknowledged_sequence':body['events'][-1]['sequence']}
        return {}
    monkeypatch.setattr(api_worker.API,'call',call)
    monkeypatch.setattr(pipeline,'probe',lambda p:{'duration':30.0,'fps':30.0,'width':160,'height':90,'transfer':1})
    def no_ffmpeg(*args,**kwargs):raise AssertionError('FFmpeg must not run when confirmed keep is shorter than 1 export frame')
    monkeypatch.setattr(pipeline,'prepare_video',no_ffmpeg)
    monkeypatch.setattr(pipeline,'render_session',no_ffmpeg)
    worker=api_worker.Agent({'state_dir':str(tmp_path/'worker'),'name':'Test','storage':roots,'processor':{}})
    assignment={'task':{'id':'t1','type':'process','input':{'storage_id':'originals','path':'clip.mov','bytes':stat.st_size,'modified_ns':stat.st_mtime_ns,'sha256':sha},
                        'parameters':{'cut_constraints_ref':{'storage_id':'processing','path':snap_rel,'sha256':snap_sha,'bytes':len(raw)}}},
                'attempt':{'id':'a1','number':1,'lease_token':'lt','lease_expires_at':9999999999,'output_prefix':{'storage_id':'processing','path':'runs/a1'}}}
    worker.execute(assignment)
    assert calls[-1][0]=='/attempts/a1/fail' and calls[-1][1]['error_code']=='processing_failed'


def test_worker_execute_omits_subframe_detector_remainder_and_records_findings(tmp_path,monkeypatch):
    import api_worker,pipeline
    from cut_constraints import version_id
    roots={name:{'path':str(tmp_path/name)} for name in ('originals','processing')}
    for r in roots.values():Path(r['path']).mkdir()
    source=Path(roots['originals']['path'])/'clip.mov';source.write_bytes(b'synthetic-video-bytes')
    sha=hashlib.sha256(source.read_bytes()).hexdigest();stat=source.stat()
    base={'id':sha,'published_at':1000.0,'processing':{'attempt_id':'b'*32},'source':{'sha256':sha,'duration_seconds':30},'clips':[{'id':'base','original_start_seconds':10,'original_end_seconds':20}]}
    # Exclude [10000, 19980] from detector [10, 20], leaving a 20ms (<33.333ms at 30fps) unforced detector remainder:
    snap={'schema_version':1,'compiler_version':'cut-constraints-compiler-v1','source_sha256':sha,'base_attempt_id':'b'*32,'base_video_version_id':version_id(sha,1000.0),'base_manifest':base,
          'selected_evidence_revisions':[{'annotation_id':'rev','version':1,'kind':'cut_feedback','context':{'publication':1000.0}}],
          'confirmed_commands':{'keep_intervals_ms':[],'exclude_intervals_ms':[{'start_ms':10000,'end_ms':19980,'source_annotation_id':'rev','source_revision':1}],'override_clip_bounds_ms':[]}}
    raw=json.dumps(snap,sort_keys=True,separators=(',',':')).encode();snap_sha=hashlib.sha256(raw).hexdigest()
    snap_rel=f'snapshots/{sha}/{snap_sha}.json';snap_path=Path(roots['processing']['path'])/snap_rel;snap_path.parent.mkdir(parents=True);snap_path.write_bytes(raw)
    calls=[]
    def call(self,path,body=None,**kwargs):
        calls.append((path,body))
        if path.endswith('/register'):return {'worker_id':'w1','worker_token':'tok'}
        if path.endswith('/heartbeat'):return {'lease_expires_at':9999999999,'server_time':100,'cancel':False}
        if path.endswith('/events'):return {'acknowledged_sequence':body['events'][-1]['sequence']}
        return {}
    monkeypatch.setattr(api_worker.API,'call',call)
    meta={'duration':30.0,'fps':30.0,'width':160,'height':90,'transfer':1,'recorded_at':'2026-01-01T00:00:00Z','original_metadata':{'container_tags':{'make':'Apple'}}}
    monkeypatch.setattr(pipeline,'probe',lambda p:meta)
    def fake_prepare(src,scratch,cfg):
        (scratch/'analysis.mp4').write_bytes(b'proxy')
        return meta,scratch/'analysis.mp4'
    monkeypatch.setattr(pipeline,'prepare_video',fake_prepare)
    monkeypatch.setattr(pipeline,'analyze_video',lambda *args:([[10.0,20.0]],{'detector':'synthetic'}))
    monkeypatch.setattr(pipeline,'export_profile',lambda m:('',[],{'fixture':True}))
    def no_media(*args,**kwargs):raise AssertionError('Subframe remainder must be omitted without running FFmpeg media encode')
    monkeypatch.setattr(pipeline,'run_media',no_media)
    monkeypatch.setattr(pipeline,'build_playback',no_media)
    worker=api_worker.Agent({'state_dir':str(tmp_path/'worker'),'name':'Test','storage':roots,'processor':{}})
    assignment={'task':{'id':'t2','type':'dryrun','input':{'storage_id':'originals','path':'clip.mov','bytes':stat.st_size,'modified_ns':stat.st_mtime_ns,'sha256':sha},
                        'parameters':{'cut_constraints_ref':{'storage_id':'processing','path':snap_rel,'sha256':snap_sha,'bytes':len(raw)}}},
                'attempt':{'id':'a2','number':1,'lease_token':'lt','lease_expires_at':9999999999,'output_prefix':{'storage_id':'processing','path':'runs/a2'}}}
    worker.execute(assignment)
    assert calls[-1][0]=='/attempts/a2/complete'
    cc=calls[-1][1]['findings']['cut_constraints']
    assert cc['export_fps']==30 and cc['min_frame_duration_ms']==33.333
    assert cc['omitted_subframe_count']==1
    assert cc['omitted_subframe_remainders_ms']==[{'start_ms':19980,'end_ms':20000,'duration_ms':20,'reason':'subframe_detector_remainder_omitted'}]


def test_worker_ai_advertisement_snapshot_merge_and_zero_clip_policies(tmp_path,monkeypatch):
    import api_worker,pipeline
    from cut_constraints import version_id
    roots={name:{'path':str(tmp_path/name)} for name in ('originals','processing')}
    for r in roots.values():Path(r['path']).mkdir()
    source=Path(roots['originals']['path'])/'clip.mov';source.write_bytes(b'synthetic-video-bytes')
    sha=hashlib.sha256(source.read_bytes()).hexdigest();stat=source.stat()
    calls=[]
    def call(self,path,body=None,**kwargs):
        calls.append((path,body))
        if path.endswith('/register'):return {'worker_id':'w1','worker_token':'tok'}
        if path.endswith('/heartbeat'):return {'lease_expires_at':9999999999,'server_time':100,'cancel':False}
        if path.endswith('/events'):return {'acknowledged_sequence':body['events'][-1]['sequence']}
        return {}
    monkeypatch.setattr(api_worker.API,'call',call)
    meta={'duration':30.0,'fps':30.0,'width':160,'height':90,'transfer':1,'recorded_at':'2026-01-01T00:00:00Z','original_metadata':{'container_tags':{'make':'Apple'}}}
    monkeypatch.setattr(pipeline,'probe',lambda p:meta)
    def fake_prepare(src,scratch,cfg):
        (scratch/'analysis.mp4').write_bytes(b'proxy')
        return meta,scratch/'analysis.mp4'
    monkeypatch.setattr(pipeline,'prepare_video',fake_prepare)
    monkeypatch.setattr(pipeline,'export_profile',lambda m:('',[],{'fixture':True}))

    worker_cfg={
        'state_dir':str(tmp_path/'worker'),
        'name':'AIWorker',
        'storage':roots,
        'processor':{},
        'ai':{'enabled':True,'providers':['gemini'],'gemini_api_key':'AIzaSyntheticWorkerKey123','gemini_models':['gemini-2.5-flash'],'default_model':'gemini:gemini-2.5-flash'},
    }
    worker=api_worker.Agent(worker_cfg)
    reg_body=calls[0][1]
    assert 'hybrid-ai-vision-v1' in reg_body['capabilities']
    assert reg_body['ai_providers']==['gemini']
    assert reg_body['default_ai_model']=='gemini:gemini-2.5-flash'
    assert [m['id'] for m in reg_body['ai_models']]==['gemini:gemini-2.5-flash']
    worker.heartbeat()
    hb_body=calls[-1][1]
    assert hb_body['default_ai_model']=='gemini:gemini-2.5-flash'

    # 1. ai_only rejected via /fail with unsupported_processing_strategy
    worker.execute({
        'task':{'id':'t_ai_only','type':'process','input':{'storage_id':'originals','path':'clip.mov','bytes':stat.st_size,'modified_ns':stat.st_mtime_ns,'sha256':sha},
                'parameters':{'processing_strategy':'ai_only'}},
        'attempt':{'id':'a_ai_only','number':1,'lease_token':'lt','lease_expires_at':9999999999,'output_prefix':{'storage_id':'processing','path':'runs/t_ai_only/a_ai_only'}},
    })
    assert calls[-1][0]=='/attempts/a_ai_only/fail'
    assert calls[-1][1]['error_code']=='unsupported_processing_strategy' and calls[-1][1]['retryable'] is False

    # 2. Nested ai_tuning snapshot merge succeeds and propagates merged tuning into analyze_video
    observed_cfg={}
    base={'id':sha,'published_at':1000.0,'processing':{'attempt_id':'b'*32},'source':{'sha256':sha,'duration_seconds':30},'clips':[{'id':'base','original_start_seconds':10,'original_end_seconds':20}]}
    snap={'schema_version':1,'compiler_version':'cut-constraints-compiler-v1','source_sha256':sha,'base_attempt_id':'b'*32,'base_video_version_id':version_id(sha,1000.0),'base_manifest':base,
          'parameter_overrides':{'processing_strategy':'hybrid_ai_vision','ai_tuning':{'aggressiveness':'aggressive'}},
          'selected_evidence_revisions':[{'annotation_id':'rev','version':1,'kind':'cut_feedback','context':{'publication':1000.0}}],
          'confirmed_commands':{'keep_intervals_ms':[],'exclude_intervals_ms':[{'start_ms':0,'end_ms':1000,'source_annotation_id':'rev','source_revision':1}],'override_clip_bounds_ms':[]}}
    raw=json.dumps(snap,sort_keys=True,separators=(',',':')).encode();snap_sha=hashlib.sha256(raw).hexdigest()
    snap_rel=f'snapshots/{sha}/{snap_sha}.json';snap_path=Path(roots['processing']['path'])/snap_rel;snap_path.parent.mkdir(parents=True);snap_path.write_bytes(raw)

    def empty_hybrid_analyze(proxy,m,cfg):
        observed_cfg.update({'ai_tuning':cfg.get('_task_parameters',{}).get('ai_tuning')})
        return [],{'detector':'hybrid-ai-vision-v1','target_visible_fraction':1.0,'ai_refinement':{'actual_ai_model':None,'candidate_count':0,'batch_count_total':0,'batch_provenance':[],'attempted_models':[]},'_ai_provenance_payload':{'candidate_decisions':[],'batch_provenance':[]}}

    monkeypatch.setattr(pipeline,'analyze_video',empty_hybrid_analyze)
    # 2a. Conflicting nested aggressiveness fails
    worker.execute({
        'task':{'id':'t_conflict','type':'process','publication_policy':'never','input':{'storage_id':'originals','path':'clip.mov','bytes':stat.st_size,'modified_ns':stat.st_mtime_ns,'sha256':sha},
                'parameters':{'cut_constraints_ref':{'storage_id':'processing','path':snap_rel,'sha256':snap_sha,'bytes':len(raw)},'processing_strategy':'hybrid_ai_vision','ai_tuning':{'aggressiveness':'conservative'}}},
        'attempt':{'id':'a_conflict','number':1,'lease_token':'lt','lease_expires_at':9999999999,'output_prefix':{'storage_id':'processing','path':'runs/t_conflict/a_conflict'}},
    })
    assert calls[-1][0]=='/attempts/a_conflict/fail' and 'parameter_override_mismatch' in calls[-1][1]['message']

    # 2b. Disjoint accuracy_tier merges cleanly under publication_policy='never' with empty clips and ai-provenance.json
    worker.execute({
        'task':{'id':'t_merge_never','type':'process','publication_policy':'never','input':{'storage_id':'originals','path':'clip.mov','bytes':stat.st_size,'modified_ns':stat.st_mtime_ns,'sha256':sha},
                'parameters':{'cut_constraints_ref':{'storage_id':'processing','path':snap_rel,'sha256':snap_sha,'bytes':len(raw)},'processing_strategy':'hybrid_ai_vision','ai_tuning':{'accuracy_tier':'thorough'}}},
        'attempt':{'id':'a_merge_never','number':1,'lease_token':'lt','lease_expires_at':9999999999,'output_prefix':{'storage_id':'processing','path':'runs/t_merge_never/a_merge_never'}},
    })
    assert observed_cfg['ai_tuning']=={'aggressiveness':'aggressive','accuracy_tier':'thorough'}
    assert calls[-1][0]=='/attempts/a_merge_never/complete'
    uploaded_dir=Path(roots['processing']['path'])/'runs/t_merge_never/a_merge_never'
    result_manifest=json.loads((uploaded_dir/'result.json').read_text())
    assert 'ai-provenance.json' in result_manifest['session']['artifacts']
    assert (uploaded_dir/'metadata/ai-provenance.json').is_file()

    # 3. publication_policy='if_no_active_result' with zero post-AI intervals fails with no_play_detected_after_ai_filter
    worker.execute({
        'task':{'id':'t_auto_empty','type':'process','publication_policy':'if_no_active_result','input':{'storage_id':'originals','path':'clip.mov','bytes':stat.st_size,'modified_ns':stat.st_mtime_ns,'sha256':sha},
                'parameters':{'processing_strategy':'hybrid_ai_vision'}},
        'attempt':{'id':'a_auto_empty','number':1,'lease_token':'lt','lease_expires_at':9999999999,'output_prefix':{'storage_id':'processing','path':'runs/t_auto_empty/a_auto_empty'}},
    })
    assert calls[-1][0]=='/attempts/a_auto_empty/fail'
    assert calls[-1][1]['error_code']=='no_play_detected_after_ai_filter' and calls[-1][1]['retryable'] is False


def test_worker_fail_payload_redacts_secrets_and_paths(tmp_path,monkeypatch):
    import api_worker,pipeline
    from hybrid_ai import ProcessingTaskError
    roots={name:{'path':str(tmp_path/name)} for name in ('originals','processing')}
    for r in roots.values():Path(r['path']).mkdir()
    source=Path(roots['originals']['path'])/'clip.mov';source.write_bytes(b'synthetic-video-bytes')
    sha=hashlib.sha256(source.read_bytes()).hexdigest();stat=source.stat()
    calls=[]
    def call(self,path,body=None,**kwargs):
        calls.append((path,body))
        if path.endswith('/register'):return {'worker_id':'w1','worker_token':'tok'}
        if path.endswith('/heartbeat'):return {'lease_expires_at':9999999999,'server_time':100,'cancel':False}
        if path.endswith('/events'):return {'acknowledged_sequence':body['events'][-1]['sequence']}
        return {}
    monkeypatch.setattr(api_worker.API,'call',call)
    meta={'duration':10.0,'fps':30.0,'width':160,'height':90,'transfer':1,'recorded_at':'2026-01-01T00:00:00Z','original_metadata':{}}
    monkeypatch.setattr(pipeline,'probe',lambda p:meta)
    def fake_prepare(src,scratch,cfg):
        (scratch/'analysis.mp4').write_bytes(b'p')
        return meta,scratch/'analysis.mp4'
    monkeypatch.setattr(pipeline,'prepare_video',fake_prepare)
    def leaky_analyze(*args,**kwargs):
        raise ProcessingTaskError(
            'ai_provider_failed',
            'Failed calling https://api.example/v1?key=secret999 with Authorization Bearer TopSecretToken999 key=AIzaLeakyKey123456 at /private/sample/secret.json',
            retryable=False,
        )
    monkeypatch.setattr(pipeline,'analyze_video',leaky_analyze)
    worker=api_worker.Agent({'state_dir':str(tmp_path/'worker'),'name':'Test','storage':roots,'processor':{}})
    worker.execute({
        'task':{'id':'t_redact','type':'dryrun','input':{'storage_id':'originals','path':'clip.mov','bytes':stat.st_size,'modified_ns':stat.st_mtime_ns,'sha256':sha},
                'parameters':{'processing_strategy':'hybrid_ai_vision'}},
        'attempt':{'id':'a_redact','number':1,'lease_token':'lt','lease_expires_at':9999999999,'output_prefix':{'storage_id':'processing','path':'runs/t_redact/a_redact'}},
    })
    assert calls[-1][0]=='/attempts/a_redact/fail'
    assert calls[-1][1]['error_code']=='ai_provider_failed'
    msg=calls[-1][1]['message']
    for secret in ('AIzaLeakyKey123456','TopSecretToken999','secret999','/private/sample/secret.json'):
        assert secret not in msg
