"""Portable API worker. Local state is a cache; the server owns leases/history."""
from __future__ import annotations
import argparse
import copy
import sys
import subprocess
import contextlib
import hashlib
import json
import logging
import os
from pathlib import Path, PurePosixPath
import shutil
import threading
import time
import urllib.error
import urllib.request
import uuid

from hybrid_ai import (
    AI_PARAMETERS,
    PARAMETERS,
    ProcessingTaskError,
    discover_worker_ai_profile,
    normalize_task_parameters,
    sanitize_error_message,
)

LOG=logging.getLogger('tennis.api')


def atomic(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.new');temp.write_text(json.dumps(value,indent=2,allow_nan=False));os.replace(temp,path)


def checksum(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''): h.update(block)
    return h.hexdigest()


class APIError(RuntimeError):
    def __init__(self,code,body):
        super().__init__(str(body));self.code=code


class API:
    def __init__(self,config,state): self.config,self.state=config,state
    def call(self,path,body=None,key=None,bootstrap=False):
        headers={'Content-Type':'application/json','Authorization':'Bearer '+(self.config['registration_token'] if bootstrap else self.state.get('worker_token',''))}
        if self.state.get('worker_id'): headers['X-Worker-ID']=self.state['worker_id']
        if key: headers['Idempotency-Key']=key
        req=urllib.request.Request(self.config['server_url'].rstrip('/')+'/api/v1'+path,data=None if body is None else json.dumps(body,allow_nan=False).encode(),headers=headers)
        try:
            with urllib.request.urlopen(req,timeout=15) as response:
                return json.load(response) if response.status!=204 else None
        except urllib.error.HTTPError as exc:
            raise APIError(exc.code,exc.read().decode()[:500]) from exc


def matches_signature(stat, expected):
    # SMB/Windows can round sub-microsecond timestamps differently from the NAS
    # host. Content hashes remain authoritative; allow only 1 ms stat rounding.
    return stat.st_size==expected['bytes'] and abs(stat.st_mtime_ns-expected['modified_ns'])<=1_000_000


class LeaseLost(RuntimeError): pass
class Cancelled(RuntimeError): pass


def _tree_bytes(path,seen=None):
    total=0
    if seen is None:seen=set()
    root=Path(path)
    if not root.exists():return 0
    stack=[root]
    while stack:
        current=stack.pop()
        try:
            with os.scandir(current) as entries:
                for entry in entries:
                    try:
                        if entry.is_symlink():continue
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(Path(entry.path))
                        elif entry.is_file(follow_symlinks=False):
                            st=entry.stat(follow_symlinks=False)
                            key=(st.st_dev,st.st_ino)
                            if key not in seen:
                                seen.add(key);total+=st.st_size
                    except OSError:
                        continue
        except OSError:
            continue
    return total


class Agent:
    def __init__(self,config):
        self.config=config;self.root=Path(config['state_dir']);self.root.mkdir(parents=True,exist_ok=True)
        self.state_path=self.root/'worker.json';self.state=json.loads(self.state_path.read_text()) if self.state_path.exists() else {}
        self.api=API(config,self.state);self.lock=threading.RLock();self.cleanup_lock=threading.Lock();self.assignment=None;self.last_progress=0;self.failed=None;self.stop=threading.Event();self.next_heartbeat=0
        self.sequence=self.state.get('sequence',0);self.pending=None;self.deadline=None
        code=hashlib.sha256()
        for name in ('api_worker.py','pipeline.py','detector.py','hybrid_ai.py','cut_constraints.py','diagnostics.py','publication.py','color.py','playback.py','timeline.py'):
            code.update((Path(__file__).parent/name).read_bytes())
        model=Path(config.get('processor',{}).get('rallyclip_path',''))/'models/rallyclip_v0.6.0/yolov8n-pose-544x960-static.onnx'
        self.runtime={'executor_version':'2.0.0','build_digest':code.hexdigest(),'model_sha256':checksum(model) if model.is_file() else None,'backend':config.get('processor',{}).get('pose_device','coreml')}

        import platform
        self.ai_profile=discover_worker_ai_profile(config)
        capabilities=['process-v2','cut-constraints-v1','heuristic-diagnostics-v1']+list(self.ai_profile.get('capabilities_extra',[]))
        self.profile={'display_name':config['name'],'platform':platform.system().lower(),'worker_version':'2.2.0','capabilities':capabilities,'accessible_storage_ids':list(config['storage']),'runtime':self.runtime}
        # Explicit empty fields remove obsolete model advertisements when AI is disabled.
        self.profile['ai_providers']=list(self.ai_profile.get('ai_providers') or [])
        self.profile['ai_models']=list(self.ai_profile.get('ai_models') or [])
        self.profile['default_ai_model']=self.ai_profile.get('default_ai_model')
        self.profile_synced=False
        if not self.state.get('worker_id'):
            data=self.api.call('/workers/register',self.profile,bootstrap=True)
            self.state.update(data);self.profile_synced=True;atomic(self.state_path,self.state)
        else:
            self.sync_profile()

    def sync_profile(self):
        if not getattr(self,'profile',None) or not self.state.get('worker_id'): return False
        with self.lock:
            try:
                refreshed=self.api.call('/workers/'+self.state['worker_id']+'/profile',self.profile)
                if isinstance(refreshed,dict) and type(refreshed.get('version')) is int:
                    self.sequence=max(self.sequence,refreshed['version'])
                self.state['sequence']=self.sequence;atomic(self.state_path,self.state)
                self.profile_synced=True
                return True
            except APIError as error:
                if error.code in (404,405,429) or 500<=error.code<600:
                    self.profile_synced=False
                    return False
                raise
            except OSError:
                self.profile_synced=False
                return False


    def ref(self,ref):
        spec=self.config['storage'][ref['storage_id']];root=Path(spec['path']).resolve()
        marker=root/spec.get('marker','.tennis-storage-id')
        if spec.get('smb_url') and sys.platform=='darwin' and not marker.is_file():
            # Reconnect only configured, credential-free SMB locations. macOS uses
            # its existing authenticated session / saved credentials, never prompts.
            url=spec['smb_url']
            if not url.startswith('//') or ':' in url.split('/',3)[2]:
                raise ValueError('invalid_smb_url')
            root.mkdir(parents=True,exist_ok=True)
            subprocess.run(['/sbin/mount_smbfs','-N',url,str(root)],check=True,
                           capture_output=True,timeout=30)
        if not root.is_dir(): raise OSError('storage_unavailable')
        if spec.get('identity') and (root/spec.get('marker','.tennis-storage-id')).read_text().strip()!=spec['identity']: raise OSError('storage_identity_mismatch')
        relative=ref['path']
        if not relative or '\\' in relative or ':' in relative or relative.startswith('/') or '..' in PurePosixPath(relative).parts: raise ValueError('invalid_storage_path')
        path=(root/relative).resolve()
        if not path.is_relative_to(root) or path==root: raise ValueError('invalid_storage_path')
        return path

    def _protected_task_ids(self):
        with self.lock:
            ids=set()
            for candidate in (self.assignment,self.state.get('active')):
                if isinstance(candidate,dict):
                    tid=candidate.get('task',{}).get('id')
                    if isinstance(tid,str) and tid:ids.add(tid)
            for key in ('pending_completion','pending_event'):
                item=self.state.get(key)
                if isinstance(item,dict):
                    tid=item.get('task_id')
                    if isinstance(tid,str) and tid:ids.add(tid)
            return ids

    def scratch_usage(self):
        tasks_dir=self.root/'tasks'
        cache_dir=self.root/'cache'
        protected=self._protected_task_ids()
        tasks_bytes=0;cleanable_bytes=0;task_count=0;cleanable_count=0;seen=set()
        if tasks_dir.is_dir():
            for entry in tasks_dir.iterdir():
                if entry.is_symlink():continue
                if entry.is_dir():
                    task_count+=1
                    size=_tree_bytes(entry,seen)
                    tasks_bytes+=size
                    if entry.name not in protected:
                        cleanable_bytes+=size;cleanable_count+=1
                elif entry.is_file():
                    try:
                        st=entry.stat()
                        key=(st.st_dev,st.st_ino)
                        if key not in seen:
                            seen.add(key);tasks_bytes+=st.st_size;cleanable_bytes+=st.st_size
                    except OSError:
                        pass
        cache_bytes=_tree_bytes(cache_dir) if cache_dir.is_dir() else 0
        return {'tasks_bytes':tasks_bytes,'cleanable_bytes':cleanable_bytes,'task_count':task_count,'cleanable_count':cleanable_count,'cache_bytes':cache_bytes}

    def cleanup_processed_tasks(self,requested_sequence=None):
        with self.cleanup_lock:
            tasks_dir=self.root/'tasks'
            removed_tasks=0;freed_bytes=0;errors=0
            if tasks_dir.is_dir():
                for entry in list(tasks_dir.iterdir()):
                    if entry.is_symlink():continue
                    if entry.name in self._protected_task_ids():continue
                    try:
                        size=_tree_bytes(entry) if entry.is_dir() else entry.stat().st_size
                        if entry.is_dir():
                            shutil.rmtree(entry);removed_tasks+=1;freed_bytes+=size
                        elif entry.is_file():
                            entry.unlink();freed_bytes+=size
                    except OSError:
                        errors+=1
            summary={'sequence':int(requested_sequence or 0),'completed_at':int(time.time()),'removed_tasks':removed_tasks,'freed_bytes':freed_bytes,'errors':errors}
            with self.lock:
                self.state['last_cleanup']=summary
                self.persist()
            return summary

    def persist(self):
        with self.lock:
            self.state['sequence']=self.sequence
            self.state['active']=self.assignment
            atomic(self.state_path,self.state)

    def heartbeat(self,allow_cleanup=True):
        scratch=self.scratch_usage()
        with self.lock:
            if not getattr(self,'profile_synced',True):
                self.sync_profile()
            self.sequence+=1
            body={'sequence':self.sequence,'runtime':self.runtime,'stage':self.state.get('stage','Idle'),'last_progress_at':self.last_progress,'scratch':scratch,'local_state':{'task_id':self.assignment['task']['id'] if self.assignment else None,'attempt_id':self.assignment['attempt']['id'] if self.assignment else None,'acknowledged_sequence':self.state.get('event_sequence',0)}}
            if self.state.get('last_cleanup'):
                body['last_cleanup']=self.state['last_cleanup']
            if getattr(self,'profile',None):
                if self.profile.get('worker_version'):
                    body['worker_version']=self.profile['worker_version']
                body['capabilities']=list(self.profile['capabilities'])
                body['accessible_storage_ids']=list(self.profile['accessible_storage_ids'])
                if 'ai_models' in self.profile:
                    body['ai_providers']=list(self.profile.get('ai_providers') or [])
                    body['ai_models']=list(self.profile.get('ai_models') or [])
                    body['default_ai_model']=self.profile.get('default_ai_model')
            if self.assignment:
                a=self.assignment['attempt'];body['active']={'attempt_id':a['id'],'lease_token':a['lease_token']}
            result=self.api.call('/workers/'+self.state['worker_id']+'/heartbeat',body)
            if self.assignment:
                self.assignment['attempt']['lease_expires_at']=result['lease_expires_at']
                self.deadline=time.monotonic()+max(0,result['lease_expires_at']-result['server_time']-5)
                if result.get('cancel'): self.failed=Cancelled('cancelled')
            self.persist()
            last_cleanup_seq=int((self.state.get('last_cleanup') or {}).get('sequence') or 0)
        requested_cleanup_seq=int(result.get('cleanup_requested_sequence') or 0) if isinstance(result,dict) else 0
        if allow_cleanup and requested_cleanup_seq>last_cleanup_seq:
            self.cleanup_processed_tasks(requested_cleanup_seq)
            try:
                return self.heartbeat(allow_cleanup=False)
            except Exception:
                pass
        return result

    def background(self):
        while not self.stop.wait(15):
            try: self.heartbeat()
            except APIError as exc:
                if exc.code in (401,403,409): self.failed=LeaseLost('stale_lease')
                LOG.warning('Heartbeat refused: %s',exc)
            except (OSError,ValueError): LOG.warning('Heartbeat unavailable; lease remains bounded')

    def guard(self):
        if self.failed: raise self.failed
        if self.assignment and self.deadline is not None and time.monotonic()>=self.deadline: raise LeaseLost('lease_expired')

    def event(self,kind,stage,**extra):
        self.guard()
        with self.lock:
            sequence=self.state.get('event_sequence',0)+1
            event=dict(sequence=sequence,type=kind,stage=stage,worker_at=time.time(),**extra)
            self.state['pending_event']=event;self.persist()
            self.send_pending()

    def send_pending(self):
        event=self.state.get('pending_event')
        if not event: return
        a=self.assignment['attempt']
        result=self.api.call('/attempts/'+a['id']+'/events',{'lease_token':a['lease_token'],'events':[event]})
        self.state['event_sequence']=result['acknowledged_sequence'];self.state.pop('pending_event',None);self.persist()

    def stage(self,stage,current=None,total=None,unit=None,video=None):
        # Adapter used by existing rendering progress; top-level stages stay stable.
        self.guard();self.last_progress=time.time()
        previous=self.state.get('detail');self.state['detail']=stage
        if previous!=stage or time.monotonic()-self.state.get('last_event_tick',0)>5:
            self.event('stage_progress',self.state.get('stage','process'),detail=stage,current=current,total=total,unit=unit)
            self.state['last_event_tick']=time.monotonic()

    def copy(self,source,target):
        target.parent.mkdir(parents=True,exist_ok=True)
        temp=target.with_suffix(target.suffix+'.partial')
        with source.open('rb') as src,temp.open('wb') as dst:
            while True:
                self.guard();block=src.read(4*1024*1024)
                if not block: break
                dst.write(block)
        os.replace(temp,target)

    def execute(self,assignment):
        import pipeline
        from publication import metadata_for_session,poster
        from timings import JobTiming
        self.assignment=assignment;self.failed=None;task=assignment['task'];attempt=assignment['attempt'];previous=self.state.get('active')
        if not previous or previous['attempt']['id']!=attempt['id']:
            self.state['event_sequence']=0
            self.state.pop('pending_event',None)
        self.state['event_attempt_id']=attempt['id']
        self.persist();self.heartbeat()
        if self.state.get('pending_event'): self.send_pending()
        scratch=self.root/'tasks'/task['id'];scratch.mkdir(parents=True,exist_ok=True)
        input_ref=task['input'];source=self.ref(input_ref);local=scratch/('source'+source.suffix.lower());out=scratch/'result'
        cache=scratch/'checkpoints.json';cache_data=json.loads(cache.read_text()) if cache.exists() else {}
        ai_profile=getattr(self,'ai_profile',{'ai_models':[],'default_ai_model':None})
        fingerprint=hashlib.sha256(json.dumps({'task':task,'runtime':self.runtime,'ai_profile':{'ai_models':ai_profile.get('ai_models',[]),'default_ai_model':ai_profile.get('default_ai_model')},'effective_parameters':{k:self.config.get('processor',{}).get(k) for k in PARAMETERS}},sort_keys=True).encode()).hexdigest()
        if cache_data.get('fingerprint')!=fingerprint: cache_data={'fingerprint':fingerprint,'stages':{}}
        timing=JobTiming(source.name,time.time(),attempt['number']);pipeline.EXECUTION_GUARD=self.guard
        cfg=dict(self.config.get('processor',{}));cfg.update(_monitor=self,_timing=timing,_guard=self.guard,_task_id=task['id'],_attempt_id=attempt['id'],_state_dir=str(self.root),_runtime=self.runtime,_ai_profile=ai_profile,_ai_config=self.config.get('ai') or {})
        result_ref=None;digest=None
        def stage(name,operation,files):
            self.guard();self.state['stage']=name;self.last_progress=time.time();timing.transition(name);self.event('stage_started',name)
            prior=cache_data['stages'].get(name)
            if prior and all(Path(p).is_file() and checksum(p)==h for p,h in prior['files'].items()):
                value=prior['value'];reused=True
            else:
                tick=time.monotonic();value=operation();reused=False
                cache_data['stages'][name]={'value':value,'files':{str(p):checksum(p) for p in files() if p.is_file()},'seconds':time.monotonic()-tick};atomic(cache,cache_data)
            self.event('stage_completed',name,reused=reused)
            return value
        try:
            if any(self.runtime.get(k)!=v for k,v in task.get('runtime_requirements',{}).items()): raise RuntimeError('required_worker_build_unavailable')
            def acquire():
                if shutil.disk_usage(scratch).free<input_ref['bytes']*3+1024**3: raise OSError('insufficient_scratch_space')
                self.copy(source,local)
                return {'bytes':local.stat().st_size}
            stage('acquire_source',acquire,lambda:[local])
            def verify():
                stat=source.stat()
                if not matches_signature(stat,input_ref): raise ValueError('source_changed')
                value=checksum(local)
                if input_ref.get('sha256') and value!=input_ref['sha256']: raise ValueError('source_hash_mismatch')
                return value
            digest=stage('verify_source',verify,lambda:[local]);timing.digest=digest
            cfg['_source_sha256']=digest
            parameters=task.get('parameters',{})
            if not isinstance(parameters,dict) or set(parameters)-PARAMETERS-{'cut_constraints_ref'}:raise ValueError('unsupported_task_parameters')
            from cut_constraints import apply,export_fps_for_source,load,validate_parameter_overrides
            validate_parameter_overrides({k:v for k,v in parameters.items() if k!='cut_constraints_ref'})
            constraints_ref=parameters.get('cut_constraints_ref')
            constraint_payload=None
            snap_overrides={}
            if constraints_ref:
                self.ref(constraints_ref)  # Verify mounted storage identity before opening the bounded artifact.
                probed=pipeline.probe(local)
                constraint_payload=load(self.config['storage']['processing']['path'],constraints_ref,scratch/'feedback-snapshot.json',digest,probed['duration'],export_fps=export_fps_for_source(probed['fps']))
                snap_overrides=constraint_payload.get('parameter_overrides') or {}
                for k,v in snap_overrides.items():
                    if k=='ai_tuning' and isinstance(v,dict) and isinstance(parameters.get('ai_tuning'),dict):
                        for sk,sv in v.items():
                            if sk in parameters['ai_tuning'] and parameters['ai_tuning'][sk]!=sv:
                                raise ValueError('parameter_override_mismatch')
                    elif k in parameters and parameters[k]!=v:
                        raise ValueError('parameter_override_mismatch')
                cfg['_allow_empty_analysis']=True

            merged_params={k:copy.deepcopy(v) for k,v in parameters.items() if k!='cut_constraints_ref'}
            if snap_overrides:
                for k,v in snap_overrides.items():
                    if k=='ai_tuning' and isinstance(v,dict):
                        merged_params['ai_tuning']={**dict(merged_params.get('ai_tuning') or {}),**v}
                    else:
                        merged_params[k]=copy.deepcopy(v)
                if snap_overrides.get('processing_strategy')=='vision_only':
                    for ai_k in AI_PARAMETERS:
                        merged_params.pop(ai_k,None)
                        cfg.pop(ai_k,None)
            effective_params=normalize_task_parameters(merged_params)
            explicit_keys=(set(parameters)-{'cut_constraints_ref'})|set(snap_overrides)
            cfg.update(effective_params)
            cfg['_task_parameters']=effective_params
            cfg['_explicit_parameter_keys']=explicit_keys
            if effective_params.get('processing_strategy')=='hybrid_ai_vision':
                cfg['_defer_empty_check']=True

            def prepare():
                meta,proxy=pipeline.prepare_video(local,scratch,cfg)
                return meta
            meta=stage('prepare_proxy',prepare,lambda:[scratch/'analysis.mp4'])
            timing.info={k:meta[k] for k in ('duration','width','height','fps','transfer')}
            analysis=stage('analyze',lambda:list(pipeline.analyze_video(scratch/'analysis.mp4',meta,cfg)),lambda:[p for p in (scratch/'pose-observations.json',) if p.is_file()])
            out.mkdir(exist_ok=True)
            def process():
                if out.exists(): shutil.rmtree(out)
                out.mkdir()
                intervals,findings=analysis[0],dict(analysis[1])
                if constraint_payload is not None:
                    assembly_findings={}
                    intervals=apply(intervals,constraint_payload,export_fps=export_fps_for_source(meta['fps']),findings_out=assembly_findings)
                    findings['cut_constraints']={'schema_version':1,'snapshot_sha256':constraints_ref['sha256'],'base_video_version_id':constraint_payload['base_video_version_id'],
                                                'base_attempt_id':constraint_payload['base_attempt_id'],'commands':{k:len(v) for k,v in constraint_payload['confirmed_commands'].items()},'applied_after_padding_and_merging':True,
                                                **assembly_findings}
                if not intervals and effective_params.get('processing_strategy')=='hybrid_ai_vision':
                    if task.get('publication_policy','never')=='if_no_active_result':
                        raise ProcessingTaskError('no_play_detected_after_ai_filter','AI filter retained zero play clips; original preserved.',retryable=False)
                    cfg['_allow_empty_analysis']=True
                return pipeline.render_session(local,out,cfg,digest,source.name,intervals,findings,meta)
            report=stage('process',process,lambda:list(out.rglob('*')))
            timing.info['clips']=report['clips']
            bundle=scratch/'bundle'
            def package():
                if bundle.exists(): shutil.rmtree(bundle)
                media=bundle/'media';metadata=bundle/'metadata';media.mkdir(parents=True);metadata.mkdir()
                _,public=metadata_for_session(out)
                for name in (['Practice.mp4']+[c['file'] for c in public['clips']] if public['clips'] else []):
                    target=media/name;target.parent.mkdir(parents=True,exist_ok=True);os.link(out/name,target)
                if public['clips']:poster(out,media/'poster.jpg')
                for name in public['artifacts']: shutil.copyfile(out/name,metadata/name)
                if (metadata/'heuristic-diagnostics.json').is_file():
                    from diagnostics import bind_publication_diagnostics
                    bind_publication_diagnostics(metadata/'heuristic-diagnostics.json',public,task_id=task['id'],attempt_id=attempt['id'])
                atomic(metadata/'original-metadata.json',meta)
                public['artifacts'].append('original-metadata.json')
                public['media_checksums']={p.relative_to(media).as_posix():checksum(p) for p in media.rglob('*') if p.is_file()}
                public['metadata_checksums']={p.name:checksum(p) for p in metadata.iterdir() if p.is_file()}
                original=source.stat()
                if not matches_signature(original,input_ref) or checksum(source)!=digest: raise ValueError('source_changed')
                atomic(bundle/'result.json',{'schema_version':1,'task_id':task['id'],'attempt_id':attempt['id'],'source_sha256':digest,'session':public,'parameters':task.get('parameters',{}),'pipeline_version':'2.0.0','runtime':self.runtime,'files':[{'path':p.relative_to(bundle).as_posix(),'bytes':p.stat().st_size,'sha256':checksum(p)} for p in bundle.rglob('*') if p.is_file()]})
                out_findings={'clips':report['clips'],'kept_seconds':report['kept_seconds']}
                if isinstance(report.get('analysis'),dict) and 'cut_constraints' in report['analysis']:
                    out_findings['cut_constraints']=report['analysis']['cut_constraints']
                if isinstance(report.get('analysis'),dict) and 'ai_refinement' in report['analysis']:
                    out_findings['ai_refinement']=report['analysis']['ai_refinement']
                return out_findings
            # Attempt IDs differ after retries; always refresh manifest packaging.
            cache_data['stages'].pop('package_result',None)
            findings=stage('package_result',package,lambda:list(bundle.rglob('*')))
            if task['type']!='dryrun':
                self.state['stage']='upload_result';timing.transition('upload_result');self.event('stage_started','upload_result')
                dest=self.ref(attempt['output_prefix']);dest.mkdir(parents=True,exist_ok=True)
                for p in bundle.rglob('*'):
                    if p.is_file() and p.name!='result.json': self.copy(p,dest/p.relative_to(bundle))
                # Verify on NAS while the upload stage owns a valid lease.
                manifest=json.loads((bundle/'result.json').read_text())
                for item in manifest['files']:
                    self.guard()
                    if checksum(dest/item['path'])!=item['sha256']: raise OSError('upload_checksum_failed')
                self.copy(bundle/'result.json',dest/'result.json');self.event('stage_completed','upload_result')
                result_ref={'storage_id':'processing','path':attempt['output_prefix']['path']+'/result.json'}
            else: self.event('stage_completed','upload_result',skipped=True,reason='dryrun_local_only')
            payload={'task_id':task['id'],'attempt_id':attempt['id'],'lease_token':attempt['lease_token'],'final_event_sequence':self.state['event_sequence'],'source_sha256':digest,'result_ref':result_ref,'findings':findings,'runtime':self.runtime,'timing':timing.report('done')}
            self.state['pending_completion']=payload;self.persist()
            self.api.call('/attempts/'+attempt['id']+'/complete',payload,key=attempt['id']+'-complete')
            self.state.pop('pending_completion',None)
        except Exception as exc:
            LOG.exception('Attempt stopped')
            err_code='cancelled' if isinstance(exc,Cancelled) else 'lease_lost' if isinstance(exc,LeaseLost) else exc.error_code if isinstance(exc,ProcessingTaskError) else 'processing_failed'
            retryable=exc.retryable if isinstance(exc,ProcessingTaskError) else isinstance(exc,OSError)
            err_msg=exc.message if isinstance(exc,ProcessingTaskError) else sanitize_error_message(str(exc))
            payload={'task_id':task['id'],'attempt_id':attempt['id'],'lease_token':attempt['lease_token'],'final_event_sequence':self.state.get('event_sequence',0),'error_code':err_code,'retryable':retryable,'message':err_msg,'timing':timing.report('failed')}
            try: self.api.call('/attempts/'+attempt['id']+'/fail',payload,key=attempt['id']+'-fail')
            except (OSError,APIError): LOG.warning('Failure report unacknowledged; server lease expiry handles recovery')
        finally:
            atomic(scratch/('timing-'+attempt['id']+'.json'),timing.report('done' if result_ref or self.state.get('pending_completion') else 'finished'))
            pipeline.EXECUTION_GUARD=None;self.assignment=None;self.state['stage']='Idle';self.persist()

    def run(self,once=False):
        # Pending accepted completion can be replayed even after lease termination.
        pending=self.state.get('pending_completion')
        if pending:
            try:
                identity=pending['attempt_id'];self.api.call('/attempts/'+identity+'/complete',pending,key=identity+'-complete');self.state.pop('pending_completion',None)
            except APIError as exc:
                if exc.code!=409: raise
        self.assignment=self.state.get('active')
        self.heartbeat() if not self.assignment else None
        thread=threading.Thread(target=self.background,daemon=True);thread.start()
        try:
            while True:
                pending=self.state.get('pending_completion')
                if pending:
                    try:
                        identity=pending['attempt_id']
                        self.api.call('/attempts/'+identity+'/complete',pending,key=identity+'-complete')
                        self.state.pop('pending_completion',None);self.assignment=None;self.persist()
                    except APIError as exc:
                        if exc.code==409:
                            self.state.pop('pending_completion',None);self.assignment=None;self.persist()
                        else: raise
                assignment=None
                if self.assignment:
                    try:
                        self.heartbeat();assignment=self.assignment
                    except APIError as exc:
                        if exc.code in (401,403,404,409):
                            self.assignment=None;self.persist()
                        else: raise
                if not assignment: assignment=self.api.call('/workers/'+self.state['worker_id']+'/claims',{'request_id':uuid.uuid4().hex})
                if assignment: self.execute(assignment)
                elif once: return
                if once: return
                time.sleep(10)
        finally: self.stop.set();thread.join(timeout=20)


@contextlib.contextmanager
def worker_lock(path):
    """OS releases this lock on crash; local JSON can then be reconciled safely."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+b') as stream:
        if os.name=='nt':
            import msvcrt
            stream.seek(0);stream.write(b'0');stream.flush();stream.seek(0)
            msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try: yield
        finally:
            if os.name=='nt':
                stream.seek(0);msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)


def is_transient_api_error(exc):
    return isinstance(exc,APIError) and (exc.code==429 or 500<=exc.code<600)


def run_supervised(config, once=False, sleep=time.sleep):
    # A short-lived launch agent can disappear before macOS displays its local
    # network permission alert. Keep the process alive and recover on reconnect.
    while True:
        try:
            Agent(config).run(once)
            return
        except (OSError,APIError) as exc:
            if once or (isinstance(exc,APIError) and not is_transient_api_error(exc)): raise
            LOG.warning('Server connection unavailable; keeping worker alive and retrying in 15 seconds')
            sleep(15)


@contextlib.contextmanager
def prevent_idle_sleep():
    # Keep Python as the launch agent's responsible executable so macOS applies
    # its Local Network permission to Python, not a caffeinate launcher wrapper.
    child=None
    if sys.platform=='darwin':
        child=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    try:yield
    finally:
        if child and child.poll() is None:
            child.terminate();child.wait(timeout=5)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--once',action='store_true');args=parser.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    config=json.loads(Path(args.config).read_text())
    try:
        with worker_lock(Path(config['state_dir'])/'agent.lock'), prevent_idle_sleep():
            run_supervised(config,args.once)
    except BlockingIOError: raise SystemExit('This worker state directory is already in use')


if __name__=='__main__': main()
