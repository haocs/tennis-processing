"""Portable API worker. Local state is a cache; the server owns leases/history."""
from __future__ import annotations
import argparse
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

LOG=logging.getLogger('tennis.api')
PARAMETERS={'target_region','min_player_height','swing_speed','arm_reach','padding_before','padding_after','merge_gap'}


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


class LeaseLost(RuntimeError): pass
class Cancelled(RuntimeError): pass


class Agent:
    def __init__(self,config):
        self.config=config;self.root=Path(config['state_dir']);self.root.mkdir(parents=True,exist_ok=True)
        self.state_path=self.root/'worker.json';self.state=json.loads(self.state_path.read_text()) if self.state_path.exists() else {}
        self.api=API(config,self.state);self.lock=threading.RLock();self.assignment=None;self.last_progress=0;self.failed=None;self.stop=threading.Event();self.next_heartbeat=0
        self.sequence=self.state.get('sequence',0);self.pending=None;self.deadline=None
        code=hashlib.sha256()
        for name in ('api_worker.py','pipeline.py','detector.py','color.py','playback.py','timeline.py'):
            code.update((Path(__file__).parent/name).read_bytes())
        model=Path(config.get('processor',{}).get('rallyclip_path',''))/'models/rallyclip_v0.6.0/yolov8n-pose-544x960-static.onnx'
        self.runtime={'executor_version':'2.0.0','build_digest':code.hexdigest(),'model_sha256':checksum(model) if model.is_file() else None,'backend':config.get('processor',{}).get('pose_device','coreml')}

        if not self.state.get('worker_id'):
            import platform
            data=self.api.call('/workers/register',{'display_name':config['name'],'platform':platform.system().lower(),'worker_version':'2.0.0','capabilities':['process-v2'],'accessible_storage_ids':list(config['storage']),'runtime':self.runtime},bootstrap=True)
            self.state.update(data);atomic(self.state_path,self.state)

    def ref(self,ref):
        spec=self.config['storage'][ref['storage_id']];root=Path(spec['path']).resolve()
        if not root.is_dir(): raise OSError('storage_unavailable')
        if spec.get('identity') and (root/spec.get('marker','.tennis-storage-id')).read_text().strip()!=spec['identity']: raise OSError('storage_identity_mismatch')
        relative=ref['path']
        if not relative or '\\' in relative or ':' in relative or relative.startswith('/') or '..' in PurePosixPath(relative).parts: raise ValueError('invalid_storage_path')
        path=(root/relative).resolve()
        if not path.is_relative_to(root) or path==root: raise ValueError('invalid_storage_path')
        return path

    def persist(self):
        with self.lock:
            self.state['sequence']=self.sequence
            self.state['active']=self.assignment
            atomic(self.state_path,self.state)

    def heartbeat(self):
        with self.lock:
            self.sequence+=1
            body={'sequence':self.sequence,'runtime':self.runtime,'stage':self.state.get('stage','Idle'),'last_progress_at':self.last_progress,'local_state':{'task_id':self.assignment['task']['id'] if self.assignment else None,'attempt_id':self.assignment['attempt']['id'] if self.assignment else None,'acknowledged_sequence':self.state.get('event_sequence',0)}}
            if self.assignment:
                a=self.assignment['attempt'];body['active']={'attempt_id':a['id'],'lease_token':a['lease_token']}
            result=self.api.call('/workers/'+self.state['worker_id']+'/heartbeat',body)
            if self.assignment:
                self.assignment['attempt']['lease_expires_at']=result['lease_expires_at']
                self.deadline=time.monotonic()+max(0,result['lease_expires_at']-result['server_time']-5)
                if result.get('cancel'): self.failed=Cancelled('cancelled')
            self.persist()
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
        fingerprint=hashlib.sha256(json.dumps({'task':task,'runtime':self.runtime,'effective_parameters':{k:self.config['processor'].get(k) for k in PARAMETERS}},sort_keys=True).encode()).hexdigest()
        if cache_data.get('fingerprint')!=fingerprint: cache_data={'fingerprint':fingerprint,'stages':{}}
        timing=JobTiming(source.name,time.time(),attempt['number']);pipeline.EXECUTION_GUARD=self.guard
        cfg=dict(self.config['processor']);cfg.update({k:v for k,v in task.get('parameters',{}).items() if k in PARAMETERS});cfg.update(_monitor=self,_timing=timing,_guard=self.guard)
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
                if stat.st_size!=input_ref['bytes'] or stat.st_mtime_ns!=input_ref['modified_ns']: raise ValueError('source_changed')
                value=checksum(local)
                if input_ref.get('sha256') and value!=input_ref['sha256']: raise ValueError('source_hash_mismatch')
                return value
            digest=stage('verify_source',verify,lambda:[local]);timing.digest=digest
            def prepare():
                meta,proxy=pipeline.prepare_video(local,scratch,cfg)
                return meta
            meta=stage('prepare_proxy',prepare,lambda:[scratch/'analysis.mp4'])
            timing.info={k:meta[k] for k in ('duration','width','height','fps','transfer')}
            analysis=stage('analyze',lambda:list(pipeline.analyze_video(scratch/'analysis.mp4',meta,cfg)),lambda:[])
            out.mkdir(exist_ok=True)
            def process():
                if out.exists(): shutil.rmtree(out)
                out.mkdir()
                return pipeline.render_session(local,out,cfg,digest,source.name,analysis[0],analysis[1],meta)
            report=stage('process',process,lambda:list(out.rglob('*')))
            timing.info['clips']=report['clips']
            bundle=scratch/'bundle'
            def package():
                if bundle.exists(): shutil.rmtree(bundle)
                media=bundle/'media';metadata=bundle/'metadata';media.mkdir(parents=True);metadata.mkdir()
                _,public=metadata_for_session(out)
                for name in ['Practice.mp4']+[c['file'] for c in public['clips']]:
                    target=media/name;target.parent.mkdir(parents=True,exist_ok=True);os.link(out/name,target)
                poster(out,media/'poster.jpg')
                for name in public['artifacts']: shutil.copyfile(out/name,metadata/name)
                atomic(metadata/'original-metadata.json',meta)
                public['artifacts'].append('original-metadata.json')
                public['media_checksums']={p.relative_to(media).as_posix():checksum(p) for p in media.rglob('*') if p.is_file()}
                public['metadata_checksums']={p.name:checksum(p) for p in metadata.iterdir() if p.is_file()}
                original=source.stat()
                if original.st_size!=input_ref['bytes'] or original.st_mtime_ns!=input_ref['modified_ns'] or checksum(source)!=digest: raise ValueError('source_changed')
                atomic(bundle/'result.json',{'schema_version':1,'task_id':task['id'],'attempt_id':attempt['id'],'source_sha256':digest,'session':public,'parameters':task.get('parameters',{}),'pipeline_version':'2.0.0','runtime':self.runtime,'files':[{'path':p.relative_to(bundle).as_posix(),'bytes':p.stat().st_size,'sha256':checksum(p)} for p in bundle.rglob('*') if p.is_file()]})
                return {'clips':report['clips'],'kept_seconds':report['kept_seconds']}
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
            payload={'task_id':task['id'],'attempt_id':attempt['id'],'lease_token':attempt['lease_token'],'final_event_sequence':self.state.get('event_sequence',0),'error_code':'cancelled' if isinstance(exc,Cancelled) else 'lease_lost' if isinstance(exc,LeaseLost) else 'processing_failed','retryable':isinstance(exc,OSError),'timing':timing.report('failed')}
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
                    except APIError:
                        self.assignment=None
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


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--once',action='store_true');args=parser.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    config=json.loads(Path(args.config).read_text())
    try:
        with worker_lock(Path(config['state_dir'])/'agent.lock'):
            Agent(config).run(args.once)
    except BlockingIOError: raise SystemExit('This worker state directory is already in use')


if __name__=='__main__': main()
