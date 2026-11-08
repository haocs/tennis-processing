"""Hash-verified, source-bound cut commands applied after detector padding/merging."""
import hashlib
import json
import math
from pathlib import Path,PurePosixPath
import re

MAX_BYTES=256*1024
CAPABILITY='cut-constraints-v1'
KINDS={'keep_intervals_ms','exclude_intervals_ms','override_clip_bounds_ms'}


def version_id(source,publication):
    return 'v_'+hashlib.sha256(json.dumps([source,float(publication)],separators=(',',':'),allow_nan=False).encode()).hexdigest()[:32]


def validate(payload,source_sha256,duration_seconds):
    def check(ok,message='invalid_cut_constraints'):
        if not ok:raise ValueError(message)
    check(isinstance(payload,dict))
    base=payload.get('base_manifest');check(isinstance(base,dict),'missing_base_manifest')
    check(type(payload.get('schema_version')) is int and payload['schema_version']==1 and payload.get('compiler_version')=='cut-constraints-compiler-v1')
    check(payload.get('source_sha256')==source_sha256 and base.get('id')==source_sha256 and base.get('source',{}).get('sha256')==source_sha256,'source_hash_mismatch')
    check(isinstance(payload.get('base_attempt_id'),str) and bool(re.fullmatch('[a-f0-9]{32}',payload['base_attempt_id'])))
    check(payload.get('base_attempt_id')==(base.get('processing') or {}).get('attempt_id'),'base_attempt_mismatch')
    check(payload.get('base_video_version_id')==version_id(source_sha256,base['published_at']),'base_version_mismatch')
    seconds=base['source'].get('duration_seconds');check(type(seconds) in (float,int) and math.isfinite(seconds) and abs(seconds-duration_seconds)<=.002,'source_duration_mismatch')
    duration=round(duration_seconds*1000)
    clips={c['id']:c for c in base.get('clips',[])}
    refs=set()
    evidence=payload.get('selected_evidence_revisions');check(isinstance(evidence,list) and 1<=len(evidence)<=50)
    for item in evidence:
        check(isinstance(item,dict) and isinstance(item.get('annotation_id'),str) and type(item.get('version')) is int and item['version']>0)
        check(item.get('kind')=='cut_feedback' and item.get('context',{}).get('publication')==base['published_at'],'evidence_version_mismatch')
        refs.add((item['annotation_id'],item['version']))
    commands=payload.get('confirmed_commands');check(isinstance(commands,dict) and set(commands)==KINDS)
    positives=[];negatives=[];override_spans=[];overridden=set();count=0
    for kind,rows in commands.items():
        check(isinstance(rows,list) and len(rows)<=50)
        for row in rows:
            check(isinstance(row,dict))
            override=kind=='override_clip_bounds_ms';keys={'clip_id','original_start_ms','original_end_ms'} if override else {'start_ms','end_ms'}
            check(set(row)==keys|{'source_annotation_id','source_revision'})
            check(type(row['source_revision']) is int and (row['source_annotation_id'],row['source_revision']) in refs,'command_evidence_missing')
            a,b=(row['original_start_ms'],row['original_end_ms']) if override else (row['start_ms'],row['end_ms'])
            check(type(a) is int and type(b) is int and 0<=a<b<=duration,'invalid_command_interval')
            if override:
                check(row['clip_id'] in clips,'override_clip_not_in_base')
                check(row['clip_id'] not in overridden and not any(max(a,x)<min(b,y) for x,y in override_spans),'contradictory_constraints')
                overridden.add(row['clip_id']);override_spans.append((a,b))
            (negatives if kind=='exclude_intervals_ms' else positives).append((a,b));count+=1
    check(count>0,'no_confirmed_commands')
    check(not any(max(a,c)<min(b,d) for a,b in positives for c,d in negatives),'contradictory_constraints')
    return payload


def load(root,ref,local_path,source_sha256,duration_seconds):
    if not isinstance(ref,dict) or set(ref)!={'storage_id','path','sha256','bytes'} or ref['storage_id']!='processing':raise ValueError('invalid_cut_constraints_ref')
    if not isinstance(ref['sha256'],str) or not re.fullmatch('[a-f0-9]{64}',ref['sha256']) or type(ref['bytes']) is not int or not 0<ref['bytes']<=MAX_BYTES:raise ValueError('invalid_cut_constraints_ref')
    expected=f"snapshots/{source_sha256}/{ref['sha256']}.json"
    if ref['path']!=expected:raise ValueError('invalid_cut_constraints_path')
    local=Path(local_path)
    if local.is_file():raw=local.read_bytes()
    else:
        path=Path(root)
        for part in PurePosixPath(ref['path']).parts:
            path=path/part
            if path.is_symlink():raise ValueError('symlink_cut_constraints')
        with path.open('rb') as stream:raw=stream.read(MAX_BYTES+1)
    if len(raw)!=ref['bytes'] or hashlib.sha256(raw).hexdigest()!=ref['sha256']:raise ValueError('cut_constraints_hash_mismatch')
    payload=json.loads(raw);validate(payload,source_sha256,duration_seconds)
    if not local.exists():
        local.parent.mkdir(parents=True,exist_ok=True);temporary=local.with_suffix('.tmp');temporary.write_bytes(raw);temporary.replace(local)
    return payload


def union(intervals):
    merged=[]
    for a,b in sorted(intervals):
        if not merged or a>merged[-1][1]:merged.append([a,b])
        else:merged[-1][1]=max(b,merged[-1][1])
    return merged


def subtract(intervals,a,b):
    result=[]
    for x,y in intervals:
        if y<=a or x>=b:result.append([x,y]);continue
        if x<a:result.append([x,a])
        if b<y:result.append([b,y])
    return result


def apply(intervals,payload):
    """Replace each base span, retain confirmed keeps, then subtract hard excludes.

    Explicit override edges remain clip boundaries even against touching detector
    ranges. No merge-gap or padding step may run after this function.
    """
    commands=payload['confirmed_commands'];base={c['id']:c for c in payload['base_manifest']['clips']}
    output=[[round(a*1000),round(b*1000)] for a,b in intervals];boundaries=set()
    for row in commands['override_clip_bounds_ms']:
        clip=base[row['clip_id']];a,b=row['original_start_ms'],row['original_end_ms']
        output=subtract(output,min(round(clip['original_start_seconds']*1000),a),max(round(clip['original_end_seconds']*1000),b));boundaries.update((a,b))
    output+= [[r['original_start_ms'],r['original_end_ms']] for r in commands['override_clip_bounds_ms']]
    output+=[[r['start_ms'],r['end_ms']] for r in commands['keep_intervals_ms']];output=union(output)
    for row in commands['exclude_intervals_ms']:output=subtract(output,row['start_ms'],row['end_ms'])
    result=[]
    for a,b in output:
        points=[a,*sorted(t for t in boundaries if a<t<b),b]
        result.extend([[x/1000,y/1000] for x,y in zip(points,points[1:]) if x<y])
    return result
