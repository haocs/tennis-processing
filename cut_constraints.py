"""Hash-verified, source-bound cut commands applied after detector padding/merging."""
import hashlib
import json
import math
from pathlib import Path,PurePosixPath
import re

MAX_BYTES=256*1024
CAPABILITY='cut-constraints-v1'
KINDS={'keep_intervals_ms','exclude_intervals_ms','override_clip_bounds_ms'}
PARAMETERS={'target_region','min_player_height','swing_speed','arm_reach','padding_before','padding_after','merge_gap'}


def version_id(source,publication):
    return 'v_'+hashlib.sha256(json.dumps([source,float(publication)],separators=(',',':'),allow_nan=False).encode()).hexdigest()[:32]


def export_fps_for_source(source_fps):
    if type(source_fps) not in (int,float) or not math.isfinite(source_fps) or source_fps<=0:
        raise ValueError('invalid_source_fps')
    return 60 if source_fps>=45 else 30


def min_frame_duration_ms(export_fps=None,min_duration_ms=None):
    if min_duration_ms is not None:
        if type(min_duration_ms) not in (int,float) or not math.isfinite(min_duration_ms) or min_duration_ms<0:
            raise ValueError('invalid_min_duration_ms')
        return float(min_duration_ms)
    if export_fps is not None:
        if type(export_fps) not in (int,float) or not math.isfinite(export_fps) or export_fps<=0:
            raise ValueError('invalid_export_fps')
        return 1000.0/float(export_fps)
    return 0.0


def validate_parameter_overrides(parameters):
    if not isinstance(parameters,dict) or set(parameters)-PARAMETERS:
        raise ValueError('invalid_parameter_overrides')
    for key,value in parameters.items():
        if key=='target_region':
            if (not isinstance(value,list) or len(value)!=4
                or any(type(v) not in (float,int) or not math.isfinite(v) or not 0<=v<=1 for v in value)
                or not value[0]<value[2] or not value[1]<value[3]):
                raise ValueError('invalid_parameter_overrides')
        elif type(value) not in (int,float) or not math.isfinite(value) or value<0 or value>10000:
            raise ValueError('invalid_parameter_overrides')
    return parameters


def validate_override_bounds(commands,clips):
    base_bounds={}
    for cid,clip in clips.items():
        cs,ce=clip.get('original_start_seconds'),clip.get('original_end_seconds')
        if type(cs) not in (int,float) or type(ce) not in (int,float) or not math.isfinite(cs) or not math.isfinite(ce) or not 0<=cs<ce:
            raise ValueError('invalid_base_clips')
        base_bounds[cid]=(round(cs*1000),round(ce*1000))
    seen=set();spans=[]
    for row in commands.get('override_clip_bounds_ms',[]):
        cid=row.get('clip_id')
        if cid not in base_bounds:raise ValueError('override_clip_not_in_base')
        a,b=row.get('original_start_ms'),row.get('original_end_ms')
        if type(a) is not int or type(b) is not int or not 0<=a<b:raise ValueError('invalid_command_interval')
        if cid in seen or any(max(a,x)<min(b,y) for x,y in spans):raise ValueError('contradictory_constraints')
        base_start,base_end=base_bounds[cid]
        if not (max(a,base_start)<min(b,base_end)):raise ValueError('override_does_not_overlap_base_clip')
        if any(max(a,o_start)<min(b,o_end) for oid,(o_start,o_end) in base_bounds.items() if oid!=cid):
            raise ValueError('override_overlaps_other_base_clip')
        seen.add(cid);spans.append((a,b))


def validate(payload,source_sha256,duration_seconds,export_fps=None,min_duration_ms=None):
    def check(ok,message='invalid_cut_constraints'):
        if not ok:raise ValueError(message)
    check(isinstance(payload,dict))
    frame_ms=min_frame_duration_ms(export_fps=export_fps,min_duration_ms=min_duration_ms)
    base=payload.get('base_manifest');check(isinstance(base,dict),'missing_base_manifest')
    check(type(payload.get('schema_version')) is int and payload['schema_version']==1 and payload.get('compiler_version')=='cut-constraints-compiler-v1')
    check(payload.get('source_sha256')==source_sha256 and base.get('id')==source_sha256 and base.get('source',{}).get('sha256')==source_sha256,'source_hash_mismatch')
    check(isinstance(payload.get('base_attempt_id'),str) and bool(re.fullmatch('[a-f0-9]{32}',payload['base_attempt_id'])))
    check(payload.get('base_attempt_id')==(base.get('processing') or {}).get('attempt_id'),'base_attempt_mismatch')
    check(payload.get('base_video_version_id')==version_id(source_sha256,base['published_at']),'base_version_mismatch')
    seconds=base['source'].get('duration_seconds');check(type(seconds) in (float,int) and math.isfinite(seconds) and abs(seconds-duration_seconds)<=.002,'source_duration_mismatch')
    duration=round(duration_seconds*1000)
    base_clips=base.get('clips',[]);check(isinstance(base_clips,list),'invalid_base_clips')
    clips={}
    for c in base_clips:
        check(isinstance(c,dict) and isinstance(c.get('id'),str) and c['id'] not in clips,'invalid_base_clips')
        clips[c['id']]=c
    refs=set()
    evidence=payload.get('selected_evidence_revisions');check(isinstance(evidence,list) and 1<=len(evidence)<=50)
    for item in evidence:
        check(isinstance(item,dict) and isinstance(item.get('annotation_id'),str) and type(item.get('version')) is int and item['version']>0)
        check(item.get('kind')=='cut_feedback' and item.get('context',{}).get('publication')==base['published_at'],'evidence_version_mismatch')
        key=(item['annotation_id'],item['version'])
        check(key not in refs,'duplicate_evidence_revision')
        refs.add(key)
    if 'parameter_overrides' in payload:
        validate_parameter_overrides(payload['parameter_overrides'])
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
            if kind!='exclude_intervals_ms' and frame_ms>0:
                check((b-a)>=frame_ms,'subframe_confirmed_interval')
            if override:
                check(row['clip_id'] in clips,'override_clip_not_in_base')
                check(row['clip_id'] not in overridden and not any(max(a,x)<min(b,y) for x,y in override_spans),'contradictory_constraints')
                overridden.add(row['clip_id']);override_spans.append((a,b))
            (negatives if kind=='exclude_intervals_ms' else positives).append((a,b));count+=1
    validate_override_bounds(commands,clips)
    check(count>0,'no_confirmed_commands')
    check(not any(max(a,c)<min(b,d) for a,b in positives for c,d in negatives),'contradictory_constraints')
    if frame_ms>0 and override_spans:
        for row in commands['keep_intervals_ms']:
            rem=[[row['start_ms'],row['end_ms']]]
            for ox,oy in override_spans:rem=subtract(rem,ox,oy)
            check(all((y-x)>=frame_ms for x,y in rem),'subframe_confirmed_interval')
    return payload


def load(root,ref,local_path,source_sha256,duration_seconds,export_fps=None,min_duration_ms=None):
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
    payload=json.loads(raw);validate(payload,source_sha256,duration_seconds,export_fps=export_fps,min_duration_ms=min_duration_ms)
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


def apply(intervals,payload,export_fps=None,min_duration_ms=None,findings_out=None):
    """Replace each base span, retain confirmed keeps, then subtract hard excludes.

    Explicit override edges remain clip boundaries even against touching detector
    ranges. No merge-gap or padding step may run after this function.
    """
    frame_ms=min_frame_duration_ms(export_fps=export_fps,min_duration_ms=min_duration_ms)
    commands=payload['confirmed_commands'];base={c['id']:c for c in payload['base_manifest']['clips']}
    validate_override_bounds(commands,base)
    override_spans=[(r['original_start_ms'],r['original_end_ms']) for r in commands['override_clip_bounds_ms']]
    keep_spans=[(r['start_ms'],r['end_ms']) for r in commands['keep_intervals_ms']]
    exclude_spans=[(r['start_ms'],r['end_ms']) for r in commands['exclude_intervals_ms']]
    if any(max(a,c)<min(b,d) for a,b in override_spans+keep_spans for c,d in exclude_spans):
        raise ValueError('contradictory_constraints')
    if frame_ms>0:
        if any((b-a)<frame_ms for a,b in override_spans+keep_spans):
            raise ValueError('subframe_confirmed_interval')
        if override_spans:
            for a,b in keep_spans:
                rem=[[a,b]]
                for ox,oy in override_spans:rem=subtract(rem,ox,oy)
                if any((y-x)<frame_ms for x,y in rem):raise ValueError('subframe_confirmed_interval')
    output=[[round(a*1000),round(b*1000)] for a,b in intervals];boundaries=set()
    for row in commands['override_clip_bounds_ms']:
        clip=base[row['clip_id']];a,b=row['original_start_ms'],row['original_end_ms']
        output=subtract(output,min(round(clip['original_start_seconds']*1000),a),max(round(clip['original_end_seconds']*1000),b));boundaries.update((a,b))
    output+=[[a,b] for a,b in override_spans]
    output+=[[a,b] for a,b in keep_spans];output=union(output)
    for a,b in exclude_spans:output=subtract(output,a,b)
    result=[];omitted=[]
    for a,b in output:
        points=[a,*sorted(t for t in boundaries if a<t<b),b]
        for x,y in zip(points,points[1:]):
            if x>=y:continue
            if frame_ms>0 and (y-x)<frame_ms:
                omitted.append({'start_ms':x,'end_ms':y,'duration_ms':y-x,'reason':'subframe_detector_remainder_omitted'})
                continue
            result.append([x/1000,y/1000])
    if isinstance(findings_out,dict):
        findings_out.update({
            'export_fps':int(export_fps) if export_fps is not None and float(export_fps).is_integer() else export_fps,
            'min_frame_duration_ms':round(frame_ms,3),
            'omitted_subframe_remainders_ms':omitted,
            'omitted_subframe_count':len(omitted),
        })
    return result
