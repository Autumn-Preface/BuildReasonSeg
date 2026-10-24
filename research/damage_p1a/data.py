"""Explicit annotations/validity, split-before-tiles, native RGB coordinates."""
from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from PIL import Image


def require_admission(config):
    gate=json.loads(Path(config['admission_file']).read_text(encoding='utf-8'))
    if gate.get('status')!='ADMITTED' or not gate.get('training_allowed'):
        raise RuntimeError('DATA_ADMISSION_BLOCKED: '+', '.join(gate.get('reasons', ['unreviewed dataset'])))
    if gate.get('class_map') is None or gate.get('validity_policy') is None:
        raise RuntimeError('DATA_ADMISSION_BLOCKED: missing authoritative mapping/validity policy')
    if gate.get('manifest_sha256')!=hashlib.sha256(Path(config['manifest']).read_bytes()).hexdigest():
        raise RuntimeError('DATA_ADMISSION_BLOCKED: manifest identity mismatch')
    return gate


def validate_manifest(rows):
    """Reject same source/content/known group across splits before creating any crop."""
    seen={}
    for row in rows:
        if row['split'] not in ('train','val','test'):raise ValueError('unknown split')
        for key in ('source_image_id','image_sha256','building_id','sequence_id','tile_parent_id'):
            value=row.get(key)
            if value is None:continue
            ident=(key,value)
            if ident in seen and seen[ident]!=row['split']:raise ValueError('cross-split leakage: '+key)
            seen[ident]=row['split']
        _known_channels(row.get('known_channels'))


def _known_channels(value):
    if not isinstance(value,(list,tuple)) or len(value)!=2 or any(type(v) is not bool for v in value):
        raise ValueError('explicit boolean channel validity required')


def parse_yolo(text, class_map):
    """class_map is explicit evidence-backed raw integer→output channel; never filenames."""
    result=[]
    for line in text.splitlines():
        if not line.strip():continue
        tokens=line.split();raw=int(tokens[0]);points=np.asarray(tokens[1:],dtype=np.float64)
        if raw not in class_map:raise ValueError('unmapped annotation ID')
        if len(points)<6 or len(points)%2 or not np.isfinite(points).all() or np.any((points<0)|(points>1)):
            raise ValueError('invalid normalized polygon')
        polygon=points.reshape(-1,2)
        signed_area_twice=np.sum(polygon[:,0]*np.roll(polygon[:,1],-1)-polygon[:,1]*np.roll(polygon[:,0],-1))
        if len(np.unique(polygon,axis=0))<3 or signed_area_twice==0:
            raise ValueError('degenerate polygon; do not turn positive annotation into background')
        result.append((class_map[raw],polygon))
    return result


def rasterize(polygons, height, width, known_channels, ignore_polygons=()):
    """COCO pixel-edge convention; union instances, preserve cross-class overlap.

    No hidden priority or inferred negative channels. Explicit unknown polygons are ignored.
    """
    from pycocotools import mask as coco
    _known_channels(known_channels)
    target=np.zeros((2,height,width),dtype=bool)
    valid=np.broadcast_to(np.asarray(known_channels,dtype=bool)[:,None,None],target.shape).copy()
    def decode(points):
        p=np.asarray(points,dtype=np.float64)*np.array([width,height])
        decoded=coco.decode(coco.frPyObjects([p.reshape(-1).tolist()],height,width))[:,:,0].astype(bool)
        if not decoded.any():raise ValueError('polygon has zero rasterized pixels; do not silently create negative GT')
        return decoded
    for channel,points in polygons:
        if channel not in (0,1):raise ValueError('invalid output channel')
        if not known_channels[channel]:raise ValueError('positive annotation in unknown channel')
        target[channel]|=decode(points)
    for channel,points in ignore_polygons:valid[channel,decode(points)]=False
    return target,valid


@dataclass(frozen=True)
class Tile:
    top:int
    left:int
    height:int
    width:int


def starts(length,size=512,overlap=128):
    if length<=0 or size<=0 or not 0<=overlap<size:raise ValueError('invalid tile geometry')
    if length<=size:return [0]
    positions=list(range(0,length-size+1,size-overlap))
    if positions[-1]!=length-size:positions.append(length-size)
    return positions


def tiles(height,width,size=512,overlap=128):
    return [Tile(y,x,min(size,height-y),min(size,width-x)) for y in starts(height,size,overlap) for x in starts(width,size,overlap)]


def crop_pad(array,tile,size=512,fill=0):
    result=np.full((*array.shape[:-2],size,size),fill,dtype=array.dtype)
    result[...,:tile.height,:tile.width]=array[...,tile.top:tile.top+tile.height,tile.left:tile.left+tile.width]
    return result


class Reconstructor:
    def __init__(self,height,width):
        self.total=np.zeros((2,height,width),np.float32)
        self.count=np.zeros((height,width),np.uint32)
    def add(self,tile,probability):
        p=np.asarray(probability)
        if not all(isinstance(v,(int,np.integer)) for v in (tile.top,tile.left,tile.height,tile.width)):
            raise ValueError('invalid tile geometry')
        if min(tile.top,tile.left)<0 or min(tile.height,tile.width)<=0 or tile.top+tile.height>self.count.shape[0] or tile.left+tile.width>self.count.shape[1]:
            raise ValueError('tile outside native image')
        if p.ndim!=3 or p.shape[0]!=2 or p.shape[1]<tile.height or p.shape[2]<tile.width or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
            raise ValueError('invalid probability shape/value')
        self.total[:,tile.top:tile.top+tile.height,tile.left:tile.left+tile.width]+=p[:,:tile.height,:tile.width]
        self.count[tile.top:tile.top+tile.height,tile.left:tile.left+tile.width]+=1
    def finish(self):
        if not self.count.all():raise ValueError('uncovered native pixels')
        return self.total/self.count[None]


def load_record(row, root, class_map):
    root=Path(root).resolve()
    def checked(name):
        path=(root/row[name]).resolve()
        if not path.is_relative_to(root):raise ValueError('path escapes data root')
        return path
    with Image.open(checked('image')) as im:
        if im.mode!='RGB':raise ValueError('RGB required; no implicit multispectral conversion')
        rgb=np.array(im)
    if hashlib.sha256(checked('image').read_bytes()).hexdigest()!=row['image_sha256']:raise ValueError('image identity drift')
    text=checked('label').read_text(encoding='utf-8-sig')
    if hashlib.sha256(checked('label').read_bytes()).hexdigest()!=row['label_sha256']:raise ValueError('label identity drift')
    masks,valid=rasterize(parse_yolo(text,class_map),*rgb.shape[:2],row['known_channels'],row.get('ignore_polygons',[]))
    return rgb,masks,valid
