"""Parse human annotations and render paired evidence. No inference or label generation."""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import collections,hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[3]
LOCAL=ROOT/'datasets/damage/local'
OUT=ROOT/'reports/damage_p0/20261019/sample_evidence'
OUT.mkdir(exist_ok=True)
font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)

def polygons_mask(size,polygons):
    m=Image.new('L',size)
    d=ImageDraw.Draw(m)
    for pts in polygons:
        d.polygon([tuple(p) for p in pts],fill=255)
    return np.asarray(m)>0

def overlay(im,planes):
    a=np.array(im,dtype=np.float32)
    for cls,m in planes.items():
        color=(255,30,160) if cls in ['0','2','crack'] else (10,220,255)
        a[m]=a[m]*0.50+np.array(color)*0.50
    return Image.fromarray(a.astype('uint8'))

records=[]
for group in ['cubit','deepcrack','cif']:
    directory=LOCAL/group
    if not directory.exists():continue
    for p in sorted(directory.iterdir()):
        if p.suffix.lower() not in ['.jpg','.jpeg']:continue
        im=Image.open(p).convert('RGB');w,h=im.size
        r={'dataset':group,'image':str(p.relative_to(ROOT)),'image_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
           'width':w,'height':h,'planes':{},'errors':[]}
        planes={}
        if group=='cubit':
            ann=p.with_suffix('.txt');poly=collections.defaultdict(list)
            lines=ann.read_text().splitlines()
            for line in lines:
                v=line.split();coords=list(map(float,v[1:]));
                if len(coords)<6 or len(coords)%2: r['errors'].append('invalid polygon');continue
                if min(coords)<0 or max(coords)>1:r['errors'].append('normalized coordinates outside 0..1')
                # Pixel-edge coordinates; no clipping or mask inflation before audit.
                pts=[(coords[i]*w,coords[i+1]*h) for i in range(0,len(coords),2)]
                poly[v[0]].append(pts)
            for c,polys in poly.items():planes[c]=polygons_mask(im.size,polys)
            r['instances']=len(lines)
        elif group=='deepcrack':
            ann=p.with_suffix('.png');label=Image.open(ann)
            r['annotation_size']=list(label.size);r['unique_values']=np.unique(label).tolist()
            if label.size!=im.size:r['errors'].append('dimension mismatch')
            planes['crack']=np.asarray(label)>0
        else:
            ann=p.with_suffix('.json');d=json.loads(ann.read_text(encoding='utf-8'))
            if (d['width'],d['height'])!=im.size:r['errors'].append('metadata dimension mismatch')
            poly=collections.defaultdict(list)
            for cls,segments in zip(d['objects']['category_id'],d['objects']['segmentation']):
                for segment in segments:
                    if len(segment)<6 or len(segment)%2:r['errors'].append('invalid polygon');continue
                    pts=[(segment[i],segment[i+1]) for i in range(0,len(segment),2)]
                    if any(x<0 or x>w or y<0 or y>h for x,y in pts):r['errors'].append('polygon out of bounds')
                    poly[str(cls)].append(pts)
            planes={c:polygons_mask(im.size,polys) for c,polys in poly.items()}
            r['instances']=len(d['objects']['id'])
        r.update(annotation=str(ann.relative_to(ROOT)),annotation_sha256=hashlib.sha256(ann.read_bytes()).hexdigest())
        for c,m in planes.items():
            small=np.asarray(Image.fromarray(m).resize((512,512),Image.Resampling.NEAREST))
            r['planes'][c]={'foreground_pixels':int(m.sum()),'foreground_fraction':float(m.mean()),'fraction_after_512_square_nearest':float(small.mean())}
        union=np.logical_or.reduce(list(planes.values()))
        yy,xx=np.where(union)
        mid=len(yy)//2;x=int(xx[mid]);y=int(yy[mid])
        crop=(max(0,min(w-256,x-128)),max(0,min(h-256,y-128)))
        crop=(*crop,crop[0]+256,crop[1]+256)
        r['native_crop_xyxy']=list(crop)
        vis=overlay(im,planes)
        sheet=Image.new('RGB',(1160,410),'#f5f5f5');draw=ImageDraw.Draw(sheet)
        draw.text((10,8),f'{group} | {p.name} | {w}x{h} | HUMAN annotation overlay (not a prediction)',font=font,fill='black')
        for x0,panel,title in [(10,im,'RGB overview'),(305,vis,'Annotation overview'),(600,im.crop(crop),'Native 256px crop'),(890,vis.crop(crop),'Same crop + annotation')]:
            panel=panel.copy();panel.thumbnail((270,300))
            sheet.paste(panel,(x0,65));draw.text((x0,40),title,font=font,fill='black')
        draw.text((10,375),'Pink: crack / CUBIT class 0; cyan: spalling / CUBIT class 1. CUBIT numeric mapping requires official confirmation.',font=font,fill='black')
        op=OUT/(group+'_'+p.stem+'.png');sheet.save(op)
        r['evidence']=str(op.relative_to(ROOT));records.append(r)
(ROOT/'evaluation/damage_p0/sample_checks.json').write_text(json.dumps({'mode':'HUMAN_GT_PARSER_NO_INFERENCE','records':records},indent=2),encoding='utf-8')
print('paired samples',len(records),'parser errors',sum(len(x['errors']) for x in records))
