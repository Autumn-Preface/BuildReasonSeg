"""Small raw-ID GT alignment and exact-duplicate evidence, never model predictions."""
import hashlib
import argparse
import io
import json
import zipfile
import zlib
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from . import audit_sources as source
from .data import parse_yolo,rasterize


def digest(data):return hashlib.sha256(data).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--existing-p0',action='store_true')
    args=parser.parse_args()
    local=source.ROOT/'artifacts/damage_p1a/alignment_samples';local.mkdir(parents=True,exist_ok=True)
    inv=json.loads((source.OUT/'official_archive_inventory.json').read_text(encoding='utf-8'))
    # Exact training members previously inspected in P0, now read from the current official archive.
    if args.existing_p0:
        image_root=source.ROOT/'datasets/damage/local/cubit'
    else:
        image_root=local
        remote=source.RemoteZip(inv['splits']['train']['archives']['images.zip']['drive_id'])
        with zipfile.ZipFile(remote) as z:
            for stem in ['CR1_1','CR1_10']:(local/(stem+'.JPG')).write_bytes(z.read('images/'+stem+'.JPG'))
    pairs=json.loads((source.OUT/'duplicate_sample_audit.json').read_text(encoding='utf-8'))['verified_pairs']
    records=[]
    examples=[('train','CR1_1',image_root/'CR1_1.JPG'),('train','CR1_10',image_root/'CR1_10.JPG')]
    examples += [(p['files'][0]['split'],Path(p['files'][0]['archive_path']).stem,source.ROOT/p['files'][0]['local']) for p in pairs]
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',19)
    small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',15)
    evidence=source.OUT/'sample_evidence';evidence.mkdir(exist_ok=True)
    for split,stem,path in examples:
        with zipfile.ZipFile(source.ROOT/'artifacts/damage_p1a/raw_labels'/f'{split}_labels.zip') as z:
            item=next(i for i in z.infolist() if Path(i.filename).stem==stem)
            text=z.read(item).decode('utf-8-sig')
        rgb=np.array(Image.open(path).convert('RGB'));h,w=rgb.shape[:2]
        metadata=next(x for x in inv['splits'][split]['archives']['images.zip']['files'] if Path(x['name']).stem==stem)
        assert len(path.read_bytes())==metadata['bytes'] and f'{zlib.crc32(path.read_bytes()):08x}'==metadata['crc32']
        # Identity mapping here means raw ID→display index only, never crack/spalling.
        polys=parse_yolo(text,{0:0,1:1});gt,_=rasterize(polys,h,w,[True,True])
        overlay=rgb.copy()
        for c,color in enumerate([(255,70,70),(70,220,255)]):overlay[gt[c]]=(.5*overlay[gt[c]]+.5*np.array(color)).astype(np.uint8)
        preview=Image.fromarray(overlay);preview.thumbnail((720,500))
        ys,xs=np.where(gt.any(0));mid=len(xs)//2
        cx,cy=(int(xs[mid]),int(ys[mid])) if len(xs) else (w//2,h//2)
        left=max(0,min(w-256,cx-128));top=max(0,min(h-256,cy-128))
        plate=Image.new('RGB',(760,920),'#101c2e');d=ImageDraw.Draw(plate)
        d.text((20,15),f'RAW GT AUDIT ONLY | {split}/{stem} | {w} x {h}',font=font,fill='white')
        d.text((20,45),'Red = numeric ID 0; cyan = numeric ID 1. Semantic mapping UNCONFIRMED.',font=small,fill='#f4cd7b')
        plate.paste(preview,(20,80))
        d.text((20,600),f'Native 256px crop at (left={left}, top={top}); image / annotation overlay',font=small,fill='white')
        plate.paste(Image.fromarray(rgb[top:top+256,left:left+256]),(20,630))
        plate.paste(Image.fromarray(overlay[top:top+256,left:left+256]),(300,630))
        d.text((20,895),'CUBIT-InSeg, Zhao et al., 2026 | CC BY 4.0 | GT audit, not predictions',font=small,fill='#b4c5dd')
        filename=stem+'_raw_gt.png';plate.save(evidence/filename)
        records.append({'split':split,'stem':stem,'image_sha256':digest(path.read_bytes()),'label_sha256':digest(text.encode()),
            'shape_hw':[h,w],'raw_id_instances':{str(c):sum(k==c for k,p in polys) for c in [0,1]},
            'raw_id_foreground_pixels':gt.sum((1,2)).tolist(),'evidence':filename,
            'image_source':'HISTORICAL_P0_READ_ONLY_SIZE_CRC_MATCH_CURRENT_DIRECTORY' if args.existing_p0 and stem.startswith('CR') else 'CURRENT_OFFICIAL_RANGE_DOWNLOAD',
            'scope':'small deterministic raw-ID alignment audit, not semantic validation or predictions'})
    p=pairs[0];plate=Image.new('RGB',(1440,660),'#101c2e');d=ImageDraw.Draw(plate)
    d.text((20,15),'CONFIRMED TRAIN / VAL EXACT DUPLICATE | no model predictions',font=font,fill='white')
    for i,row in enumerate(p['files']):
        image=Image.open(source.ROOT/row['local']);image.thumbnail((690,500));plate.paste(image,(20+i*710,90))
        d.text((20+i*710,55),row['split']+'/'+Path(row['archive_path']).name,font=font,fill='white')
    d.text((20,605),'Same encoded JPEG SHA256 AND decoded RGB SHA256; full hashes in duplicate_sample_audit.json.',font=small,fill='#f4cd7b')
    plate.save(evidence/'confirmed_train_val_duplicate.png')
    source.dump('sample_alignment.json',records)
    print(json.dumps(records,indent=2))


if __name__=='__main__':main()
