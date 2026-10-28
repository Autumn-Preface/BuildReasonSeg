"""Derived visual audit aids, never GT or formal labels; source pixels not edited."""
import json
from pathlib import Path
from PIL import Image,ImageDraw
BASE=Path(__file__).resolve().parent
REPO=BASE.parents[1]
s=json.loads((BASE/'samples.json').read_text(encoding='utf8'))
out=REPO/'reports/cubit_text_pilot/20261029/visual';out.mkdir(exist_ok=False)
manifest=[]
for r in s['images']:
    with Image.open(Path(s['source_root'])/r['image_relpath']) as original:
        im=original.convert('RGB');w,h=im.size
        overview=im.copy();overview.thumbnail((900,600));d=ImageDraw.Draw(overview)
        for inst in r['instances']:
            points=[(x*overview.width,y*overview.height) for x,y in inst['polygon_normalized']];d.line(points+[points[0]],fill='#ff5030',width=2);d.text(points[0],str(inst['line_1based']),fill='yellow')
        panel=Image.new('RGB',(900,630),'#1b2d35');panel.paste(overview,(0,30));ImageDraw.Draw(panel).text((8,8),r['key']+' | DERIVED OVERVIEW / NOT GT',fill='white');panel.save(out/(r['key']+'_overview.png'))
        # Center on the first original polygon; original pixel spacing, no resize in detail.
        p=r['instances'][0]['polygon_normalized'];cx=int((min(x for x,y in p)+max(x for x,y in p))/2*w);cy=int((min(y for x,y in p)+max(y for x,y in p))/2*h)
        x0=max(0,min(w-1200,cx-600));y0=max(0,min(h-900,cy-450));box=(x0,y0,min(w,x0+1200),min(h,y0+900))
        raw=im.crop(box);raw.save(out/(r['key']+'_detail_raw.png'))
        overlay=raw.copy();d=ImageDraw.Draw(overlay)
        for inst in r['instances']:
            pts=[(x*w-x0,y*h-y0) for x,y in inst['polygon_normalized']];d.line(pts+[pts[0]],fill='#ff5030',width=2)
        overlay.save(out/(r['key']+'_detail_overlay.png'))
        manifest.append(dict(key=r['key'],crop_xyxy_pixels=box,resize=False,focus_line=1,limitations='first polygon center crop is not an exhaustive whole-image pixel audit',source_image_sha256=r['image_sha256'],source_label_sha256=r['label_sha256']))
(out/'view_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
