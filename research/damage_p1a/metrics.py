"""Native-pixel confusion aggregation; all undefined denominators are None."""
import numpy as np


def ratio(a,b):return float(a/b) if b else None


class Metrics:
    def __init__(self):self.rows=[]
    def add(self,image_id,probability,target,valid,threshold=0.5):
        if not (probability.shape==target.shape==valid.shape) or target.shape[0]!=2:raise ValueError('shape mismatch')
        if not np.isfinite(probability).all() or np.any((probability<0)|(probability>1)):raise ValueError('invalid probability')
        prediction=probability>=threshold
        rows=[]
        for c in range(2):
            p,g,v=prediction[c],target[c].astype(bool),valid[c].astype(bool)
            tp=int((p&g&v).sum());fp=int((p&~g&v).sum());fn=int((~p&g&v).sum())
            rows.append({'image_id':image_id,'channel':c,'valid_pixels':int(v.sum()),'gt_pixels':tp+fn,
                'tp':tp,'fp':fp,'fn':fn,'iou':ratio(tp,tp+fp+fn),'dice':ratio(2*tp,2*tp+fp+fn),
                'precision':ratio(tp,tp+fp),'recall':ratio(tp,tp+fn)})
        self.rows.extend(rows)
        return rows
    def summary(self):
        classes=[]
        for c in range(2):
            rows=[r for r in self.rows if r['channel']==c and r['valid_pixels']]
            tp,fp,fn=(sum(r[k] for r in rows) for k in ('tp','fp','fn'))
            negative=[r for r in rows if r['gt_pixels']==0]
            macro=[r['iou'] for r in rows if r['iou'] is not None]
            classes.append({'channel':c,'tp':tp,'fp':fp,'fn':fn,'iou':ratio(tp,tp+fp+fn),
                'dice':ratio(2*tp,2*tp+fp+fn),'precision':ratio(tp,tp+fp),'recall':ratio(tp,tp+fn),
                'macro_image_iou':float(np.mean(macro)) if macro else None,'macro_image_count':len(macro),
                'known_images':len(rows),'negative_images':len(negative),
                'negative_fp_images':sum(r['fp']>0 for r in negative),
                'negative_fp_image_rate':ratio(sum(r['fp']>0 for r in negative),len(negative)),
                'negative_mean_fp_area_fraction':float(np.mean([r['fp']/r['valid_pixels'] for r in negative])) if negative else None})
        # A missing GT class cannot masquerade as a supported two-class task.
        miou=float(np.mean([c['iou'] for c in classes])) if all(c['tp']+c['fn']>0 for c in classes) else None
        return {'classes':classes,'foreground_miou':miou,'scope':'NATIVE_PIXEL_MICRO_PER_CLASS_NO_BACKGROUND'}
