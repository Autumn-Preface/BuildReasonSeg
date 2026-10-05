# Task 8B.3-REF01-E3C2 — Selection Scalar Forensics

## Scope
- locked proposal records only
- detector/model calls = 0
- proposal regeneration = NO
- selector repair implementation = NO
- external write = NO
- product source change = NO

## Predeclared threshold-free ranking rules
- `AREA_FIRST`
- `CONFIDENCE_FIRST`
- `BBOX_AREA_FIRST`
- `FILL_RATIO_FIRST`
- `AREA_X_CONF`
- `BBOX_AREA_X_CONF`

## Case results

### right
- current repaired selected = `1`
- best-IoU id = `1`
- best IoU = `0.558869701727`
- extent exceptions = `[]`
- selected-minus-best = `{'mask_area': 0, 'confidence': 0.0, 'bbox_area': 0, 'fill_ratio': 0.0, 'bbox_extent_ratio': 0.0}`
- best Pareto dominators (area, confidence) = `[]`
- best Pareto dominators (bbox area, confidence) = `[2]`

| rule | selected | best-IoU rank | matches best |
|---|---:|---:|---|
| AREA_FIRST | 1 | 1 | True |
| CONFIDENCE_FIRST | 2 | 3 | False |
| BBOX_AREA_FIRST | 2 | 2 | False |
| FILL_RATIO_FIRST | 1 | 1 | True |
| AREA_X_CONF | 1 | 1 | True |
| BBOX_AREA_X_CONF | 2 | 2 | False |

### left
- current repaired selected = `14`
- best-IoU id = `30`
- best IoU = `0.650067294751`
- extent exceptions = `[]`
- selected-minus-best = `{'mask_area': 743, 'confidence': -0.0009241700172424316, 'bbox_area': 1358, 'fill_ratio': 0.0039265835270908545, 'bbox_extent_ratio': 0.015625}`
- best Pareto dominators (area, confidence) = `[]`
- best Pareto dominators (bbox area, confidence) = `[]`

| rule | selected | best-IoU rank | matches best |
|---|---:|---:|---|
| AREA_FIRST | 14 | 3 | False |
| CONFIDENCE_FIRST | 21 | 3 | False |
| BBOX_AREA_FIRST | 43 | 4 | False |
| FILL_RATIO_FIRST | 2 | 26 | False |
| AREA_X_CONF | 14 | 2 | False |
| BBOX_AREA_X_CONF | 14 | 3 | False |

### above
- current repaired selected = `5`
- best-IoU id = `5`
- best IoU = `0.903250188964`
- extent exceptions = `[5, 6]`
- selected-minus-best = `{'mask_area': 0, 'confidence': 0.0, 'bbox_area': 0, 'fill_ratio': 0.0, 'bbox_extent_ratio': 0.0}`
- best Pareto dominators (area, confidence) = `[]`
- best Pareto dominators (bbox area, confidence) = `[]`

| rule | selected | best-IoU rank | matches best |
|---|---:|---:|---|
| AREA_FIRST | 5 | 1 | True |
| CONFIDENCE_FIRST | 5 | 1 | True |
| BBOX_AREA_FIRST | 5 | 1 | True |
| FILL_RATIO_FIRST | 2 | 6 | False |
| AREA_X_CONF | 5 | 1 | True |
| BBOX_AREA_X_CONF | 5 | 1 | True |

### below
- current repaired selected = `1`
- best-IoU id = `2`
- best IoU = `0.616549685999`
- extent exceptions = `[]`
- selected-minus-best = `{'mask_area': 312, 'confidence': 0.014322161674499512, 'bbox_area': 66, 'fill_ratio': 0.108817287285356, 'bbox_extent_ratio': 0.00390625}`
- best Pareto dominators (area, confidence) = `[1]`
- best Pareto dominators (bbox area, confidence) = `[1]`

| rule | selected | best-IoU rank | matches best |
|---|---:|---:|---|
| AREA_FIRST | 1 | 2 | False |
| CONFIDENCE_FIRST | 1 | 2 | False |
| BBOX_AREA_FIRST | 1 | 2 | False |
| FILL_RATIO_FIRST | 1 | 2 | False |
| AREA_X_CONF | 1 | 2 | False |
| BBOX_AREA_X_CONF | 1 | 2 | False |

## Global diagnostic
- globally perfect predeclared rules on locked four = `[]`
- no scalar repair rule is selected by this task
- no threshold is tuned by this task

## Status / NEXT
- REF-01 = `ACTIVE`
- PROP-01 = `PROP01_OPEN_ENGINEERING_DEFECT`
- outcome = `REF01_SELECTION_SCALAR_FORENSICS_COMPLETE`
- NEXT = `CHATGPT_SELECTION_REPAIR_DECISION`
