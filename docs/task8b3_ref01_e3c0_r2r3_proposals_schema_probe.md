# Task 8B.3-REF01-E3C0-R2-R3 — proposals.json Schema Probe

- starting HEAD = `fa9ede22f9bee640a6211f0dfd807439d0e8bc57`
- exact files read = 4
- detector/model calls = 0
- proposal regeneration = NO
- actual replay = NO
- external write = NO
- broad search = NO

## Per-case top-level schema

### right
- path = `C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\inference\output\diagnostics\1010\proposals.json`
- bytes = `2411`
- sha256 = `2b09041ee86d5e7d71c05c32f2cea8669bdfbb5b7f96cda33d5e12c99a0ac45b`
- top-level type = `dict`
- top-level length = `2`
- top-level keys = `['count', 'items']`
- top-level first-item type = `None`
- top-level first-item keys = `None`

Child summaries:
```json
{
  "count": {
    "type": "int",
    "preview": 6
  },
  "items": {
    "type": "list",
    "length": 6,
    "first_item_type": "dict",
    "first_item_keys": [
      "proposal_id",
      "source_tile_id",
      "confidence",
      "mask_area",
      "global_bbox",
      "centroid",
      "touches_image_border",
      "border_clearance",
      "bbox_extent_ratio",
      "raw_index"
    ]
  }
}
```

### left
- path = `C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\inference\output\diagnostics\1003\proposals.json`
- bytes = `21263`
- sha256 = `62865e42b4d92077ce336522b97f33931b351d46ab4ec7443fb1afe2dfd798a6`
- top-level type = `dict`
- top-level length = `2`
- top-level keys = `['count', 'items']`
- top-level first-item type = `None`
- top-level first-item keys = `None`

Child summaries:
```json
{
  "count": {
    "type": "int",
    "preview": 53
  },
  "items": {
    "type": "list",
    "length": 53,
    "first_item_type": "dict",
    "first_item_keys": [
      "proposal_id",
      "source_tile_id",
      "confidence",
      "mask_area",
      "global_bbox",
      "centroid",
      "touches_image_border",
      "border_clearance",
      "bbox_extent_ratio",
      "raw_index"
    ]
  }
}
```

### above
- path = `C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\inference\output\diagnostics\1008\proposals.json`
- bytes = `3631`
- sha256 = `257297112fb2fe35f2225725e8c0a0e79338dcde10f9f4dbe20641c8fbb85e0b`
- top-level type = `dict`
- top-level length = `2`
- top-level keys = `['count', 'items']`
- top-level first-item type = `None`
- top-level first-item keys = `None`

Child summaries:
```json
{
  "count": {
    "type": "int",
    "preview": 9
  },
  "items": {
    "type": "list",
    "length": 9,
    "first_item_type": "dict",
    "first_item_keys": [
      "proposal_id",
      "source_tile_id",
      "confidence",
      "mask_area",
      "global_bbox",
      "centroid",
      "touches_image_border",
      "border_clearance",
      "bbox_extent_ratio",
      "raw_index"
    ]
  }
}
```

### below
- path = `C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\inference\output\diagnostics\1009\proposals.json`
- bytes = `2412`
- sha256 = `7c1a45690c860910b5626fc603fe942cc0497157bf6fc5d1645657029c167752`
- top-level type = `dict`
- top-level length = `2`
- top-level keys = `['count', 'items']`
- top-level first-item type = `None`
- top-level first-item keys = `None`

Child summaries:
```json
{
  "count": {
    "type": "int",
    "preview": 6
  },
  "items": {
    "type": "list",
    "length": 6,
    "first_item_type": "dict",
    "first_item_keys": [
      "proposal_id",
      "source_tile_id",
      "confidence",
      "mask_area",
      "global_bbox",
      "centroid",
      "touches_image_border",
      "border_clearance",
      "bbox_extent_ratio",
      "raw_index"
    ]
  }
}
```

## Outcome
- `PROPOSALS_JSON_SCHEMA_CAPTURED`
- readiness was NOT determined
- NEXT was NOT executed
