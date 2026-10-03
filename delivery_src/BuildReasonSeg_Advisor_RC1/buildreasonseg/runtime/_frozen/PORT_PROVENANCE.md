# Frozen port provenance (Task 8B)

source: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\buildreasonseg_mvp`

target: `C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\buildreasonseg\runtime\_frozen\mvp`

copy mode: verbatim file copy, with the only textual change being the import prefix
`buildreasonseg_mvp.` -> `buildreasonseg.runtime._frozen.mvp.`

The delivery runtime therefore executes the identical frozen implementation for the relation
fields, the D-B1 decoder, the SAM2 encoder bridge, the reference resolver and the ProgramHead.
Path constants are redirected to the delivery model package by
`buildreasonseg/runtime/frozen_paths.py` before any of these modules is imported.
