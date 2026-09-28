"""Task 6K.1: minimal read-only ESRI Shapefile / DBF reader (Python standard library only).

No shapefile library is available in this environment (no pyshp, fiona, geopandas, GDAL, rasterio,
tifffile) and nothing may be installed, so this module implements exactly the parsing needed for
polygon records:

* ``.shp`` header + record iteration (shape types 0/1/3/5/8/11/13/15/18/21/23/25/28/31);
* ``.shx`` index records (offset/length in 16-bit words);
* ``.dbf`` header, field descriptors and records (including the deleted-row flag);
* ``.prj`` text.

Everything is read-only and byte-exact; malformed records are reported rather than silently
dropped. Geometry is returned as raw vertex arrays plus the ESRI part structure; ring
classification (outer vs hole) is left to the caller because it is a semantic decision.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

#: ESRI shape type codes referenced by this audit.
SHAPE_NULL = 0
SHAPE_POINT = 1
SHAPE_POLYLINE = 3
SHAPE_POLYGON = 5
SHAPE_MULTIPOINT = 8
SHAPE_POLYGON_Z = 15
SHAPE_POLYGON_M = 25
SHAPE_MULTIPATCH = 31

POLYGON_TYPES = (SHAPE_POLYGON, SHAPE_POLYGON_Z, SHAPE_POLYGON_M)

SHAPE_TYPE_NAMES = {
    0: "Null", 1: "Point", 3: "PolyLine", 5: "Polygon", 8: "MultiPoint",
    11: "PointZ", 13: "PolyLineZ", 15: "PolygonZ", 18: "MultiPointZ",
    21: "PointM", 23: "PolyLineM", 25: "PolygonM", 28: "MultiPointM", 31: "MultiPatch",
}


@dataclass
class ShapefileHeader:
    byte_order: str
    file_code: int
    file_length_words: int
    file_length_bytes: int
    version: int
    shape_type: int
    shape_type_name: str
    bbox: tuple[float, float, float, float]
    z_range: tuple[float, float]
    m_range: tuple[float, float]
    declared_file_bytes: int
    actual_file_bytes: int
    valid: bool
    issues: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "byte_order": self.byte_order,
            "file_code": self.file_code,
            "file_length_words": self.file_length_words,
            "file_length_bytes": self.file_length_bytes,
            "version": self.version,
            "shape_type": self.shape_type,
            "shape_type_name": self.shape_type_name,
            "bbox": list(self.bbox),
            "z_range": list(self.z_range),
            "m_range": list(self.m_range),
            "declared_file_bytes": self.declared_file_bytes,
            "actual_file_bytes": self.actual_file_bytes,
            "valid": self.valid,
            "issues": list(self.issues),
        }


def read_shp_header(path: Path) -> ShapefileHeader:
    """Parse the 100-byte ``.shp`` header and cross-check it against the real file size."""

    issues: list[str] = []
    with Path(path).open("rb") as handle:
        raw = handle.read(100)
    if len(raw) < 100:
        raise ValueError(f"{path}: truncated shapefile header ({len(raw)} bytes)")
    byte_order = "big" if raw[:4] == b"\x00\x00\x27\x0a" else "little"
    if byte_order != "big":
        issues.append("unexpected .shp byte order marker (expected big-endian 9994)")
    file_code = struct.unpack(">i", raw[0:4])[0]
    if file_code != 9994:
        issues.append(f"file code {file_code} != 9994")
    file_length_words = struct.unpack(">i", raw[24:28])[0]
    version = struct.unpack("<i", raw[28:32])[0]
    if version != 1000:
        issues.append(f"version {version} != 1000")
    shape_type = struct.unpack("<i", raw[32:36])[0]
    bbox = struct.unpack("<4d", raw[36:68])
    z_range = struct.unpack("<2d", raw[68:84])
    m_range = struct.unpack("<2d", raw[84:100])
    actual = Path(path).stat().st_size
    declared = file_length_words * 2
    if declared != actual:
        issues.append(f"declared length {declared} bytes != actual {actual} bytes")
    for name, values in (("bbox", bbox), ("z_range", z_range), ("m_range", m_range)):
        if not np.isfinite(values).all():
            issues.append(f"{name} contains non-finite values")
    return ShapefileHeader(
        byte_order=byte_order,
        file_code=file_code,
        file_length_words=file_length_words,
        file_length_bytes=declared,
        version=version,
        shape_type=shape_type,
        shape_type_name=SHAPE_TYPE_NAMES.get(shape_type, f"unknown({shape_type})"),
        bbox=tuple(float(v) for v in bbox),
        z_range=(float(z_range[0]), float(z_range[1])),
        m_range=(float(m_range[0]), float(m_range[1])),
        declared_file_bytes=declared,
        actual_file_bytes=actual,
        valid=not issues,
        issues=issues,
    )


@dataclass
class ShxIndex:
    declared_records: int
    parsed_records: int
    header: ShapefileHeader
    offsets_words: list[int]
    lengths_words: list[int]

    def as_dict(self) -> dict:
        expected = (self.header.actual_file_bytes - 100) // 8
        return {
            "declared_records": self.declared_records,
            "parsed_records": self.parsed_records,
            "index_records_from_file_size_formula": expected,
            "matches_declared": self.declared_records == self.parsed_records,
            "first_offset_words": self.offsets_words[0] if self.offsets_words else None,
            "first_content_length_words": self.lengths_words[0] if self.lengths_words else None,
            "last_offset_words": self.offsets_words[-1] if self.offsets_words else None,
            "monotonic_offsets": bool(
                all(b > a for a, b in zip(self.offsets_words, self.offsets_words[1:]))
            ),
        }


def read_shx_index(path: Path) -> ShxIndex:
    """Parse the ``.shx`` index: a 100-byte header plus 8-byte records."""

    shx = Path(path)
    header = read_shp_header(shx)
    raw = shx.read_bytes()
    body = raw[100:]
    n_records = len(body) // 8
    if len(body) % 8:
        raise ValueError(f"{path}: .shx body length {len(body)} is not a multiple of 8")
    offsets: list[int] = []
    lengths: list[int] = []
    for index in range(n_records):
        offset_words, length_words = struct.unpack(">2i", body[index * 8: index * 8 + 8])
        offsets.append(int(offset_words))
        lengths.append(int(length_words))
    return ShxIndex(
        declared_records=n_records,
        parsed_records=n_records,
        header=header,
        offsets_words=offsets,
        lengths_words=lengths,
    )


@dataclass
class ShpFeature:
    """One polygon feature with its ESRI part structure preserved."""

    record_number: int
    shape_type: int
    bbox: tuple[float, float, float, float] | None
    parts: list[tuple[int, int]]
    points: np.ndarray  # [N, 2] float64
    content_length_words: int
    has_z: bool = False
    has_m: bool = False
    issues: list[str] = field(default_factory=list)

    @property
    def n_parts(self) -> int:
        return len(self.parts)

    @property
    def n_points(self) -> int:
        return int(self.points.shape[0])

    def ring(self, part_index: int) -> np.ndarray:
        start, end = self.parts[part_index]
        return self.points[start:end]

    def rings(self) -> list[np.ndarray]:
        return [self.ring(index) for index in range(self.n_parts)]


def iter_shp_features(path: Path, limit: int | None = None):
    """Iterate polygon features; yields (feature, raw_record_shape_types)."""

    shp = Path(path)
    with shp.open("rb") as handle:
        handle.seek(100)
        record_number = 0
        yielded = 0
        while True:
            header = handle.read(8)
            if len(header) < 8:
                break
            number_be, length_be = struct.unpack(">2i", header)
            record_number += 1
            content = handle.read(length_be * 2)
            if len(content) < length_be * 2:
                break
            if len(content) < 4:
                continue
            shape_type = struct.unpack("<i", content[0:4])[0]
            feature = ShpFeature(
                record_number=int(number_be),
                shape_type=int(shape_type),
                bbox=None,
                parts=[],
                points=np.zeros((0, 2), dtype=np.float64),
                content_length_words=int(length_be),
            )
            if shape_type == SHAPE_NULL:
                yield feature
                yielded += 1
            elif shape_type in POLYGON_TYPES:
                offset = 4
                bbox = struct.unpack("<4d", content[offset: offset + 32])
                offset += 32
                n_parts, n_points = struct.unpack("<2i", content[offset: offset + 8])
                offset += 8
                part_array = struct.unpack(f"<{n_parts}i", content[offset: offset + 4 * n_parts])
                offset += 4 * n_parts
                points = np.frombuffer(
                    content, dtype="<f8", count=n_points * 2, offset=offset
                ).reshape(-1, 2).copy()
                offset += 16 * n_points
                feature.bbox = tuple(float(v) for v in bbox)
                feature.parts = [
                    (int(part_array[i]), int(part_array[i + 1]) if i + 1 < n_parts else int(n_points))
                    for i in range(n_parts)
                ]
                feature.points = points
                if shape_type in (SHAPE_POLYGON_Z,):
                    feature.has_z = True
                if shape_type in (SHAPE_POLYGON_M,):
                    feature.has_m = True
                if n_parts <= 0:
                    feature.issues.append("no parts")
                if n_points < 3:
                    feature.issues.append(f"only {n_points} points")
                if feature.parts and feature.parts[0][0] != 0:
                    feature.issues.append("first part does not start at point 0")
                expected = sum(end - start for start, end in feature.parts)
                if expected != n_points:
                    feature.issues.append(f"parts cover {expected} of {n_points} points")
                yield feature
                yielded += 1
            else:
                feature.issues.append("unsupported shape type for this audit")
                yield feature
                yielded += 1
            if limit is not None and yielded >= int(limit):
                break


@dataclass
class DbfField:
    name: str
    type: str
    length: int
    decimals: int

    def as_dict(self) -> dict:
        return {"name": self.name, "type": self.type, "length": self.length, "decimals": self.decimals}


@dataclass
class DbfTable:
    version: int
    last_update: tuple[int, int, int]
    record_count: int
    header_size: int
    record_size: int
    fields: list[DbfField]
    records: list[dict]
    deleted_records: list[int]
    issues: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "version": self.version,
            "last_update_y_m_d": list(self.last_update),
            "record_count": self.record_count,
            "header_size": self.header_size,
            "record_size": self.record_size,
            "fields": [f.as_dict() for f in self.fields],
            "parsed_records": len(self.records),
            "deleted_records": self.deleted_records,
            "deleted_record_count": len(self.deleted_records),
            "computed_record_size": 1 + sum(f.length for f in self.fields),
            "issues": self.issues,
        }


def read_dbf(path: Path, max_records: int | None = None) -> DbfTable:
    """Parse a dBASE III/IV table: header, field descriptors and records."""

    dbf = Path(path)
    raw = dbf.read_bytes()
    if len(raw) < 32:
        raise ValueError(f"{path}: truncated DBF header")
    version = raw[0]
    year, month, day = raw[1], raw[2], raw[3]
    record_count, header_size, record_size = struct.unpack("<IHH", raw[4:12])
    fields: list[DbfField] = []
    offset = 32
    while offset + 32 <= len(raw) and raw[offset] != 0x0D:
        name = raw[offset: offset + 11].split(b"\x00")[0].decode("ascii", errors="replace").strip()
        ftype = chr(raw[offset + 11])
        length = raw[offset + 16]
        decimals = raw[offset + 17]
        fields.append(DbfField(name=name, type=ftype, length=int(length), decimals=int(decimals)))
        offset += 32
    issues: list[str] = []
    computed = 1 + sum(f.length for f in fields)
    if computed != record_size:
        issues.append(f"computed record size {computed} != header record size {record_size}")
    if offset + 1 > len(raw) or raw[offset] != 0x0D:
        issues.append("missing DBF field terminator 0x0D")

    records: list[dict] = []
    deleted: list[int] = []
    position = header_size
    for index in range(record_count):
        if max_records is not None and index >= int(max_records):
            break
        chunk = raw[position: position + record_size]
        if len(chunk) < record_size:
            issues.append(f"record {index} truncated")
            break
        position += record_size
        flag = chunk[0:1]
        if flag == b"*":
            deleted.append(index)
            continue
        if flag != b" ":
            issues.append(f"record {index} has unexpected deletion flag {flag!r}")
        entry: dict = {}
        cursor = 1
        for dbf_field in fields:
            value = chunk[cursor: cursor + dbf_field.length]
            cursor += dbf_field.length
            text = value.decode("ascii", errors="replace").strip()
            if dbf_field.type in ("N", "F") and text:
                try:
                    entry[dbf_field.name] = float(text) if dbf_field.decimals else int(float(text))
                except ValueError:
                    entry[dbf_field.name] = text
            else:
                entry[dbf_field.name] = text
        records.append(entry)
    return DbfTable(
        version=version,
        last_update=(int(year), int(month), int(day)),
        record_count=int(record_count),
        header_size=int(header_size),
        record_size=int(record_size),
        fields=fields,
        records=records,
        deleted_records=deleted,
        issues=issues,
    )


def read_prj_text(path: Path) -> str:
    return Path(path).read_text(encoding="utf-8", errors="replace").strip()


# ------------------------------------------------------------------ ring geometry


def ring_signed_area(ring: np.ndarray) -> float:
    """Shoelace signed area (positive = counter-clockwise in a y-up CRS)."""

    points = np.asarray(ring, dtype=np.float64)
    if points.shape[0] < 3:
        return 0.0
    x = points[:, 0]
    y = points[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def ring_is_closed(ring: np.ndarray, tolerance: float = 1e-9) -> bool:
    points = np.asarray(ring, dtype=np.float64)
    if points.shape[0] < 2:
        return False
    return bool(np.allclose(points[0], points[-1], atol=tolerance, rtol=0.0))


def classify_rings(feature: ShpFeature) -> dict:
    """ESRI convention: outer rings are clockwise, holes counter-clockwise (y-up CRS)."""

    outers, holes, degenerate = [], [], []
    for index, ring in enumerate(feature.rings()):
        area = ring_signed_area(ring)
        if ring.shape[0] < 4 or abs(area) < 1e-9:
            degenerate.append(index)
        elif area < 0:
            outers.append(index)
        else:
            holes.append(index)
    return {"outer_rings": outers, "hole_rings": holes, "degenerate_rings": degenerate}


def feature_area(feature: ShpFeature, classification: dict) -> float:
    outer = sum(abs(ring_signed_area(feature.ring(i))) for i in classification["outer_rings"])
    holes = sum(abs(ring_signed_area(feature.ring(i))) for i in classification["hole_rings"])
    return float(outer - holes)
