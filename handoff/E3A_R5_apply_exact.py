from __future__ import annotations

import base64
import hashlib
import subprocess
import sys
from pathlib import Path

REPO = Path(r"C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg")
EXTERNAL = Path(r"C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1")
DETECTOR = REPO / "delivery_src" / "BuildReasonSeg_Advisor_RC1" / "buildreasonseg" / "runtime" / "detector.py"
TEST = REPO / "delivery_src" / "BuildReasonSeg_Advisor_RC1" / "tests" / "test_task8b_runtime.py"
EXTERNAL_DETECTOR = EXTERNAL / "buildreasonseg" / "runtime" / "detector.py"

EXPECTED_BRANCH = "fix/task8b3-ref01-eligibility-repair-impl"
EXPECTED_HEAD = "34d1a263ac91ada6ef517fc54a99c19463e9ddd3"
EXPECTED_EXTERNAL_SHA = "82531dc3b758cd8a864f77d0fa97e4132113cb46eb5a33a11f483bf51bc0a738"

DETECTOR_REPLACEMENT_B64 = "ZGVmIGVsaWdpYmxlX3Byb3Bvc2Fscyhwcm9wb3NhbHM6IGxpc3RbR2xvYmFsUHJvcG9zYWxdLCAqLCBmYW1pbHk6IHN0ciA9ICJsYXJnZXN0IgogICAgICAgICAgICAgICAgICAgICAgICkgLT4gbGlzdFtHbG9iYWxQcm9wb3NhbF06CiAgICByZXR1cm4gW3Byb3Bvc2FsIGZvciBwcm9wb3NhbCBpbiBwcm9wb3NhbHMgaWYgZWxpZ2libGUocHJvcG9zYWwsIGZhbWlseT1mYW1pbHkpXQoKCmRlZiBfcmVmZXJlbmNlX3JhbmsocHJvcG9zYWw6IEdsb2JhbFByb3Bvc2FsKSAtPiB0dXBsZVtpbnQsIGZsb2F0LCBpbnRdOgogICAgcmV0dXJuICgtcHJvcG9zYWwubWFza19hcmVhLCAtcHJvcG9zYWwuY29uZmlkZW5jZSwgcHJvcG9zYWwucHJvcG9zYWxfaWQpCgoKZGVmIF9sYXJnZXN0X3JlZmVyZW5jZV9jYW5kaWRhdGVzX3dpdGhfZXh0ZW50X2V4Y2VwdGlvbigKICAgICAgICBwcm9wb3NhbHM6IGxpc3RbR2xvYmFsUHJvcG9zYWxdKSAtPiBsaXN0W0dsb2JhbFByb3Bvc2FsXToKICAgICIiIkZyb3plbiBiYXNlIGNhbmRpZGF0ZXMgcGx1cyB0aGUgUkMxIGxhcmdlc3Qtb25seSBleHRlbnQtZG9taW5hbmNlIGV4Y2VwdGlvbi4iIiIKCiAgICBiYXNlX2NhbmRpZGF0ZXMgPSBlbGlnaWJsZV9wcm9wb3NhbHMocHJvcG9zYWxzLCBmYW1pbHk9Imxhcmdlc3QiKQogICAgaWYgbm90IGJhc2VfY2FuZGlkYXRlczoKICAgICAgICByZXR1cm4gW10KCiAgICBiYXNlbGluZSA9IHNvcnRlZChiYXNlX2NhbmRpZGF0ZXMsIGtleT1fcmVmZXJlbmNlX3JhbmspWzBdCiAgICBleGNlcHRpb25zID0gWwogICAgICAgIHByb3Bvc2FsIGZvciBwcm9wb3NhbCBpbiBwcm9wb3NhbHMKICAgICAgICBpZiBwcm9wb3NhbC5tYXNrX2FyZWEgPiAwCiAgICAgICAgYW5kIHByb3Bvc2FsLm1hc2tfY3JvcC5hbnkoKQogICAgICAgIGFuZCBub3QgcHJvcG9zYWwudG91Y2hlc19pbWFnZV9ib3JkZXIKICAgICAgICBhbmQgcHJvcG9zYWwuYmJveF9leHRlbnRfcmF0aW8gPiBNRVJHRV9CQk9YX0VYVEVOVF9SQVRJT19NQVgKICAgICAgICBhbmQgcHJvcG9zYWwubWFza19hcmVhID4gYmFzZWxpbmUubWFza19hcmVhCiAgICAgICAgYW5kIHByb3Bvc2FsLmNvbmZpZGVuY2UgPiBiYXNlbGluZS5jb25maWRlbmNlCiAgICBdCiAgICByZXR1cm4gYmFzZV9jYW5kaWRhdGVzICsgZXhjZXB0aW9ucwoKCmRlZiBzZWxlY3RfcmVmZXJlbmNlKHByb3Bvc2FsczogbGlzdFtHbG9iYWxQcm9wb3NhbF0sICosIGZhbWlseTogc3RyID0gImxhcmdlc3QiKSAtPiBHbG9iYWxQcm9wb3NhbCB8IE5vbmU6CiAgICAiIiJTZWxlY3QgYSByZWZlcmVuY2Ugd2l0aCB0aGUgZnJvemVuIGJhc2UgcnVsZSBhbmQgdGhlIGxhcmdlc3Qtb25seSBSQzEgZXhjZXB0aW9uLiIiIgoKICAgIGlmIGZhbWlseSA9PSAibGFyZ2VzdCI6CiAgICAgICAgY2FuZGlkYXRlcyA9IF9sYXJnZXN0X3JlZmVyZW5jZV9jYW5kaWRhdGVzX3dpdGhfZXh0ZW50X2V4Y2VwdGlvbihwcm9wb3NhbHMpCiAgICBlbHNlOgogICAgICAgIGNhbmRpZGF0ZXMgPSBlbGlnaWJsZV9wcm9wb3NhbHMocHJvcG9zYWxzLCBmYW1pbHk9ZmFtaWx5KQogICAgaWYgbm90IGNhbmRpZGF0ZXM6CiAgICAgICAgcmV0dXJuIE5vbmUKICAgIHJldHVybiBzb3J0ZWQoY2FuZGlkYXRlcywga2V5PV9yZWZlcmVuY2VfcmFuaylbMF0KCgo="
TEST_INSERT_B64 = "ZGVmIF9yZWZlcmVuY2VfcmVjdChoZWlnaHQ6IGludCwgd2lkdGg6IGludCwgY29uZmlkZW5jZTogZmxvYXQsIHByb3Bvc2FsX2lkOiBpbnQsCiAgICAgICAgICAgICAgICAgICAgKiwgdG9wOiBpbnQgPSAxMDAsIGxlZnQ6IGludCA9IDEwMCkgLT4gR2xvYmFsUHJvcG9zYWw6CiAgICBtYXNrID0gbnAuemVyb3MoKDUxMiwgNTEyKSwgZHR5cGU9Ym9vbCkKICAgIG1hc2tbdG9wOnRvcCArIGhlaWdodCwgbGVmdDpsZWZ0ICsgd2lkdGhdID0gVHJ1ZQogICAgcmV0dXJuIF9wcm9wb3NhbChtYXNrLCBjb25maWRlbmNlLCBmInJ7cHJvcG9zYWxfaWR9IiwgcHJvcG9zYWxfaWQsIHByb3Bvc2FsX2lkLAogICAgICAgICAgICAgICAgICAgICBwcm9wb3NhbF9pZD1wcm9wb3NhbF9pZCkKCgpkZWYgdGVzdF9sYXJnZXN0X2V4dGVudF9kb21pbmFuY2VfZXhjZXB0aW9uX3NlbGVjdHNfc3RyaWN0bHlfZG9taW5hbnRfY2FuZGlkYXRlKCkgLT4gTm9uZToKICAgIGJhc2VsaW5lID0gX3JlZmVyZW5jZV9yZWN0KDgwLCA4MCwgMC42MCwgMSkKICAgIGRvbWluYW50ID0gX3JlZmVyZW5jZV9yZWN0KDEyMCwgMTIwLCAwLjYxLCAyLCB0b3A9MjUwLCBsZWZ0PTI1MCkKICAgIGFzc2VydCBlbGlnaWJsZShiYXNlbGluZSwgZmFtaWx5PSJsYXJnZXN0IikgaXMgVHJ1ZQogICAgYXNzZXJ0IGVsaWdpYmxlKGRvbWluYW50LCBmYW1pbHk9Imxhcmdlc3QiKSBpcyBGYWxzZQogICAgYXNzZXJ0IGRldGVjdG9yLmVsaWdpYmxlX3Byb3Bvc2FscyhbYmFzZWxpbmUsIGRvbWluYW50XSwgZmFtaWx5PSJsYXJnZXN0IikgPT0gW2Jhc2VsaW5lXQogICAgYXNzZXJ0IHNlbGVjdF9yZWZlcmVuY2UoW2Jhc2VsaW5lLCBkb21pbmFudF0sIGZhbWlseT0ibGFyZ2VzdCIpIGlzIGRvbWluYW50CgoKZGVmIHRlc3RfbGFyZ2VzdF9leHRlbnRfZXhjZXB0aW9uX3JlamVjdHNfbG93ZXJfY29uZmlkZW5jZV9jYW5kaWRhdGUoKSAtPiBOb25lOgogICAgYmFzZWxpbmUgPSBfcmVmZXJlbmNlX3JlY3QoODAsIDgwLCAwLjYwLCAxKQogICAgY2FuZGlkYXRlID0gX3JlZmVyZW5jZV9yZWN0KDEyMCwgMTIwLCAwLjU5LCAyLCB0b3A9MjUwLCBsZWZ0PTI1MCkKICAgIGFzc2VydCBzZWxlY3RfcmVmZXJlbmNlKFtiYXNlbGluZSwgY2FuZGlkYXRlXSwgZmFtaWx5PSJsYXJnZXN0IikgaXMgYmFzZWxpbmUKCgpkZWYgdGVzdF9sYXJnZXN0X2V4dGVudF9leGNlcHRpb25fcmVxdWlyZXNfc3RyaWN0X2NvbmZpZGVuY2VfZ2FpbigpIC0+IE5vbmU6CiAgICBiYXNlbGluZSA9IF9yZWZlcmVuY2VfcmVjdCg4MCwgODAsIDAuNjAsIDEpCiAgICBjYW5kaWRhdGUgPSBfcmVmZXJlbmNlX3JlY3QoMTIwLCAxMjAsIDAuNjAsIDIsIHRvcD0yNTAsIGxlZnQ9MjUwKQogICAgYXNzZXJ0IHNlbGVjdF9yZWZlcmVuY2UoW2Jhc2VsaW5lLCBjYW5kaWRhdGVdLCBmYW1pbHk9Imxhcmdlc3QiKSBpcyBiYXNlbGluZQoKCmRlZiB0ZXN0X2xhcmdlc3RfZXh0ZW50X2V4Y2VwdGlvbl9yZXF1aXJlc19zdHJpY3RfYXJlYV9nYWluKCkgLT4gTm9uZToKICAgIGJhc2VsaW5lID0gX3JlZmVyZW5jZV9yZWN0KDgwLCA4MCwgMC42MCwgMSkKICAgIHNhbWVfYXJlYSA9IF9yZWZlcmVuY2VfcmVjdCg0MCwgMTYwLCAwLjk1LCAyLCB0b3A9MjUwLCBsZWZ0PTI1MCkKICAgIGFzc2VydCBiYXNlbGluZS5tYXNrX2FyZWEgPT0gc2FtZV9hcmVhLm1hc2tfYXJlYQogICAgYXNzZXJ0IGVsaWdpYmxlKHNhbWVfYXJlYSwgZmFtaWx5PSJsYXJnZXN0IikgaXMgRmFsc2UKICAgIGFzc2VydCBzZWxlY3RfcmVmZXJlbmNlKFtiYXNlbGluZSwgc2FtZV9hcmVhXSwgZmFtaWx5PSJsYXJnZXN0IikgaXMgYmFzZWxpbmUKCgpkZWYgdGVzdF9sYXJnZXN0X2V4dGVudF9leGNlcHRpb25fbmV2ZXJfYWRtaXRzX2JvcmRlcl9wcm9wb3NhbCgpIC0+IE5vbmU6CiAgICBiYXNlbGluZSA9IF9yZWZlcmVuY2VfcmVjdCg4MCwgODAsIDAuNjAsIDEpCiAgICBib3JkZXIgPSBfcmVmZXJlbmNlX3JlY3QoMTIwLCAxMjAsIDAuOTUsIDIsIHRvcD0wLCBsZWZ0PTI1MCkKICAgIGFzc2VydCBib3JkZXIudG91Y2hlc19pbWFnZV9ib3JkZXIgaXMgVHJ1ZQogICAgYXNzZXJ0IHNlbGVjdF9yZWZlcmVuY2UoW2Jhc2VsaW5lLCBib3JkZXJdLCBmYW1pbHk9Imxhcmdlc3QiKSBpcyBiYXNlbGluZQoKCmRlZiB0ZXN0X2xhcmdlc3RfZXh0ZW50X2V4Y2VwdGlvbl9yZXF1aXJlc19mcm96ZW5fYmFzZWxpbmUoKSAtPiBOb25lOgogICAgb25seV9leHRlbnRfdmlvbGF0aW9uID0gX3JlZmVyZW5jZV9yZWN0KDEyMCwgMTIwLCAwLjk1LCAxKQogICAgYXNzZXJ0IGVsaWdpYmxlKG9ubHlfZXh0ZW50X3Zpb2xhdGlvbiwgZmFtaWx5PSJsYXJnZXN0IikgaXMgRmFsc2UKICAgIGFzc2VydCBzZWxlY3RfcmVmZXJlbmNlKFtvbmx5X2V4dGVudF92aW9sYXRpb25dLCBmYW1pbHk9Imxhcmdlc3QiKSBpcyBOb25lCgoKZGVmIHRlc3RfbGFyZ2VzdF9leHRlbnRfZXhjZXB0aW9uX2tlZXBzX3Byb2R1Y3Rpb25fYXJlYV9vcmRlcigpIC0+IE5vbmU6CiAgICBiYXNlbGluZSA9IF9yZWZlcmVuY2VfcmVjdCg4MCwgODAsIDAuNjAsIDEpCiAgICBleGNlcHRpb25fYSA9IF9yZWZlcmVuY2VfcmVjdCgxMjAsIDEyMCwgMC43MCwgMiwgdG9wPTI1MCwgbGVmdD01MCkKICAgIGV4Y2VwdGlvbl9iID0gX3JlZmVyZW5jZV9yZWN0KDEzMCwgMTMwLCAwLjYxLCAzLCB0b3A9MjUwLCBsZWZ0PTI1MCkKICAgIGFzc2VydCBzZWxlY3RfcmVmZXJlbmNlKFtiYXNlbGluZSwgZXhjZXB0aW9uX2EsIGV4Y2VwdGlvbl9iXSwgZmFtaWx5PSJsYXJnZXN0IikgaXMgZXhjZXB0aW9uX2IKCgpkZWYgdGVzdF9sYXJnZXN0X2V4dGVudF9leGNlcHRpb25fZG9lc19ub3RfY2hhbmdlX3NtYWxsZXN0X2ZhbWlseSgpIC0+IE5vbmU6CiAgICBiYXNlbGluZSA9IF9yZWZlcmVuY2VfcmVjdCg4MCwgODAsIDAuNjAsIDEpCiAgICBleHRlbnRfdmlvbGF0aW9uID0gX3JlZmVyZW5jZV9yZWN0KDEyMCwgMTIwLCAwLjk1LCAyLCB0b3A9MjUwLCBsZWZ0PTI1MCkKICAgIGFzc2VydCBlbGlnaWJsZShleHRlbnRfdmlvbGF0aW9uLCBmYW1pbHk9InNtYWxsZXN0IikgaXMgRmFsc2UKICAgIGFzc2VydCBzZWxlY3RfcmVmZXJlbmNlKFtiYXNlbGluZSwgZXh0ZW50X3Zpb2xhdGlvbl0sIGZhbWlseT0ic21hbGxlc3QiKSBpcyBiYXNlbGluZQoKCg=="

START_SENTINEL = b'def eligible_proposals('
END_SENTINEL = b'def proposal_by_id('
TEST_MARKER = b'# ---------------------------------------------------------------- reasoning context'
TEST_SENTINEL = b'def test_largest_extent_dominance_exception_selects_strictly_dominant_candidate'


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_text(*args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(REPO), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def newline(raw: bytes) -> bytes:
    crlf = raw.count(b"\r\n")
    lf = raw.count(b"\n")
    if crlf and crlf == lf:
        return b"\r\n"
    if not crlf and lf:
        return b"\n"
    raise AssertionError(f"unsupported mixed newline state: CRLF={crlf}, LF={lf}")


def adapt_payload(payload_b64: str, nl: bytes) -> bytes:
    payload = base64.b64decode(payload_b64)
    assert b"\r" not in payload
    return payload.replace(b"\n", nl)


def main() -> int:
    assert git_text("branch", "--show-current") == EXPECTED_BRANCH
    assert git_text("rev-parse", "HEAD") == EXPECTED_HEAD
    assert EXTERNAL_DETECTOR.is_file()
    assert sha256(EXTERNAL_DETECTOR) == EXPECTED_EXTERNAL_SHA

    detector_raw = DETECTOR.read_bytes()
    test_raw = TEST.read_bytes()
    dnl = newline(detector_raw)
    tnl = newline(test_raw)

    assert detector_raw.count(START_SENTINEL) == 1
    assert detector_raw.count(END_SENTINEL) == 1
    assert detector_raw.count(b'def select_reference(') == 1
    assert detector_raw.count(b'def _reference_rank(') == 0
    assert detector_raw.count(b'def _largest_reference_candidates_with_extent_exception(') == 0

    start = detector_raw.index(START_SENTINEL)
    end = detector_raw.index(END_SENTINEL)
    assert start < end
    old_slice = detector_raw[start:end]
    assert b'candidates = eligible_proposals(proposals, family=family)' in old_slice

    detector_after = detector_raw[:start] + adapt_payload(DETECTOR_REPLACEMENT_B64, dnl) + detector_raw[end:]

    assert detector_after.count(b'def _reference_rank(') == 1
    assert detector_after.count(b'def _largest_reference_candidates_with_extent_exception(') == 1
    assert detector_after.count(b'proposal.mask_area > baseline.mask_area') == 1
    assert detector_after.count(b'proposal.confidence > baseline.confidence') == 1

    assert test_raw.count(TEST_MARKER) == 1
    assert test_raw.count(TEST_SENTINEL) == 0
    marker_pos = test_raw.index(TEST_MARKER)
    test_after = test_raw[:marker_pos] + adapt_payload(TEST_INSERT_B64, tnl) + test_raw[marker_pos:]
    assert test_after.count(b'def test_largest_extent_') == 8

    DETECTOR.write_bytes(detector_after)
    TEST.write_bytes(test_after)

    print("PATCHER_BRANCH_HEAD: PASS")
    print("EXTERNAL_DETECTOR_IDENTITY: PASS")
    print("DETECTOR_BOUNDARY_REPLACEMENT: PASS")
    print("TEST_MARKER_INSERTION: PASS")
    print("NEW_TEST_COUNT: 8")
    print("PATCHER_RESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
