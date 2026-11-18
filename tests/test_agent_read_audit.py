import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "agent_read_audit", Path(__file__).resolve().parents[1] / "scripts" / "agent_read_audit.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_only_error_results_count_as_failed_reads():
    # file text that happens to say "too large" is still a successful read (DESIGN.md (o))
    assert not audit.failed("401\tthe kernel matrix is too large to invert", None)
    assert not audit.failed([{"type": "text", "text": "1\tContext (the full lecture transcript):"}], None)
    assert audit.failed("File content (25014 tokens) exceeds maximum allowed tokens (25000).", None)
    assert audit.failed("<persisted-output>\nOutput too large (30.5KB).", None)
    assert audit.failed("1\tanything", True)
