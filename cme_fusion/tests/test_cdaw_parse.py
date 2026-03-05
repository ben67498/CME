from pathlib import Path

from src.parse_cdaw import parse_univ_all


def test_parse_cdaw(tmp_path: Path):
    text = """
# header
2011/01/20 01:24 123 45 360 1200 halo
2011/01/20 03:00 130 50 120 900
"""
    p = tmp_path / "univ_all.txt"
    p.write_text(text)
    df = parse_univ_all(p)
    assert len(df) == 2
    assert "first_c2_datetime_utc" in df.columns
