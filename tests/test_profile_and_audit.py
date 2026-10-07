"""Scaffolding tests: profile + airspace block, ceiling sanity check,
automatic airport profile (stubbed airport database), check-data audit."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from proximity_pipeline import airspace, config, loading
from proximity_pipeline import goaround_adapter as ga


def test_kbna_profile_loads_both_configs(kbna):
    assert ga.config.AIRPORT_ICAO == "KBNA"
    assert ga.config.LOCAL_TZ == "America/Chicago"
    assert ga.config.PRESET == "air_carrier"
    assert config.AIRSPACE_CEILING_FT_MSL == 4600
    assert config.VOLUME_RADIUS_NM == 10
    assert config.AIRSPACE_CEILING_MODE == "auto"
    assert abs(airspace.charted_ceiling_ft_agl() - 4001.0) < 1e-6
    assert set(config.FLOWS) == {"north", "south"}
    # the vendored loader ignores the extra blocks (no error, no effect)
    assert "airspace" in kbna and "flows" in kbna


def test_ceiling_input_sanity(kbna):
    saved = config.AIRSPACE_CEILING_FT_MSL
    try:
        config.AIRSPACE_CEILING_FT_MSL = 599.0 + 500.0
        with pytest.raises(ValueError, match="above field"):
            airspace.check_ceiling_input()
        config.AIRSPACE_CEILING_FT_MSL = 599.0 + 15000.0
        with pytest.raises(ValueError):
            airspace.check_ceiling_input()
    finally:
        config.AIRSPACE_CEILING_FT_MSL = saved


def test_computation_volume_is_the_widest(kbna):
    # encounters are computed once in the widest volume any sweep can ask
    # for, plus the buffer (section 12)
    assert airspace.computation_ceiling_ft_agl() == 10000 + 2000
    assert airspace.computation_radius_nm() == 10


def test_ceiling_check_decisions(kbna):
    charted = airspace.charted_ceiling_ft_agl()
    # contained, no spill -> charted used
    chk = airspace.ceiling_check(pd.Series([2000.0, 3000.0, 3500.0]),
                                 pd.Series([1000.0, 2000.0]), 0.0)
    assert not chk["override"]
    assert chk["effective_ceiling_ft_agl"] == charted
    # not contained -> raised to round_up_500(P95), never below charted
    chk = airspace.ceiling_check(pd.Series([5100.0] * 20),
                                 pd.Series([1000.0]), 0.0)
    assert chk["override"] and chk["effective_ceiling_ft_agl"] == 5500.0
    # spill above limit -> raised in 500 ft steps until spill <= 10 %
    heights = pd.Series([1000.0] * 8 + [4300.0, 4800.0])
    chk = airspace.ceiling_check(pd.Series([2000.0]), heights, 0.0)
    assert chk["override"]
    assert chk["effective_ceiling_ft_agl"] == charted + 500.0
    assert chk["spill_share_at_effective"] <= config.CEILING_SPILL_MAX
    # cap
    chk = airspace.ceiling_check(pd.Series([15000.0] * 5), pd.Series([]), 0)
    assert chk["effective_ceiling_ft_agl"] == config.CEILING_CAP_FT_AGL
    # fixed mode: reported but not applied
    saved = config.AIRSPACE_CEILING_MODE
    try:
        config.AIRSPACE_CEILING_MODE = "fixed"
        chk = airspace.ceiling_check(pd.Series([9000.0] * 5),
                                     pd.Series([]), 0)
        assert not chk["override"] and not chk["ga_contained"]
    finally:
        config.AIRSPACE_CEILING_MODE = saved


def _fake_ourairports(cache: Path) -> None:
    """A two-row OurAirports database (KBNA, one runway pair) so that the
    automatic profile needs no network."""
    cache.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "id": 1, "ident": "KBNA", "type": "large_airport",
        "name": "Nashville International Airport", "latitude_deg": 36.1245,
        "longitude_deg": -86.6782, "elevation_ft": 599,
        "scheduled_service": "yes", "gps_code": "KBNA"}]).to_csv(
        cache / "airports.csv", index=False)
    pd.DataFrame([{
        "airport_ident": "KBNA", "closed": 0,
        "le_ident": "02L", "le_latitude_deg": 36.11769867,
        "le_longitude_deg": -86.68650055,
        "he_ident": "20R", "he_latitude_deg": 36.13779831,
        "he_longitude_deg": -86.6785965}, {
        "airport_ident": "KBNA", "closed": 1,
        "le_ident": "09", "le_latitude_deg": 36.12,
        "le_longitude_deg": -86.69,
        "he_ident": "27", "he_latitude_deg": 36.12,
        "he_longitude_deg": -86.67}]).to_csv(cache / "runways.csv",
                                             index=False)


def test_automatic_profile(kbna, synth_dataset, tmp_path, monkeypatch):
    """A missing profile is created with no manual input: geometry from
    the (stubbed) OurAirports database, timezone from the coordinates,
    preset and arrivals assumption from the data, airspace block."""
    from proximity_pipeline import profile_setup as ps
    import yaml
    _fake_ourairports(tmp_path / ".ourairports")
    monkeypatch.setattr(ps, "OA_CACHE", tmp_path / ".ourairports")
    monkeypatch.setattr(ps, "AIRPORTS_DIR", tmp_path)

    path, lines = ps.create_profile("KBNA", synth_dataset)
    prof = yaml.safe_load(path.read_text())
    assert prof["timezone"] == "America/Chicago"
    # synthetic finals are flown at ~140 kt -> air carrier
    assert prof["preset"] == "air_carrier"
    # synthetic landings roll out on the ground -> touchdown evidence
    # decides, no arrivals assumption, no coverage-floor overrides
    assert prof["assume_arrivals_dataset"] is False
    assert prof["overrides"] == {}
    assert set(prof["runways"]) == {"02L", "20R"}      # closed 09/27 skipped
    b02, b20 = prof["runways"]["02L"][2], prof["runways"]["20R"][2]
    assert abs(b02 - 17.7) < 0.5 and abs(b20 - 197.7) < 0.5
    assert prof["airspace"]["ceiling_ft_msl"] == 4600
    assert prof["airspace"]["ceiling_mode"] == "auto"
    assert any("preset: air_carrier" in x for x in lines)
    with pytest.raises(FileExistsError):
        ps.create_profile("KBNA", synth_dataset)       # never overwritten


def test_ensure_profile_adds_missing_airspace_block(tmp_path, monkeypatch):
    from proximity_pipeline import profile_setup as ps
    import yaml
    monkeypatch.setattr(ps, "AIRPORTS_DIR", tmp_path)
    (tmp_path / "KTST.yaml").write_text(
        "icao: KTST\nname: Test\nlatitude: 40.0\nlongitude: -100.0\n"
        "elevation_ft: 1234.0\nruns: {}\n")
    ps.ensure_profile("KTST")
    prof = yaml.safe_load((tmp_path / "KTST.yaml").read_text())
    assert prof["airspace"]["ceiling_ft_msl"] == 5200   # 1234 + 4000


def test_check_data_audit(synth_dataset, kbna):
    table = loading.audit_dataset(synth_dataset)
    assert len(table) == 2 and table["ok"].all()
    assert (table["aircraft"] > 50).all()
    assert (table["geo_alt_share"] > 0.99).all()
    assert (table["interp_share"] < 0.2).all()
