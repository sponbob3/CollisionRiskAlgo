"""Scaffolding tests: profile + airspace block, ceiling sanity check,
new-airport generation (stubbed airport database), check-data audit."""

from __future__ import annotations

import types

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


def test_new_airport_appends_airspace_block(kbna, tmp_path, monkeypatch):
    """new-airport = the vendored generator + the airspace block with the
    ceiling pre-filled as field elevation + 4,000 ft. The OurAirports
    lookup is stubbed so the test needs no network."""
    from goaround_pipeline import profiles

    rwy = pd.DataFrame({"name": ["09", "27"],
                        "latitude": [40.0, 40.0],
                        "longitude": [-100.02, -99.98],
                        "bearing": [90.0, 270.0]})
    apt = types.SimpleNamespace(name="Test Field", latlon=(40.0, -100.0),
                                altitude=1234.0,
                                runways=types.SimpleNamespace(data=rwy))
    fake_traffic_data = types.SimpleNamespace(airports={"KTST": apt})
    monkeypatch.setitem(__import__("sys").modules, "traffic.data",
                        fake_traffic_data)
    monkeypatch.setattr(profiles, "AIRPORTS_DIR", tmp_path)
    monkeypatch.setattr(airspace, "AIRPORTS_DIR", tmp_path)

    path = airspace.create_profile("KTST")
    text = path.read_text()
    assert "airspace:" in text and "ceiling_mode: auto" in text
    assert "ceiling_ft_msl: 5200" in text        # 1234 + 4000 -> 5200
    assert "VERIFY AGAINST THE SECTIONAL" in text
    import yaml
    prof = yaml.safe_load(text)
    assert prof["airspace"]["radius_nm"] == 10
    assert prof["runways"]["09"][2] == 90.0


def test_check_data_audit(synth_dataset, kbna):
    table = loading.audit_dataset(synth_dataset)
    assert len(table) == 2 and table["ok"].all()
    assert (table["aircraft"] > 50).all()
    assert (table["geo_alt_share"] > 0.99).all()
    assert (table["interp_share"] < 0.2).all()
