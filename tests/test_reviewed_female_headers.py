import pytest

from kicraft.design.lowering import lower_requirement
from kicraft.design.models import CircuitRequirement
from kicraft.design.part_identity import reviewed_part


def requirement(count, *, gender="female", rows=1, exact=None):
    return CircuitRequirement(
        id="header", sheet="MAIN", role="connector", family="pin-header",
        parameters={"gender": gender, "rows": rows}, exact_part=exact,
        ports={f"pin{i}": f"NET{i}" for i in range(1, count + 1)},
    )


@pytest.mark.parametrize("count", [2, 4, 8])
@pytest.mark.parametrize("explicit", [False, True])
def test_female_header_realizes_requested_physical_interface(count, explicit):
    exact = f"PM254V-11-{count:02d}-H85" if explicit else None
    artifact = lower_requirement(requirement(count, exact=exact))
    group = artifact.groups[0]
    record = reviewed_part(group.mpn)
    assert record is not None
    assert {"pin-header", "female", "single-row"} <= record.physical_features
    assert set(record.contacts) == {str(i) for i in range(1, count + 1)}
    assert {pin.pin: pin.net for pin in artifact.pins} == {
        str(i): f"NET{i}" for i in range(1, count + 1)
    }
    assert not artifact.no_connects
    if exact:
        assert group.mpn.casefold() == exact.casefold()


@pytest.mark.parametrize("count,rows", [(1, 1), (6, 1), (4, 2), (8, 2)])
def test_unsupported_female_shapes_remain_generic(count, rows):
    group = lower_requirement(requirement(count, rows=rows)).groups[0]
    assert group.mpn is None
    assert group.footprint.startswith("Connector_PinSocket_2.54mm:")


@pytest.mark.parametrize("count,gender,rows,exact", [
    (2, "female", 1, "TSW-102-07-G-S"),
    (2, "male", 2, "TSW-102-07-G-S"),
    (4, "male", 1, "TSW-102-07-G-S"),
    (2, "male", 1, "PM254V-11-02-H85"),
    (4, "female", 1, "PM254V-11-02-H85"),
    (4, "female", 2, "PM254V-11-04-H85"),
    (6, "female", 1, "PM254V-11-06-H85"),
])
def test_explicit_identity_must_match_gender_count_and_rows(count, gender, rows, exact):
    with pytest.raises(ValueError):
        lower_requirement(requirement(count, gender=gender, rows=rows, exact=exact))


def test_explicit_male_samtec_remains_available():
    group = lower_requirement(requirement(2, gender="male", exact="TSW-102-07-G-S")).groups[0]
    assert group.mpn.casefold() == "tsw-102-07-g-s"
    assert group.symbol == "Connector_Generic:Conn_01x02"
    assert group.footprint == "Connector_PinHeader_2.54mm:PinHeader_1x02_P2.54mm_Vertical"
