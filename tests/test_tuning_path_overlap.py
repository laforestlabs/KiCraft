import pytest

from kicraft.tuning.corpus import freeze_workspace
from kicraft.tuning.workspace import prepare_scratch


@pytest.mark.parametrize("operation", ["scratch", "freeze"])
@pytest.mark.parametrize("relationship", ["same", "destination_parent", "destination_child"])
def test_overlapping_workspace_paths_never_delete_source(tmp_path, operation, relationship):
    source = tmp_path / "source"
    source.mkdir()
    marker = source / "valuable"
    marker.write_text("original")
    destination = {"same": source, "destination_parent": tmp_path,
                   "destination_child": source / "nested"}[relationship]

    with pytest.raises(ValueError, match="overlap"):
        if operation == "scratch":
            prepare_scratch(source, destination, fresh_leaves=True)
        else:
            freeze_workspace(source, destination)
    assert marker.read_text() == "original"
