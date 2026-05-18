"""Tests for kicraft.leaf_library.sexpr_edit.

Golden-input pairs derived from the actual KiCad 9 sexpr shapes we
need to rewrite.
"""

from __future__ import annotations

from kicraft.leaf_library.sexpr_edit import (
    renumber_pcb_text,
    renumber_schematic_text,
)


SCH_LIB_SYMBOL = """
(kicad_sch
  (version 20250114)
  (lib_symbols
    (symbol "Device:R"
      (pin_numbers (hide yes))
      (pin_names (offset 0.254) (hide yes))
      (exclude_from_sim no)
      (in_bom yes)
      (on_board yes)
      (property "Reference" "R"
        (at 2.032 0 90)
        (effects (font (size 1.27 1.27))))
      (property "Value" "R" (at 0 0 90)
        (effects (font (size 1.27 1.27)))))
  )
  (symbol
    (lib_id "Device:R")
    (at 100 100 0)
    (unit 1)
    (in_bom yes)
    (on_board yes)
    (uuid "aaaa-bbbb")
    (property "Reference" "R1" (at 105 100 0))
    (property "Value" "10k" (at 100 105 0))
    (instances
      (project "TEST"
        (path "/aaaa-bbbb"
          (reference "R1")
          (unit 1)
        )
      )
    )
  )
)
""".strip()


def test_schematic_rewrites_instance_and_skip_lib_symbol():
    new_text, counts = renumber_schematic_text(SCH_LIB_SYMBOL, {"R1": "R7"})
    # lib_symbols definition (Reference "R" — letter only) is untouched.
    assert '(property "Reference" "R"\n' in new_text
    # Instance property is rewritten.
    assert '(property "Reference" "R7"' in new_text
    # Instance path reference is rewritten.
    assert '(reference "R7")' in new_text
    # Counts match (1 property rewrite + 1 reference instance rewrite).
    assert counts == {"property_reference": 1, "reference_instance": 1}


def test_schematic_unmapped_refs_pass_through():
    text = '(property "Reference" "U99" (at 0 0 0))'
    new_text, counts = renumber_schematic_text(text, {"U1": "U2"})
    assert new_text == text
    assert counts == {"property_reference": 0, "reference_instance": 0}


PCB_FRAGMENT = """
(footprint "Resistor_SMD:R_0402_1005Metric"
  (layer "F.Cu")
  (uuid "abc")
  (at 10 20 0)
  (property "Reference" "R5"
    (at 0 -1.17 180)
    (layer "F.SilkS"))
  (property "Value" "10k"
    (at 0 1.17 180))
  (fp_text reference "R5" (at 0 0 0) (layer "F.SilkS"))
  (fp_text user "R5_DEBUG" (at 0 0 0) (layer "F.SilkS"))
)
""".strip()


def test_pcb_rewrites_property_and_fp_text_reference():
    new_text, counts = renumber_pcb_text(PCB_FRAGMENT, {"R5": "R12"})
    assert '(property "Reference" "R12"' in new_text
    assert '(fp_text reference "R12"' in new_text
    # The fp_text user "R5_DEBUG" doesn't match ^[A-Z]+[0-9]+$ exactly
    # (it has the _DEBUG suffix) so it's not rewritten.
    assert '(fp_text user "R5_DEBUG"' in new_text
    assert counts == {
        "property_reference": 1,
        "fp_text_reference": 1,
        "fp_text_user": 0,
    }


def test_pcb_fp_text_user_defensive_scan_rewrites_pure_ref_match():
    """Hand-placed silk like (fp_text user "U1") IS rewritten because
    it exactly matches the ref pattern AND is in the map."""
    text = '(fp_text user "U1" (at 0 0 0) (layer "F.SilkS"))'
    new_text, counts = renumber_pcb_text(text, {"U1": "U7"})
    assert '(fp_text user "U7"' in new_text
    assert counts["fp_text_user"] == 1


def test_multi_ref_rewrite_in_single_pass():
    text = """
(property "Reference" "U1" (at 0 0 0))
(property "Reference" "U2" (at 5 5 0))
(property "Reference" "C1" (at 10 10 0))
""".strip()
    new_text, _ = renumber_schematic_text(text, {"U1": "U5", "U2": "U6", "C1": "C9"})
    assert '"Reference" "U5"' in new_text
    assert '"Reference" "U6"' in new_text
    assert '"Reference" "C9"' in new_text
    assert "U1" not in new_text
    assert "U2" not in new_text
