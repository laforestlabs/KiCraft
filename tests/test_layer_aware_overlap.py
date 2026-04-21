from kicraft.autoplacer.brain.types import Point
from kicraft.autoplacer.brain.subcircuit_composer import can_overlap

def test_two_smt_children_disjoint_layers_can_overlap():
    a_front = (Point(0, 0), Point(10, 10))
    a_back = None
    a_tht = None
    
    b_front = None
    b_back = (Point(0, 0), Point(10, 10))
    b_tht = None
    
    assert can_overlap((a_front, a_back, a_tht), (b_front, b_back, b_tht)) is True

def test_two_smt_children_same_layer_cannot_overlap():
    a_front = (Point(0, 0), Point(10, 10))
    a_back = None
    a_tht = None
    
    b_front = (Point(5, 5), Point(15, 15))
    b_back = None
    b_tht = None
    
    assert can_overlap((a_front, a_back, a_tht), (b_front, b_back, b_tht)) is False

def test_tht_keepout_blocks_overlap():
    a_front = (Point(0, 0), Point(10, 10))
    a_back = None
    a_tht = (Point(2, 2), Point(8, 8))
    
    b_front = None
    b_back = (Point(0, 0), Point(10, 10))
    b_tht = None
    
    # Overlaps! b_back overlaps with a_tht
    assert can_overlap((a_front, a_back, a_tht), (b_front, b_back, b_tht)) is False

def test_can_overlap_symmetric():
    a_env = (
        (Point(0, 0), Point(10, 10)),
        None,
        (Point(2, 2), Point(8, 8))
    )
    b_env = (
        None,
        (Point(0, 0), Point(10, 10)),
        None
    )
    assert can_overlap(a_env, b_env) == can_overlap(b_env, a_env)
