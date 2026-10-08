import copy

import numpy as np

from twinguard.cstr import episode_design as b5
from twinguard.cstr import feed_temperature_fault as c16a


def test_feed_temperature_fault_is_plant_only_and_activates_at_onset():
    template = b5.build_template(b5.MilestoneB5Config())
    onset = template.t + 2
    nominal, plant = c16a.configure_pair(template, "feed_temperature_step", 20.0, onset)
    assert nominal.faults.feed_temperature(onset, nominal.Tin_base) == 350.0
    assert plant.faults.feed_temperature(onset - 1, plant.Tin_base) == 350.0
    assert plant.faults.feed_temperature(onset, plant.Tin_base) == 370.0
    assert not nominal.fault_cfg.enable
    assert plant.fault_cfg.feed_temp_step_on


def test_default_fault_configuration_preserves_feed_temperature():
    template = b5.build_template(b5.MilestoneB5Config())
    assert template.faults.feed_temperature(template.t, template.Tin_base) == template.Tin_base


def test_action_and_severity_design_is_predeclared():
    assert len(c16a.ACTIONS_LPS) == 9
    assert np.all(np.diff(c16a.ACTIONS_LPS) > 0)
    assert len(c16a.SEVERITIES["feed_temperature_step"]) == 3
