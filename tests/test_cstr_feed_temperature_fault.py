import copy

import numpy as np

from twinguard.cstr import episode_design as episode_design
from twinguard.cstr import feed_temperature_fault as cstr_fault_check


def test_feed_temperature_fault_is_plant_only_and_activates_at_onset():
    template = episode_design.build_template(episode_design.EpisodeDesignConfig())
    onset = template.t + 2
    nominal, plant = cstr_fault_check.configure_pair(template, "feed_temperature_step", 20.0, onset)
    assert nominal.faults.feed_temperature(onset, nominal.Tin_base) == 350.0
    assert plant.faults.feed_temperature(onset - 1, plant.Tin_base) == 350.0
    assert plant.faults.feed_temperature(onset, plant.Tin_base) == 370.0
    assert not nominal.fault_cfg.enable
    assert plant.fault_cfg.feed_temp_step_on


def test_default_fault_configuration_preserves_feed_temperature():
    template = episode_design.build_template(episode_design.EpisodeDesignConfig())
    assert template.faults.feed_temperature(template.t, template.Tin_base) == template.Tin_base


def test_action_and_severity_design_is_predeclared():
    assert len(cstr_fault_check.ACTIONS_LPS) == 9
    assert np.all(np.diff(cstr_fault_check.ACTIONS_LPS) > 0)
    assert len(cstr_fault_check.SEVERITIES["feed_temperature_step"]) == 3
