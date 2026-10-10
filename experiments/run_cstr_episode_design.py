"""Generate the CSTR episode population (hidden heat-transfer fouling, observation histories and
evaluator-only truth) used to fit the CSTR discrepancy model.
"""

from _runner import cstr_stage

if __name__ == "__main__":
    cstr_stage("twinguard.cstr.episode_design", __doc__, "cstr_episode_design")
