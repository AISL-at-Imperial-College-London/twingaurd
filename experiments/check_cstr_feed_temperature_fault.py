"""Define and smoke-test the CSTR feed-temperature-step degradation used in the mixed-degradation
evaluation (recorded output: artifacts/cstr/feed_temperature_fault).
"""

from _runner import cstr_stage

if __name__ == "__main__":
    cstr_stage("twinguard.cstr.feed_temperature_fault", __doc__, "cstr_feed_temperature_fault")
