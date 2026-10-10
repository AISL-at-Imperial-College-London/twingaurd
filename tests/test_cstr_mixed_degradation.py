from twinguard.cstr import mixed_degradation as cstr_mixed


def test_population_assignment_is_balanced_and_deterministic():
    assignments=[cstr_mixed._fault_assignment(i) for i in range(600)]
    assert sum(fault=="ua_fouling" for fault,_ in assignments)==300
    assert sum(fault=="feed_temperature_step" for fault,_ in assignments)==300
    for fault in cstr_mixed.FAULTS:
        for severity in ("low","medium","high"):
            assert assignments.count((fault,severity))==100


def test_seed_blocks_and_operational_features_are_fault_blind():
    assert len(set(cstr_mixed.ROLE_SEEDS.values()))==4
    forbidden=("fault","severity","magnitude","ua")
    assert not any(token in feature.lower() for feature in cstr_mixed.FEATURE_NAMES for token in forbidden)


def test_full_configuration_freezes_transferred_method():
    config=cstr_mixed.Config(); config.validate()
    assert (config.training_count,config.final_count,config.trajectories)==(600,600,2000)
    assert (config.bootstrap_target,config.ensemble_size)==(50,10)
