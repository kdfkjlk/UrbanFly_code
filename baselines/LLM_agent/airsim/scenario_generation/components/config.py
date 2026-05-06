ACTOR_TYPE = {
    'person': 0,
    'bird': 1,
    'marker': 2
}

MODE_TYPE = {
    'random': 0,
    'straight': 1,
    'loop': 2
}

BP_TYPE = {
    ACTOR_TYPE['person']: 'manBP',
    ACTOR_TYPE['marker']: 'bp_marker',
    ACTOR_TYPE['bird']: 'bp_bird'
}