from kloppy.domain.models.event import Event


def split_by_time(events: list[Event], test_frac: float = 0.2):
    """Split events by timestamps, so that the "train" data
    comes before all "test" data. This is useful when you want to have a certain
    number of events that doesn't cleanly map to a number of matches/seasons.

    Note that the resulting data sets are sorted, so be careful when making any
    further splits.
    """
    sorted_events = sorted(events, key=lambda event: event.absolute_timestamp)
    cutpoint = int((1 - test_frac) * len(sorted_events))
    train_events = sorted_events[:cutpoint]
    test_events = sorted_events[cutpoint:]
    return train_events, test_events