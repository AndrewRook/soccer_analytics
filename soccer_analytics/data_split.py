from kloppy.domain.models.event import EventDataset


def split_by_time(
        events: list[EventDataset], test_frac: float = 0.2,
        timestamp_attribute: str = "kickoff_timestamp"
) -> tuple[list[EventDataset], list[EventDataset]]:
    """Split matches (EventDataset instances, actually) by timestamps, so that the "train" data
    comes before all "test" data. This is useful when you want to have a certain
    number of events that doesn't cleanly map to a number of seasons.

    Note that the resulting data sets are sorted, so be careful when making any
    further splits.
    """
    sorted_events = sorted(events, key=lambda event: getattr(event, timestamp_attribute))
    # Flooring to int rather than rounding should guarantee that the test dataset has records
    # >= test_frac, never less than.
    cutpoint = int((1 - test_frac) * len(sorted_events))
    train_events = sorted_events[:cutpoint]
    test_events = sorted_events[cutpoint:]
    return train_events, test_events