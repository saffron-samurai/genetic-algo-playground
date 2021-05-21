from ..data import get_baggage


def test_get_baggage_data():
    baggage_df = get_baggage()
    assert not baggage_df.empty
