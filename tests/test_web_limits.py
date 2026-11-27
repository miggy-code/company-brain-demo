"""The persona demo's visitor limits: sliding windows per client, and which address is the
client's when the backend sits behind proxies."""

from brain.interfaces.web.limits import RateLimiter, Window, client_ip


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_a_window_admits_up_to_its_limit_then_says_when_to_retry() -> None:
    clock = Clock()
    limiter = RateLimiter({"answers": (Window(limit=4, seconds=600),)}, clock=clock)

    assert limiter.take("answers", "1.2.3.4", cost=3) is None
    assert limiter.take("answers", "1.2.3.4", cost=1) is None
    clock.now += 100
    retry = limiter.take("answers", "1.2.3.4", cost=1)

    assert retry is not None and 499 <= retry <= 500


def test_a_refused_request_costs_nothing() -> None:
    clock = Clock()
    limiter = RateLimiter({"answers": (Window(limit=4, seconds=600),)}, clock=clock)

    assert limiter.take("answers", "a", cost=3) is None
    assert limiter.take("answers", "a", cost=2) is not None
    assert limiter.take("answers", "a", cost=1) is None


def test_the_window_slides() -> None:
    clock = Clock()
    limiter = RateLimiter({"answers": (Window(limit=2, seconds=60),)}, clock=clock)

    assert limiter.take("answers", "a", cost=2) is None
    assert limiter.take("answers", "a") is not None
    clock.now += 61
    assert limiter.take("answers", "a") is None


def test_every_window_must_admit_and_clients_are_counted_apart() -> None:
    clock = Clock()
    windows = (Window(limit=10, seconds=600), Window(limit=3, seconds=86_400))
    limiter = RateLimiter({"answers": windows}, clock=clock)

    assert limiter.take("answers", "a", cost=3) is None
    clock.now += 700  # past the short window, inside the day
    assert limiter.take("answers", "a") is not None
    assert limiter.take("answers", "b", cost=3) is None


def test_a_request_bigger_than_a_window_is_refused_outright() -> None:
    limiter = RateLimiter({"answers": (Window(limit=2, seconds=60),)}, clock=Clock())

    assert limiter.take("answers", "a", cost=3) is not None


def test_without_trusted_proxies_the_forwarded_header_is_ignored() -> None:
    assert client_ip("10.0.0.9", "6.6.6.6", trusted_proxies=0) == "10.0.0.9"


def test_with_trusted_proxies_the_client_is_the_hop_they_appended() -> None:
    # The client spoofs 6.6.6.6; the platform's one proxy appends the address it saw.
    assert client_ip("10.0.0.9", "6.6.6.6, 203.0.113.7", trusted_proxies=1) == "203.0.113.7"
    assert client_ip("10.0.0.9", "6.6.6.6, 203.0.113.7, 10.0.0.2", trusted_proxies=2) == (
        "203.0.113.7"
    )


def test_a_short_forwarded_header_falls_back_to_the_peer() -> None:
    assert client_ip("10.0.0.9", None, trusted_proxies=1) == "10.0.0.9"
    assert client_ip("10.0.0.9", "203.0.113.7", trusted_proxies=2) == "10.0.0.9"


def test_a_request_charged_to_several_buckets_is_admitted_by_all_or_costs_nothing() -> None:
    limiter = RateLimiter(
        {"unshares": (Window(limit=2, seconds=600),), "answers": (Window(limit=3, seconds=600),)},
        clock=Clock(),
    )

    assert limiter.take_all({"answers": 3}, "a") is None
    assert limiter.take_all({"unshares": 1, "answers": 1}, "a") is not None
    # The refused unshare took no unshare.
    assert limiter.take_all({"unshares": 2}, "a") is None
