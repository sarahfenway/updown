import logging
import re

import arrow
import tweepy
import atproto
from arrow import ParserError
from django.conf import settings
from django.utils import timezone
from django.utils.timezone import make_aware

logger = logging.getLogger(__name__)

A_DAY_IN_SECONDS = 60 * 60 * 24

TFL_NAME_CORRECTIONS = {
    "King's Cross": "King's Cross St. Pancras",
    "Cutty Sark": "Cutty Sark for Maritime Greenwich",
    "London Liverpool Street": "Liverpool Street",
}

# TfL spells this prefix inconsistently across otherwise-identical entries:
# "No Step Free Access - ", "No Step Free Access- ", "No Step Free Access: ",
# "No step free Access - ", and occasionally doubled. The separator is
# required so that real sentences starting "No step free access between..."
# are left alone.
STEP_FREE_PREFIX = re.compile(
    r"^(?:no step free access\s*[-–—:]+\s*)+", re.IGNORECASE
)


def strip_step_free_prefix(text):
    """Remove TfL's "No Step Free Access" prefix, however it is punctuated.

    An exact-string replace let spelling variants through as distinct
    report texts for the same disruption, which is what fed the
    duplicate-incident loop in consolidate_incidents.
    """
    return STEP_FREE_PREFIX.sub("", text)


def remove_tfl_specifics(text):
    # NOTE: re.sub's 4th positional argument is ``count``, not ``flags`` —
    # these all used to pass ``re.IGNORECASE`` (== 2) there, so matching was
    # case-sensitive and silently capped at 2 replacements. They now pass
    # ``flags=re.IGNORECASE`` properly. Patterns are raw strings so the
    # ``\.`` escapes are valid (and don't emit SyntaxWarnings on 3.12+).
    text = re.sub(r"(Please )?[Cc]all.*0[38]43 ?222 ?1234.*journey\.?", "", text)
    # Word-bounded so case-insensitive matching turns TfL's first-person
    # "we"/"We" into "TfL" without mangling words that merely end in "we"
    # (e.g. "Crewe ").
    text = re.sub(r"\bwe ", "TfL ", text, flags=re.IGNORECASE)
    text = re.sub(
        r"member of staff", "member of TfL staff", text, flags=re.IGNORECASE
    )
    text = re.sub(
        r"Call our Travel Information Centre on 0343 222 1234 if you need more help\.?",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"Call our Travel Information Centre on 0343 222 1234 for further help\.?",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"Call our Travel Information Centre on 0343 222 1234 for help planning your journey\.?",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"Call us on 0343 222 1234 if you need more help\.?",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"Call 0343 222 1234 for further help\.?", "", text, flags=re.IGNORECASE
    )
    text = re.sub(
        r"Call 0343 222 1234 if you need further help\.?",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"Call us on 0343 222 1234 for further help\.?",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"Call us on 0343 222 1234 for further info\.?",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return text


def parse_date(text):
    possible_formats = ["D MMMM YYYY", "D MMMM", "MMMM YYYY", "MMMM", "dddd D MMMM"]
    no_month_formats = ["dddd D"]

    try:
        date_ = arrow.get(text, possible_formats)

        if "early" in text:
            date_ = date_.replace(day=10)
        if "mid" in text:
            date_ = date_.replace(day=20)
        if "late" in text:
            date_ = date_.ceil("month")

        if date_.year == 1:
            date_ = date_.replace(year=arrow.utcnow().year)

    except ParserError:
        try:
            date_ = arrow.get(text, no_month_formats)
            date_ = date_.replace(year=2)
        except ParserError:
            date_ = None

    if date_:
        date_ = make_aware(date_.naive)

    return date_


def find_dates(text):
    start_and_end = re.search("From (.*?) until (.*?)(,|there|due)", text, flags=re.I)
    multi_date = re.search(r"(\w+ \d+), .*? and (\w+ \d+ \w+)", text, flags=re.I)

    start_date = None
    end_date = None

    if start_and_end:
        start_date = parse_date(start_and_end.groups()[0])
        end_date = parse_date(start_and_end.groups()[1])

    elif multi_date:
        start_date = parse_date(multi_date.groups()[0])
        end_date = parse_date(multi_date.groups()[1])

    while start_date and end_date and end_date < start_date:
        start_date = start_date.replace(year=start_date.year - 1)

    if start_date and end_date and start_date.year == 2:
        start_date = start_date.replace(year=end_date.year, month=end_date.month)

    return start_date, end_date


def fix_additional_info_grammar(text):
    # It appears that full stops and spaces aren't always used...
    text = re.sub(r"\.([A-Z<])", r". \1", text)
    text = re.sub(r"[^ \.](Please allow extra time for your journey.)", r". \1", text)

    return text


def send_tweet(tweet_text):
    """Send a tweet"""
    if settings.DEBUG is False:
        try:
            twitter = tweepy.Client(
                consumer_key=settings.TWITTER_API_KEY,
                consumer_secret=settings.TWITTER_API_SECRET,
                access_token=settings.TWITTER_ACCESS_TOKEN,
                access_token_secret=settings.TWITTER_ACCESS_TOKEN_SECRET,
            )
            twitter.create_tweet(text=tweet_text)
        # Twitter has many transient foibles; never let a post failure break
        # the update run — but log it so a broken integration is visible
        # instead of silently posting nothing for weeks.
        except Exception:
            logger.exception("Failed to send tweet")
    else:
        print("Should have tweeted: " + tweet_text)


def send_bluesky(tweet_text):
    """Send a blue sky post"""
    if settings.DEBUG is False:
        try:
            client = atproto.Client()
            profile = client.login(
                settings.BLUESKY_USER_NAME, settings.BLUESKY_PASSWORD
            )
            print("Welcome,", profile.display_name)

            text = atproto.client_utils.TextBuilder().text(tweet_text)
            client.send_post(text)
        except Exception:
            logger.exception("Failed to send Bluesky post")
    else:
        print("Should have posted on blue sky: " + tweet_text)


def send_bluesky_messages(messages):
    """Send multiple Bluesky posts using one login session."""
    messages = list(messages)
    if not messages:
        return

    if settings.DEBUG is False:
        try:
            client = atproto.Client()
            profile = client.login(
                settings.BLUESKY_USER_NAME, settings.BLUESKY_PASSWORD
            )
            print("Welcome,", profile.display_name)

            for message in messages:
                text = atproto.client_utils.TextBuilder().text(message)
                client.send_post(text)
        except Exception:
            logger.exception("Failed to send Bluesky posts")
    else:
        for message in messages:
            print("Should have posted on blue sky: " + message)


def update_last_updated():
    with open("last_updated", "w") as f:
        f.write(timezone.now().astimezone().strftime("%H:%M %d %b"))


def get_last_updated():
    try:
        with open("last_updated", "r") as f:
            return f.read()
    except FileNotFoundError:
        return None
