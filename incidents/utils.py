import re

import arrow
import tweepy
from arrow import ParserError
from django.conf import settings
from django.utils import timezone
from django.utils.timezone import make_aware

A_DAY_IN_SECONDS = 60 * 60 * 24

TFL_NAME_CORRECTIONS = {
    "King's Cross": "King's Cross St. Pancras",
    "Cutty Sark": "Cutty Sark for Maritime Greenwich",
    "London Liverpool Street": "Liverpool Street",
}


def remove_tfl_specifics(text):
    # We always reset this just in case there is an update
    text = re.sub("(Please )?[Cc]all.*0[38]43 ?222 ?1234.*journey\.?", "", text)
    text = re.sub("we ", "TfL ", text, re.IGNORECASE)
    text = re.sub("member of staff", "member of TfL staff", text, re.IGNORECASE)
    text = re.sub(
        "Call our Travel Information Centre on 0343 222 1234 if you need more help\.?",
        "",
        text,
        re.IGNORECASE,
    )
    text = re.sub(
        "Call our Travel Information Centre on 0343 222 1234 for further help\.?",
        "",
        text,
        re.IGNORECASE,
    )
    text = re.sub(
        "Call our Travel Information Centre on 0343 222 1234 for help planning your journey\.?",
        "",
        text,
        re.IGNORECASE,
    )
    text = re.sub(
        "Call us on 0343 222 1234 if you need more help\.?", "", text, re.IGNORECASE
    )
    text = re.sub("Call 0343 222 1234 for further help\.?", "", text, re.IGNORECASE)
    text = re.sub(
        "Call 0343 222 1234 if you need further help\.?", "", text, re.IGNORECASE
    )
    text = re.sub(
        "Call us on 0343 222 1234 for further help\.?", "", text, re.IGNORECASE
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

    start_date = None
    end_date = None

    if start_and_end:
        start_date = parse_date(start_and_end.groups()[0])
        end_date = parse_date(start_and_end.groups()[1])

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
        # Except everything. TODO: Look into some of twitters annoying foibles
        except:
            pass
    else:
        print("Should have tweeted: " + tweet_text)


def update_last_updated():
    with open("last_updated", "w") as f:
        f.write(timezone.now().astimezone().strftime("%H:%M %d %b"))


def get_last_updated():
    try:
        with open("last_updated", "r") as f:
            return f.read()
    except FileNotFoundError:
        return None
