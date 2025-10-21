from django.test import TestCase
from django.utils import timezone
from .utils import find_dates
from .sources.tflapiv1 import check
from .models import Report
from unittest.mock import patch

class FindDatesTest(TestCase):
    def test_parse_multi_date(self):
        text = "Tuesday 21, Wednesday 22 and Thursday 23 October"
        start_date, end_date = find_dates(text)
        self.assertEqual(start_date.day, 21)
        self.assertEqual(start_date.month, 10)
        self.assertEqual(end_date.day, 23)
        self.assertEqual(end_date.month, 10)

    def test_parse_fail(self):
        text = "Some random text"
        start_date, end_date = find_dates(text)
        self.assertIsNone(start_date)
        self.assertIsNone(end_date)

class TflApiV1Test(TestCase):
    @patch('incidents.sources.tflapiv1.requests.get')
    def test_fallback_to_now(self, mock_get):
        mock_get.return_value.status_code = 200
        mock_get.return_value.text = '''
            [{
                "description": "Some disruption with no step free access",
                "atcoCode": "123",
                "appearance": "PlannedWork",
                "additionalInformation": ""
            }]
        '''
        with patch('incidents.sources.tflapiv1.find_station_from_naptan') as mock_find_station:
            mock_find_station.return_value = True
            check()
            report = Report.objects.first()
            self.assertIsNotNone(report.start_time)
            self.assertTrue((timezone.now() - report.start_time).total_seconds() < 1)
