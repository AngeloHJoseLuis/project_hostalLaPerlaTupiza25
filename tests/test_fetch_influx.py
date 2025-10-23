import os
import pandas as pd
import numpy as np
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from iot_pipeline import fetch_influx_data, query_api


class TestFetchInflux(unittest.TestCase):
    def test_fetch_influx_data_monkeypatch(self):
        # Create a fake DataFrame similar to what query_data_frame would return
        times = pd.date_range('2025-01-01', periods=10, freq='1min')
        df = pd.DataFrame({
            '_time': times,
            '_value': np.linspace(10,20,10),
            'device': ['plug01']*10
        })

        # Monkeypatch the query_api.query_data_frame function
        original = getattr(query_api, 'query_data_frame', None)
        try:
            setattr(query_api, 'query_data_frame', lambda q: df)
            res = fetch_influx_data(hours=1, device='plug01')
            self.assertFalse(res.empty)
            self.assertTrue('time' in res.columns or '_time' in res.columns)
        finally:
            if original is not None:
                setattr(query_api, 'query_data_frame', original)

if __name__ == '__main__':
    unittest.main()
