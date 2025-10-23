import os
import tempfile
import numpy as np
import pandas as pd
import unittest

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from iot_pipeline import build_features, train_autoencoder


def make_synthetic_df(n=200, freq='5s'):
    rng = pd.date_range('2025-01-01', periods=n, freq=freq)
    power = 50 + 10 * np.sin(np.linspace(0, 6.28, n)) + np.random.randn(n) * 0.5
    df = pd.DataFrame({'time': rng, 'value': power, 'device': ['plug01']*n})
    return df


class TestIoTPipeline(unittest.TestCase):
    def test_build_features_basic(self):
        df = make_synthetic_df(n=100)
        # build_features expects columns ['time','value',...'device'] or similar
        feats = build_features(df[['time','value']].rename(columns={'time':'time','value':'value'}).reset_index(drop=True).assign(time=df['time']))
        self.assertIn('power_w', feats.columns)
        self.assertIn('hour', feats.columns)
        self.assertGreater(len(feats), 0)

    def test_train_autoencoder_saves(self):
        df = make_synthetic_df(n=300)
        feats = build_features(df[['time','value']].rename(columns={'time':'time','value':'value'}).reset_index(drop=True).assign(time=df['time']))
        feats = feats.dropna()
        X = feats[['power_w']].values

        # split X into train/test
        split = int(len(X) * 0.8)
        X_train = X[:split]
        X_test = X[split:]

        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'ae_test.keras')
            ae, history = train_autoencoder(X_train, X_test, epochs=1, batch_size=16, model_path=model_path)
            self.assertTrue(os.path.exists(model_path))
            self.assertIn('loss', history.history)

if __name__ == '__main__':
    unittest.main()
