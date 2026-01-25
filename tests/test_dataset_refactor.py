import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
# Import from src.data to verify package stability before/after refactor
from src.data import prepare_training_data

class TestDatasetRefactor(unittest.TestCase):
    def setUp(self):
        # Create dummy dataframe mimicking session data
        self.n_samples = 100
        dates = [datetime(2022, 1, 1) + timedelta(hours=i) for i in range(self.n_samples)]
        
        # Base features
        self.df = pd.DataFrame({
            'session_id': [f'sess_{i}' for i in range(self.n_samples)],
            'user_id': [f'user_{i%5}' for i in range(self.n_samples)], # 5 users
            'start_time': dates,
            'duration_minutes': np.random.rand(self.n_samples) * 60,
            'is_weekend': np.random.randint(0, 2, self.n_samples),
            'is_after_hours': np.random.randint(0, 2, self.n_samples),
            # Labels
            'label_session': np.zeros(self.n_samples, dtype=int),
            'label_user': np.zeros(self.n_samples, dtype=int),
        })
        
        # Add some numeric features expected by FeatureExtractor
        for col in ['file_copy_count', 'exe_copy_count', 'doc_copy_count', 'pdf_copy_count', 
                    'archive_copy_count', 'total_actions', 'action_density']:
            self.df[col] = np.random.rand(self.n_samples)
            
        # Optional/Categorical
        self.df['role'] = 'Employee'
        
        # Mark one user as insider
        self.insider_user = 'user_4'
        self.df.loc[self.df['user_id'] == self.insider_user, 'label_user'] = 1
        self.df.loc[self.df['user_id'] == self.insider_user, 'label_session'] = 1 # Simple case
        
        self.insider_users = [self.insider_user]
        self.insider_incidents = [{'user': self.insider_user, 'start': dates[0], 'end': dates[-1]}]

    def test_prepare_training_data_structure(self):
        """Test that prepare_training_data returns correct structure and keys."""
        data = prepare_training_data(
            self.df,
            sequence_length=10,
            train_ratio=0.5,
            val_ratio=0.2,
            test_ratio=0.3,
            insider_users=self.insider_users,
            insider_incidents=self.insider_incidents
        )
        
        # Check Critical Keys
        required_keys = [
            'train_sequences', 'val_sequences', 'test_sequences',
            'test_user_ids', 'feature_extractor'
        ]
        for key in required_keys:
            self.assertIn(key, data, f"Missing key: {key}")
            
        # Check shapes
        # Train should only have normal users (user_0 to user_3)
        # Test should have all insider sessions + some normal keys
        
        self.assertTrue(len(data['train_sequences']) > 0)
        self.assertTrue(len(data['test_sequences']) > 0)
        
        # Check test_user_ids length matches test_sequences
        self.assertEqual(len(data['test_user_ids']), len(data['test_sequences']))
        
        # Verify persistence capability of extractor (since we updated it)
        extractor = data['feature_extractor']
        self.assertTrue(hasattr(extractor, 'save'), "FeatureExtractor missing save method")
        self.assertTrue(hasattr(extractor, 'load'), "FeatureExtractor missing load method")
        
    def test_import_location(self):
        """Verify we can verify the import location."""
        # This test passes if the import at top level worked
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
