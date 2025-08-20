"""
Test Suite for Customer Segmentation Model
==========================================

Unit tests for the customer segmentation functionality.
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from customer_segmentation import CustomerSegmentationModel, find_optimal_clusters
from data_generator import generate_ecommerce_data, get_data_summary


class TestCustomerSegmentationModel(unittest.TestCase):
    """Test cases for CustomerSegmentationModel class."""
    
    def setUp(self):
        """Set up test data and model instance."""
        self.model = CustomerSegmentationModel(n_clusters=3, random_state=42)
        self.test_data = generate_ecommerce_data(
            n_customers=100,
            n_products=20,
            date_range_days=180,
            random_state=42
        )
    
    def test_model_initialization(self):
        """Test model initialization."""
        self.assertEqual(self.model.n_clusters, 3)
        self.assertEqual(self.model.random_state, 42)
        self.assertFalse(self.model.is_fitted)
    
    def test_prepare_features(self):
        """Test feature preparation."""
        features = self.model.prepare_features(self.test_data)
        
        # Check if all required columns are present
        expected_columns = ['customer_id', 'recency', 'frequency', 'monetary', 
                          'total_quantity', 'category_diversity', 'avg_order_value']
        self.assertTrue(all(col in features.columns for col in expected_columns))
        
        # Check if number of customers matches unique customers in data
        self.assertEqual(len(features), self.test_data['customer_id'].nunique())
        
        # Check if all values are non-negative
        self.assertTrue((features['frequency'] >= 0).all())
        self.assertTrue((features['monetary'] >= 0).all())
        self.assertTrue((features['total_quantity'] >= 0).all())
    
    def test_model_fitting(self):
        """Test model fitting process."""
        # Fit the model
        fitted_model = self.model.fit(self.test_data)
        
        # Check if model is fitted
        self.assertTrue(self.model.is_fitted)
        self.assertIsNotNone(self.model.labels_)
        self.assertIsNotNone(self.model.cluster_centers_)
        
        # Check if labels are in correct range
        self.assertTrue(all(0 <= label < self.model.n_clusters for label in self.model.labels_))
        
        # Check return type
        self.assertIsInstance(fitted_model, CustomerSegmentationModel)
    
    def test_prediction(self):
        """Test prediction functionality."""
        # First fit the model
        self.model.fit(self.test_data)
        
        # Test prediction on same data
        predictions = self.model.predict(self.test_data)
        self.assertEqual(len(predictions), self.test_data['customer_id'].nunique())
        self.assertTrue(all(0 <= pred < self.model.n_clusters for pred in predictions))
    
    def test_prediction_without_fitting(self):
        """Test that prediction raises error when model is not fitted."""
        with self.assertRaises(ValueError):
            self.model.predict(self.test_data)
    
    def test_fit_predict(self):
        """Test fit_predict method."""
        predictions = self.model.fit_predict(self.test_data)
        
        # Check if model is fitted
        self.assertTrue(self.model.is_fitted)
        
        # Check predictions
        self.assertEqual(len(predictions), self.test_data['customer_id'].nunique())
        self.assertTrue(all(0 <= pred < self.model.n_clusters for pred in predictions))
    
    def test_cluster_summary(self):
        """Test cluster summary generation."""
        # Fit model first
        self.model.fit(self.test_data)
        
        summary = self.model.get_cluster_summary(self.test_data)
        
        # Check if summary has correct structure
        self.assertEqual(len(summary), self.model.n_clusters)
        self.assertTrue('count' in summary.columns.get_level_values(1))
        self.assertTrue('mean' in summary.columns.get_level_values(1))
        self.assertTrue('median' in summary.columns.get_level_values(1))
    
    def test_model_evaluation(self):
        """Test model evaluation metrics."""
        # Fit model first
        self.model.fit(self.test_data)
        
        evaluation = self.model.evaluate_model(self.test_data)
        
        # Check if evaluation contains expected metrics
        self.assertIn('silhouette_score', evaluation)
        self.assertIn('n_clusters', evaluation)
        self.assertIn('n_customers', evaluation)
        
        # Check silhouette score is between -1 and 1
        self.assertTrue(-1 <= evaluation['silhouette_score'] <= 1)
        self.assertEqual(evaluation['n_clusters'], self.model.n_clusters)
    
    def test_segment_insights(self):
        """Test segment insights generation."""
        # Fit model first
        self.model.fit(self.test_data)
        
        insights = self.model.get_segment_insights(self.test_data)
        
        # Check if insights are generated for all clusters
        self.assertEqual(len(insights), self.model.n_clusters)
        
        # Check if all insights are strings
        self.assertTrue(all(isinstance(insight, str) for insight in insights.values()))


class TestDataGenerator(unittest.TestCase):
    """Test cases for data generation functionality."""
    
    def test_generate_ecommerce_data(self):
        """Test e-commerce data generation."""
        data = generate_ecommerce_data(
            n_customers=50,
            n_products=10,
            date_range_days=90,
            random_state=42
        )
        
        # Check if data has correct structure
        expected_columns = ['customer_id', 'order_id', 'order_date', 'product_id',
                          'product_category', 'quantity', 'unit_price', 'total_amount']
        self.assertTrue(all(col in data.columns for col in expected_columns))
        
        # Check if number of unique customers is reasonable
        self.assertTrue(data['customer_id'].nunique() <= 50)
        self.assertTrue(data['customer_id'].nunique() > 0)
        
        # Check if all amounts are positive
        self.assertTrue((data['total_amount'] > 0).all())
        self.assertTrue((data['quantity'] > 0).all())
        self.assertTrue((data['unit_price'] > 0).all())
        
        # Check if dates are within expected range
        self.assertTrue(pd.to_datetime(data['order_date']).dt.date.max() <= pd.Timestamp.now().date())
    
    def test_get_data_summary(self):
        """Test data summary generation."""
        data = generate_ecommerce_data(n_customers=30, random_state=42)
        summary = get_data_summary(data)
        
        # Check if summary contains expected keys
        expected_keys = ['total_transactions', 'unique_customers', 'unique_products',
                        'unique_orders', 'date_range', 'total_revenue', 'avg_order_value',
                        'categories', 'category_distribution']
        self.assertTrue(all(key in summary for key in expected_keys))
        
        # Check if values make sense
        self.assertTrue(summary['unique_customers'] <= 30)
        self.assertTrue(summary['total_revenue'] > 0)
        self.assertTrue(summary['avg_order_value'] > 0)


class TestOptimalClusters(unittest.TestCase):
    """Test cases for optimal cluster selection."""
    
    def test_find_optimal_clusters(self):
        """Test optimal cluster finding functionality."""
        test_data = generate_ecommerce_data(n_customers=100, random_state=42)
        
        optimal_k, wcss_values = find_optimal_clusters(test_data, max_clusters=6)
        
        # Check if optimal k is reasonable
        self.assertTrue(2 <= optimal_k <= 6)
        
        # Check if WCSS values decrease
        self.assertEqual(len(wcss_values), 6)
        self.assertTrue(all(wcss_values[i] >= wcss_values[i+1] for i in range(len(wcss_values)-1)))


if __name__ == '__main__':
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)