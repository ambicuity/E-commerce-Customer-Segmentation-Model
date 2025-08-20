"""
E-commerce Customer Segmentation Model Package
=============================================

This package provides tools for customer segmentation analysis in e-commerce
businesses using machine learning techniques with scikit-learn and pandas.
"""

__version__ = "1.0.0"
__author__ = "E-commerce Analytics Team"

from .customer_segmentation import CustomerSegmentationModel, find_optimal_clusters
from .data_generator import generate_ecommerce_data, load_sample_data, get_data_summary

__all__ = [
    'CustomerSegmentationModel',
    'find_optimal_clusters',
    'generate_ecommerce_data',
    'load_sample_data',
    'get_data_summary'
]