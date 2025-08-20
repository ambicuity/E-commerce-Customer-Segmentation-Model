#!/usr/bin/env python3
"""
Simple Customer Segmentation Example
====================================

A quick example showing how to use the customer segmentation model
for basic analysis.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from customer_segmentation import CustomerSegmentationModel
from data_generator import generate_ecommerce_data


def main():
    """Run a simple customer segmentation example."""
    
    print("Simple Customer Segmentation Example")
    print("=" * 50)
    
    # Step 1: Generate sample data
    print("\n1. Generating sample data...")
    data = generate_ecommerce_data(n_customers=200, random_state=42)
    print(f"✓ Generated {len(data)} transactions for {data['customer_id'].nunique()} customers")
    
    # Step 2: Create and train model
    print("\n2. Training segmentation model...")
    model = CustomerSegmentationModel(n_clusters=3, random_state=42)
    model.fit(data)
    print("✓ Model trained successfully")
    
    # Step 3: Get predictions
    print("\n3. Predicting customer segments...")
    segments = model.predict(data)
    print(f"✓ Assigned {len(set(segments))} segments to {len(set(model.prepare_features(data)['customer_id']))} customers")
    
    # Step 4: Analyze results
    print("\n4. Analysis Results:")
    print("-" * 20)
    
    # Model evaluation
    evaluation = model.evaluate_model(data)
    print(f"Silhouette Score: {evaluation['silhouette_score']:.3f}")
    
    # Segment insights
    insights = model.get_segment_insights(data)
    customer_features = model.prepare_features(data)
    customer_features['cluster'] = model.labels_
    
    for cluster, insight in insights.items():
        cluster_size = len(customer_features[customer_features['cluster'] == cluster])
        percentage = (cluster_size / len(customer_features)) * 100
        print(f"\nSegment {cluster} ({cluster_size} customers - {percentage:.1f}%):")
        print(f"  {insight}")
    
    print("\n" + "=" * 50)
    print("Analysis complete! See full analysis script for detailed insights.")


if __name__ == "__main__":
    main()