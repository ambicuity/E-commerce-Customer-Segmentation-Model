#!/usr/bin/env python3
"""
Customer Segmentation Analysis Script
====================================

Example script demonstrating how to use the customer segmentation model
for e-commerce data analysis and marketing strategy development.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import matplotlib.pyplot as plt
from customer_segmentation import CustomerSegmentationModel, find_optimal_clusters
from data_generator import generate_ecommerce_data, get_data_summary


def main():
    """Main function to demonstrate customer segmentation workflow."""
    
    print("=" * 60)
    print("E-COMMERCE CUSTOMER SEGMENTATION ANALYSIS")
    print("=" * 60)
    print()
    
    # Step 1: Generate sample data
    print("1. GENERATING SAMPLE E-COMMERCE DATA")
    print("-" * 40)
    
    # Generate larger dataset for demonstration
    data = generate_ecommerce_data(
        n_customers=800,
        n_products=75,
        date_range_days=365,
        random_state=42
    )
    
    print(f"✓ Generated {len(data):,} transactions")
    print(f"✓ {data['customer_id'].nunique():,} unique customers")
    print(f"✓ {data['order_id'].nunique():,} unique orders")
    print(f"✓ Date range: {data['order_date'].min()} to {data['order_date'].max()}")
    
    # Save data
    data_path = os.path.join('..', 'data', 'sample_data.csv')
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    data.to_csv(data_path, index=False)
    print(f"✓ Data saved to {data_path}")
    print()
    
    # Step 2: Data summary
    print("2. DATA SUMMARY")
    print("-" * 40)
    
    summary = get_data_summary(data)
    for key, value in summary.items():
        if key != 'category_distribution':
            print(f"{key.replace('_', ' ').title()}: {value}")
    
    print("\nProduct Category Distribution:")
    for category, count in summary['category_distribution'].items():
        percentage = (count / summary['total_transactions']) * 100
        print(f"  {category}: {count:,} transactions ({percentage:.1f}%)")
    print()
    
    # Step 3: Find optimal number of clusters
    print("3. FINDING OPTIMAL NUMBER OF CLUSTERS")
    print("-" * 40)
    
    print("Analyzing optimal cluster count using elbow method...")
    optimal_k, wcss_values = find_optimal_clusters(data, max_clusters=8)
    
    print(f"✓ Optimal number of clusters: {optimal_k}")
    print("✓ WCSS values for different k:", [round(w, 0) for w in wcss_values])
    print()
    
    # Step 4: Build segmentation model
    print("4. BUILDING CUSTOMER SEGMENTATION MODEL")
    print("-" * 40)
    
    # Create and train model
    model = CustomerSegmentationModel(n_clusters=optimal_k, random_state=42)
    print(f"✓ Initialized model with {optimal_k} clusters")
    
    # Fit model
    model.fit(data)
    print("✓ Model training completed")
    
    # Step 5: Evaluate model
    print("5. MODEL EVALUATION")
    print("-" * 40)
    
    evaluation = model.evaluate_model(data)
    print(f"Silhouette Score: {evaluation['silhouette_score']:.3f}")
    print(f"Number of Customers: {evaluation['n_customers']:,}")
    print(f"Number of Clusters: {evaluation['n_clusters']}")
    print()
    
    # Step 6: Analyze segments
    print("6. CUSTOMER SEGMENT ANALYSIS")
    print("-" * 40)
    
    # Get cluster summary
    summary = model.get_cluster_summary(data)
    print("Segment Summary Statistics:")
    print(summary.round(2))
    print()
    
    # Generate business insights
    insights = model.get_segment_insights(data)
    print("BUSINESS INSIGHTS FOR EACH SEGMENT:")
    print("=" * 50)
    
    for cluster, insight in insights.items():
        customer_features = model.prepare_features(data)
        customer_features['cluster'] = model.labels_
        cluster_size = len(customer_features[customer_features['cluster'] == cluster])
        percentage = (cluster_size / len(customer_features)) * 100
        
        print(f"\nSEGMENT {cluster} ({cluster_size} customers - {percentage:.1f}%)")
        print("-" * 30)
        print(f"Insight: {insight}")
        
        # Detailed statistics
        cluster_data = customer_features[customer_features['cluster'] == cluster]
        print(f"Average Recency: {cluster_data['recency'].mean():.1f} days")
        print(f"Average Frequency: {cluster_data['frequency'].mean():.1f} orders")
        print(f"Average Monetary: ${cluster_data['monetary'].mean():.2f}")
        print(f"Average Order Value: ${cluster_data['avg_order_value'].mean():.2f}")
    
    print()
    
    # Step 7: Marketing recommendations
    print("7. MARKETING STRATEGY RECOMMENDATIONS")
    print("-" * 40)
    
    print("RECOMMENDED MARKETING ACTIONS:")
    print("=" * 35)
    
    for cluster, insight in insights.items():
        customer_features = model.prepare_features(data)
        customer_features['cluster'] = model.labels_
        cluster_data = customer_features[customer_features['cluster'] == cluster]
        
        print(f"\nSEGMENT {cluster}:")
        
        if "VIP" in insight:
            print("• Implement VIP loyalty program with exclusive benefits")
            print("• Offer early access to new products")
            print("• Provide premium customer support")
            print("• Send personalized product recommendations")
        
        elif "Loyal" in insight:
            print("• Create targeted cross-selling campaigns")
            print("• Offer bundle discounts on complementary products")
            print("• Implement referral reward programs")
            print("• Send regular newsletters with product updates")
        
        elif "At-Risk" in insight:
            print("• Launch win-back email campaigns with special offers")
            print("• Provide limited-time discount codes")
            print("• Send surveys to understand why they stopped buying")
            print("• Retarget with social media advertising")
        
        elif "New" in insight or "Low-Value" in insight:
            print("• Create onboarding email series")
            print("• Offer new customer welcome discounts")
            print("• Provide product education and tutorials")
            print("• Implement progressive incentive programs")
        
        else:
            print("• Run targeted promotional campaigns")
            print("• Test different communication channels")
            print("• Offer seasonal discounts and promotions")
            print("• Monitor engagement and adjust strategy")
    
    print()
    
    # Step 8: Save results
    print("8. SAVING RESULTS")
    print("-" * 40)
    
    # Prepare customer segmentation results
    customer_features = model.prepare_features(data)
    customer_features['segment'] = model.labels_
    
    # Add segment insights
    customer_features['segment_description'] = customer_features['segment'].map(insights)
    
    # Save segmented customers
    results_path = os.path.join('..', 'data', 'customer_segments.csv')
    customer_features.to_csv(results_path, index=False)
    print(f"✓ Customer segments saved to {results_path}")
    
    # Create visualization (save plot)
    print("✓ Generating visualization...")
    plt.style.use('default')
    model.plot_segments(data, figsize=(16, 12))
    plot_path = os.path.join('..', 'data', 'segmentation_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Segmentation plots saved to {plot_path}")
    
    print()
    print("=" * 60)
    print("ANALYSIS COMPLETE!")
    print("=" * 60)
    print("Next Steps:")
    print("• Review the generated insights and recommendations")
    print("• Implement suggested marketing strategies")
    print("• Monitor segment performance over time")
    print("• Re-run analysis periodically with new data")
    print()


if __name__ == "__main__":
    main()