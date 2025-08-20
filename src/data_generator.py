"""
Data Generation Module
=====================

Generates synthetic e-commerce customer data for demonstration and testing
purposes of the customer segmentation model.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple
import random


def generate_ecommerce_data(
    n_customers: int = 1000,
    n_products: int = 100,
    date_range_days: int = 365,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic e-commerce transaction data.
    
    Args:
        n_customers (int): Number of unique customers
        n_products (int): Number of unique products
        date_range_days (int): Number of days to generate data for
        random_state (int): Random seed for reproducibility
        
    Returns:
        pd.DataFrame: Generated transaction data
    """
    np.random.seed(random_state)
    random.seed(random_state)
    
    # Product categories and their characteristics
    categories = {
        'Electronics': {'avg_price': 299, 'price_std': 150, 'purchase_frequency': 0.1},
        'Clothing': {'avg_price': 59, 'price_std': 30, 'purchase_frequency': 0.3},
        'Books': {'avg_price': 19, 'price_std': 10, 'purchase_frequency': 0.2},
        'Home & Garden': {'avg_price': 89, 'price_std': 50, 'purchase_frequency': 0.15},
        'Sports': {'avg_price': 79, 'price_std': 40, 'purchase_frequency': 0.15},
        'Beauty': {'avg_price': 39, 'price_std': 20, 'purchase_frequency': 0.25}
    }
    
    # Generate customer profiles
    customer_types = {
        'VIP': {'purchase_prob': 0.8, 'avg_order_frequency': 15, 'price_sensitivity': 0.2},
        'Regular': {'purchase_prob': 0.4, 'avg_order_frequency': 8, 'price_sensitivity': 0.5},
        'Occasional': {'purchase_prob': 0.15, 'avg_order_frequency': 3, 'price_sensitivity': 0.8},
        'New': {'purchase_prob': 0.05, 'avg_order_frequency': 1, 'price_sensitivity': 0.9}
    }
    
    # Assign customer types
    customer_type_distribution = [0.1, 0.3, 0.4, 0.2]  # VIP, Regular, Occasional, New
    customer_types_list = np.random.choice(
        list(customer_types.keys()),
        size=n_customers,
        p=customer_type_distribution
    )
    
    # Generate date range
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=date_range_days)
    
    transactions = []
    
    for customer_id in range(1, n_customers + 1):
        customer_type = customer_types_list[customer_id - 1]
        customer_profile = customer_types[customer_type]
        
        # Generate number of orders for this customer
        n_orders = np.random.poisson(customer_profile['avg_order_frequency'])
        
        for _ in range(n_orders):
            # Random order date (more recent dates have higher probability for active customers)
            if customer_type in ['VIP', 'Regular']:
                # Recent customers - higher chance of recent purchases
                days_ago = np.random.exponential(30)
            else:
                # Less active customers - more uniform distribution
                days_ago = np.random.uniform(0, date_range_days)
            
            order_date = end_date - timedelta(days=int(days_ago))
            order_date = max(order_date, start_date)  # Ensure within range
            
            # Generate order ID
            order_id = f"ORD_{customer_id}_{random.randint(100000, 999999)}"
            
            # Generate number of items in this order
            if customer_type == 'VIP':
                n_items = np.random.choice([1, 2, 3, 4], p=[0.3, 0.3, 0.25, 0.15])
            elif customer_type == 'Regular':
                n_items = np.random.choice([1, 2, 3], p=[0.5, 0.35, 0.15])
            else:
                n_items = np.random.choice([1, 2], p=[0.8, 0.2])
            
            for _ in range(n_items):
                # Select category
                freq_probs = [cat['purchase_frequency'] for cat in categories.values()]
                # Normalize probabilities to sum to 1
                freq_probs = [p / sum(freq_probs) for p in freq_probs]
                category = np.random.choice(
                    list(categories.keys()),
                    p=freq_probs
                )
                
                category_info = categories[category]
                
                # Generate product
                product_id = f"PROD_{category[:3].upper()}_{random.randint(1, n_products)}"
                
                # Generate price based on category and customer type
                base_price = np.random.normal(
                    category_info['avg_price'],
                    category_info['price_std']
                )
                base_price = max(base_price, 5)  # Minimum price
                
                # Apply customer type discount/premium
                if customer_type == 'VIP':
                    price = base_price * np.random.uniform(0.85, 1.1)  # VIP gets discounts sometimes
                elif customer_type == 'New':
                    price = base_price * np.random.uniform(0.9, 0.95)  # New customer discounts
                else:
                    price = base_price
                
                # Generate quantity
                if customer_type == 'VIP':
                    quantity = np.random.choice([1, 2, 3], p=[0.6, 0.3, 0.1])
                else:
                    quantity = np.random.choice([1, 2], p=[0.85, 0.15])
                
                total_amount = round(price * quantity, 2)
                
                transactions.append({
                    'customer_id': customer_id,
                    'order_id': order_id,
                    'order_date': order_date,
                    'product_id': product_id,
                    'product_category': category,
                    'quantity': quantity,
                    'unit_price': round(price, 2),
                    'total_amount': total_amount
                })
    
    df = pd.DataFrame(transactions)
    
    # Sort by date
    df = df.sort_values('order_date').reset_index(drop=True)
    
    return df


def load_sample_data() -> pd.DataFrame:
    """
    Load or generate sample e-commerce data for demonstration.
    
    Returns:
        pd.DataFrame: Sample transaction data
    """
    return generate_ecommerce_data(
        n_customers=500,
        n_products=50,
        date_range_days=365,
        random_state=42
    )


def get_data_summary(df: pd.DataFrame) -> dict:
    """
    Get summary statistics of the e-commerce data.
    
    Args:
        df (pd.DataFrame): Transaction data
        
    Returns:
        dict: Summary statistics
    """
    summary = {
        'total_transactions': len(df),
        'unique_customers': df['customer_id'].nunique(),
        'unique_products': df['product_id'].nunique(),
        'unique_orders': df['order_id'].nunique(),
        'date_range': f"{df['order_date'].min()} to {df['order_date'].max()}",
        'total_revenue': df['total_amount'].sum(),
        'avg_order_value': df.groupby('order_id')['total_amount'].sum().mean(),
        'categories': df['product_category'].unique().tolist(),
        'category_distribution': df['product_category'].value_counts().to_dict()
    }
    
    return summary


if __name__ == "__main__":
    # Generate sample data and save to CSV
    print("Generating sample e-commerce data...")
    data = generate_ecommerce_data(n_customers=1000, random_state=42)
    
    # Save to data directory
    data.to_csv('../data/sample_ecommerce_data.csv', index=False)
    print(f"Generated {len(data)} transactions for {data['customer_id'].nunique()} customers")
    print(f"Data saved to ../data/sample_ecommerce_data.csv")
    
    # Print summary
    summary = get_data_summary(data)
    print("\nData Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")