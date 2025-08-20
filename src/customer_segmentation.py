"""
E-commerce Customer Segmentation Model
=====================================

A comprehensive customer segmentation solution for e-commerce businesses
using machine learning techniques with scikit-learn and pandas.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')


class CustomerSegmentationModel:
    """
    A customer segmentation model using K-means clustering to identify
    distinct customer groups based on their purchasing behavior.
    """
    
    def __init__(self, n_clusters: int = 4, random_state: int = 42):
        """
        Initialize the customer segmentation model.
        
        Args:
            n_clusters (int): Number of customer segments to create
            random_state (int): Random state for reproducibility
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names = []
        self.cluster_centers_ = None
        self.labels_ = None
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for customer segmentation.
        
        Args:
            df (pd.DataFrame): Customer transaction data
            
        Returns:
            pd.DataFrame: Prepared features for clustering
        """
        # Create RFM features (Recency, Frequency, Monetary)
        customer_features = df.groupby('customer_id').agg({
            'order_date': lambda x: (df['order_date'].max() - x.max()).days,  # Recency
            'order_id': 'count',  # Frequency
            'total_amount': 'sum',  # Monetary
            'quantity': 'sum',  # Total items purchased
            'product_category': lambda x: len(x.unique())  # Category diversity
        }).reset_index()
        
        customer_features.columns = ['customer_id', 'recency', 'frequency', 'monetary', 'total_quantity', 'category_diversity']
        
        # Add average order value
        customer_features['avg_order_value'] = customer_features['monetary'] / customer_features['frequency']
        
        # Store feature names for later use
        self.feature_names = ['recency', 'frequency', 'monetary', 'total_quantity', 'category_diversity', 'avg_order_value']
        
        return customer_features
    
    def fit(self, df: pd.DataFrame) -> 'CustomerSegmentationModel':
        """
        Fit the customer segmentation model on the provided data.
        
        Args:
            df (pd.DataFrame): Customer transaction data
            
        Returns:
            CustomerSegmentationModel: Fitted model instance
        """
        # Prepare features
        customer_features = self.prepare_features(df)
        
        # Extract features for clustering
        X = customer_features[self.feature_names].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit KMeans model
        self.kmeans.fit(X_scaled)
        
        # Store results
        self.cluster_centers_ = self.scaler.inverse_transform(self.kmeans.cluster_centers_)
        self.labels_ = self.kmeans.labels_
        self.is_fitted = True
        
        return self
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict customer segments for new data.
        
        Args:
            df (pd.DataFrame): Customer transaction data
            
        Returns:
            np.ndarray: Predicted cluster labels
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        customer_features = self.prepare_features(df)
        X = customer_features[self.feature_names].values
        X_scaled = self.scaler.transform(X)
        
        return self.kmeans.predict(X_scaled)
    
    def fit_predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Fit the model and predict customer segments.
        
        Args:
            df (pd.DataFrame): Customer transaction data
            
        Returns:
            np.ndarray: Predicted cluster labels
        """
        return self.fit(df).predict(df)
    
    def get_cluster_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get summary statistics for each customer segment.
        
        Args:
            df (pd.DataFrame): Customer transaction data
            
        Returns:
            pd.DataFrame: Cluster summary statistics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting cluster summary")
            
        customer_features = self.prepare_features(df)
        customer_features['cluster'] = self.labels_
        
        summary = customer_features.groupby('cluster')[self.feature_names].agg([
            'count', 'mean', 'median', 'std'
        ]).round(2)
        
        return summary
    
    def evaluate_model(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate the clustering model using silhouette score.
        
        Args:
            df (pd.DataFrame): Customer transaction data
            
        Returns:
            Dict[str, float]: Model evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
            
        customer_features = self.prepare_features(df)
        X = customer_features[self.feature_names].values
        X_scaled = self.scaler.transform(X)
        
        silhouette_avg = silhouette_score(X_scaled, self.labels_)
        
        return {
            'silhouette_score': silhouette_avg,
            'n_clusters': self.n_clusters,
            'n_customers': len(customer_features)
        }
    
    def plot_segments(self, df: pd.DataFrame, figsize: Tuple[int, int] = (15, 10)) -> None:
        """
        Visualize customer segments using various plots.
        
        Args:
            df (pd.DataFrame): Customer transaction data
            figsize (Tuple[int, int]): Figure size for plots
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before plotting")
            
        customer_features = self.prepare_features(df)
        customer_features['cluster'] = self.labels_
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('Customer Segmentation Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Recency vs Frequency
        axes[0, 0].scatter(customer_features['recency'], customer_features['frequency'], 
                          c=customer_features['cluster'], cmap='viridis', alpha=0.6)
        axes[0, 0].set_xlabel('Recency (days)')
        axes[0, 0].set_ylabel('Frequency (orders)')
        axes[0, 0].set_title('Recency vs Frequency')
        
        # Plot 2: Frequency vs Monetary
        axes[0, 1].scatter(customer_features['frequency'], customer_features['monetary'], 
                          c=customer_features['cluster'], cmap='viridis', alpha=0.6)
        axes[0, 1].set_xlabel('Frequency (orders)')
        axes[0, 1].set_ylabel('Monetary (total spent)')
        axes[0, 1].set_title('Frequency vs Monetary')
        
        # Plot 3: Cluster distribution
        cluster_counts = customer_features['cluster'].value_counts().sort_index()
        axes[0, 2].bar(cluster_counts.index, cluster_counts.values, 
                      color=plt.cm.viridis(np.linspace(0, 1, len(cluster_counts))))
        axes[0, 2].set_xlabel('Cluster')
        axes[0, 2].set_ylabel('Number of Customers')
        axes[0, 2].set_title('Customer Distribution by Segment')
        
        # Plot 4: Average order value distribution
        axes[1, 0].boxplot([customer_features[customer_features['cluster'] == i]['avg_order_value'] 
                           for i in range(self.n_clusters)])
        axes[1, 0].set_xlabel('Cluster')
        axes[1, 0].set_ylabel('Average Order Value')
        axes[1, 0].set_title('Average Order Value by Segment')
        
        # Plot 5: Recency distribution
        axes[1, 1].boxplot([customer_features[customer_features['cluster'] == i]['recency'] 
                           for i in range(self.n_clusters)])
        axes[1, 1].set_xlabel('Cluster')
        axes[1, 1].set_ylabel('Recency (days)')
        axes[1, 1].set_title('Recency by Segment')
        
        # Plot 6: Category diversity
        axes[1, 2].boxplot([customer_features[customer_features['cluster'] == i]['category_diversity'] 
                           for i in range(self.n_clusters)])
        axes[1, 2].set_xlabel('Cluster')
        axes[1, 2].set_ylabel('Category Diversity')
        axes[1, 2].set_title('Category Diversity by Segment')
        
        plt.tight_layout()
        plt.show()
    
    def get_segment_insights(self, df: pd.DataFrame) -> Dict[int, str]:
        """
        Generate business insights for each customer segment.
        
        Args:
            df (pd.DataFrame): Customer transaction data
            
        Returns:
            Dict[int, str]: Insights for each segment
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before generating insights")
            
        customer_features = self.prepare_features(df)
        customer_features['cluster'] = self.labels_
        
        insights = {}
        
        for cluster in range(self.n_clusters):
            cluster_data = customer_features[customer_features['cluster'] == cluster]
            
            avg_recency = cluster_data['recency'].mean()
            avg_frequency = cluster_data['frequency'].mean()
            avg_monetary = cluster_data['monetary'].mean()
            avg_order_value = cluster_data['avg_order_value'].mean()
            
            if avg_recency < 30 and avg_frequency > 5 and avg_monetary > customer_features['monetary'].quantile(0.75):
                insights[cluster] = "VIP Customers: High value, frequent, recent buyers. Focus on retention and premium offerings."
            elif avg_frequency > customer_features['frequency'].median() and avg_monetary > customer_features['monetary'].median():
                insights[cluster] = "Loyal Customers: Regular buyers with good spend. Ideal for cross-selling and upselling."
            elif avg_recency > customer_features['recency'].quantile(0.75):
                insights[cluster] = "At-Risk Customers: Haven't purchased recently. Target with win-back campaigns."
            elif avg_frequency < 2 and avg_monetary < customer_features['monetary'].quantile(0.25):
                insights[cluster] = "New/Low-Value Customers: Limited engagement. Focus on onboarding and activation."
            else:
                insights[cluster] = f"Moderate Customers: Average engagement level. Consider targeted promotions to increase activity."
        
        return insights


def find_optimal_clusters(df: pd.DataFrame, max_clusters: int = 10) -> Tuple[int, List[float]]:
    """
    Find the optimal number of clusters using the elbow method.
    
    Args:
        df (pd.DataFrame): Customer transaction data
        max_clusters (int): Maximum number of clusters to test
        
    Returns:
        Tuple[int, List[float]]: Optimal number of clusters and list of WCSS values
    """
    wcss = []
    
    # Prepare features
    model = CustomerSegmentationModel()
    customer_features = model.prepare_features(df)
    X = customer_features[model.feature_names].values
    X_scaled = StandardScaler().fit_transform(X)
    
    # Test different number of clusters
    for i in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)
    
    # Simple elbow detection (find the point with maximum change in slope)
    differences = [wcss[i] - wcss[i+1] for i in range(len(wcss)-1)]
    optimal_k = differences.index(max(differences)) + 2  # +2 because we start from k=1 and take the next point
    
    return optimal_k, wcss