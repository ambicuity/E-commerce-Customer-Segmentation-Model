# E-commerce Customer Segmentation Model

A comprehensive customer segmentation solution for e-commerce businesses using machine learning techniques with scikit-learn and pandas. This project demonstrates the application of unsupervised learning to analyze customer behavior and identify distinct customer groups for targeted marketing strategies.

## 🎯 Project Overview

This project implements a customer segmentation model that:
- **Analyzes large-scale e-commerce transaction data** to identify distinct customer groups
- **Uses RFM analysis** (Recency, Frequency, Monetary) combined with additional behavioral features
- **Applies K-means clustering** to segment customers into meaningful groups
- **Provides business insights** and marketing recommendations for each segment
- **Demonstrates machine learning fundamentals** and data wrangling techniques

## 🚀 Features

- **Automated Customer Segmentation**: K-means clustering with optimal cluster selection
- **RFM Analysis**: Comprehensive customer behavior analysis using Recency, Frequency, and Monetary metrics
- **Business Intelligence**: Automated generation of actionable insights for each customer segment
- **Data Visualization**: Rich plotting capabilities for segment analysis and interpretation
- **Synthetic Data Generation**: Built-in capability to generate realistic e-commerce transaction data
- **Marketing Strategy Integration**: Specific recommendations aligned with business goals
- **Comprehensive Testing**: Full test suite ensuring model reliability

## 📊 Customer Segments Identified

The model typically identifies segments such as:
- **VIP Customers**: High-value, frequent, recent buyers
- **Loyal Customers**: Regular buyers with consistent engagement
- **At-Risk Customers**: Previously active customers who may churn
- **New/Potential Customers**: Limited engagement, growth opportunity

## 🛠️ Installation

### Prerequisites
- Python 3.7+
- pip package manager

### Setup
1. Clone the repository:
```bash
git clone https://github.com/ambicuity/E-commerce-Customer-Segmentation-Model.git
cd E-commerce-Customer-Segmentation-Model
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📈 Quick Start

### Option 1: Run Complete Analysis Script
```bash
cd scripts
python run_segmentation_analysis.py
```

This will:
- Generate sample e-commerce data
- Find optimal number of clusters
- Train the segmentation model  
- Analyze customer segments
- Generate business insights and marketing recommendations
- Save results and visualizations

### Option 2: Use in Your Code
```python
from src.customer_segmentation import CustomerSegmentationModel
from src.data_generator import generate_ecommerce_data

# Generate or load your data
data = generate_ecommerce_data(n_customers=1000)

# Create and train model
model = CustomerSegmentationModel(n_clusters=4)
model.fit(data)

# Get predictions
segments = model.predict(data)

# Analyze segments
insights = model.get_segment_insights(data)
model.plot_segments(data)
```

## 📁 Project Structure

```
E-commerce-Customer-Segmentation-Model/
├── src/                          # Source code
│   ├── customer_segmentation.py  # Main segmentation model
│   └── data_generator.py         # Synthetic data generation
├── scripts/                      # Executable scripts
│   └── run_segmentation_analysis.py
├── tests/                        # Unit tests
│   └── test_segmentation.py
├── notebooks/                    # Jupyter notebooks (optional)
├── data/                        # Data directory
│   ├── sample_data.csv          # Generated sample data
│   ├── customer_segments.csv    # Segmentation results
│   └── segmentation_analysis.png # Visualization output
├── requirements.txt             # Python dependencies
├── README.md                   # This file
└── LICENSE                     # Apache 2.0 License
```

## 🔬 Technical Details

### Features Used for Segmentation
- **Recency**: Days since last purchase
- **Frequency**: Total number of orders
- **Monetary**: Total amount spent
- **Average Order Value**: Average spending per order
- **Total Quantity**: Total items purchased
- **Category Diversity**: Number of different product categories purchased

### Machine Learning Approach
- **Algorithm**: K-means clustering with StandardScaler normalization
- **Optimization**: Elbow method for optimal cluster selection
- **Evaluation**: Silhouette score for cluster quality assessment
- **Validation**: Comprehensive unit testing and cross-validation

### Business Integration
- **Customer Profiling**: Detailed analysis of each segment's characteristics
- **Marketing Recommendations**: Specific, actionable strategies for each segment
- **Performance Metrics**: Business-relevant KPIs and segment health indicators

## 🧪 Testing

Run the test suite to validate model functionality:

```bash
cd tests
python test_segmentation.py
```

The tests cover:
- Model initialization and fitting
- Feature preparation and validation
- Prediction accuracy
- Data generation consistency
- Edge cases and error handling

## 📊 Sample Output

The model generates comprehensive analysis including:

### Segment Summary
```
Segment 0 (VIP Customers - 15.2%):
- Average Recency: 28.3 days
- Average Frequency: 12.7 orders  
- Average Monetary: $2,847.32
- Recommended Strategy: Premium retention programs

Segment 1 (Loyal Customers - 31.4%):
- Average Recency: 45.8 days
- Average Frequency: 6.2 orders
- Average Monetary: $1,243.76
- Recommended Strategy: Cross-selling and upselling
```

### Visualizations
- Customer distribution across segments
- RFM analysis scatter plots
- Segment characteristics box plots
- Purchase behavior patterns

## 🎯 Business Applications

This model enables data-driven marketing strategies:

1. **Personalized Marketing**: Tailor campaigns to each segment's characteristics
2. **Customer Retention**: Identify at-risk customers for proactive engagement
3. **Revenue Optimization**: Focus resources on high-value customer segments
4. **Product Strategy**: Understand product preferences across segments
5. **Lifecycle Management**: Guide customers through value progression stages

## 🤝 Collaboration Features

The project simulates collaboration with product analysts by:
- **Clear Documentation**: Business-friendly explanations of technical concepts
- **Configurable Parameters**: Easy adjustment of model parameters for different business needs
- **Interpretable Results**: Human-readable insights and recommendations
- **Business Metrics**: Focus on KPIs that matter to stakeholders

## 📚 Dependencies

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms
- **matplotlib**: Data visualization
- **seaborn**: Statistical data visualization
- **jupyter**: Interactive development environment
- **plotly**: Interactive visualizations (optional)

## 🔄 Future Enhancements

Potential improvements and extensions:
- **Real-time Segmentation**: Stream processing for live customer data
- **Advanced Features**: Seasonal patterns, product affinity, geographic data
- **Deep Learning**: Neural network-based customer embeddings
- **A/B Testing Framework**: Campaign performance measurement
- **Customer Lifetime Value**: Predictive CLV modeling

## 📝 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙋 Support

For questions, issues, or contributions:
1. Check existing issues in the GitHub repository
2. Create a new issue with detailed description
3. Follow the contribution guidelines for pull requests

## 📞 Contact

This project was developed to demonstrate machine learning fundamentals and data-driven marketing strategy development in e-commerce contexts.

---

**Note**: This is a demonstration project using synthetic data. For production use with real customer data, ensure compliance with privacy regulations and data protection laws.