import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_pairwise_relationships(df):
    """
    Plot pairwise relationships between the top 4 most correlated features and wine quality.
    
    Args:
        df (pd.DataFrame): DataFrame containing wine data with 'quality' column
    """
    # Get four most correlated features from dataframe
    correlation_matrix = df.corr()
    feature_importance = correlation_matrix['quality'].abs().sort_values(ascending=False)
    top_correlations = feature_importance.head(4).index.tolist()  # Convert important features to list
    
    plt.figure(figsize=(16, 12))
    for i, feature in enumerate(top_correlations, 1):
        plt.subplot(2, 2, i) 
        scatter = plt.scatter(df[feature], df['quality'], c=df['quality'], 
                             cmap='viridis', alpha=0.6, s=50)
        
        z = np.polyfit(df[feature], df['quality'], 1)  # Get the polynomial function slope and intercept after fitting
        p = np.poly1d(z)  # Obtain the equation of straight-line, i.e. linear regression p(x) = slope * x + intercept
        plt.plot(df[feature], p(df[feature]), "r--", alpha=0.8, linewidth=2)
        
        plt.title(f'{feature.title()} vs Quality', fontweight='bold')
        plt.xlabel(feature.title())
        plt.ylabel('Quality Score')
        plt.colorbar(scatter, label='Quality Score')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Example usage:
# df = pd.read_csv('your_wine_data.csv')
# plot_pairwise_relationships(df) 