"""
🍷 Wine Quality Dataset - Comprehensive Data Analysis & Visualization with Plotly
This script provides comprehensive interactive visualizations to extract insights from the wine quality dataset.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.subplots as sp
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

def load_and_explore_data():
    """Load and explore the wine quality dataset."""
    print("Loading wine quality dataset...")
    
    # Load the dataset
    df = pd.read_csv("WineQT.csv")
    
    # Display basic information
    print(f"\nDataset Shape: {df.shape}")
    print(f"Features: {list(df.columns)}")
    print(f"Target Variable: quality")
    
    print("\nFirst few rows:")
    print(df.head())
    
    print("\nDataset Info:")
    print(df.info())
    
    print("\nBasic Statistics:")
    print(df.describe())
    
    return df

def analyze_quality_distribution(df):
    """Analyze the distribution of wine quality scores using Plotly."""
    print("\n" + "="*50)
    print("1. 📊 TARGET VARIABLE DISTRIBUTION (QUALITY)")
    print("="*50)
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Wine Quality Distribution', 'Quality Distribution (Percentage)', 'Quality Categories'),
        specs=[[{"type": "bar"}, {"type": "pie"}, {"type": "bar"}]]
    )
    
    # Subplot 1: Bar plot
    quality_counts = df['quality'].value_counts().sort_index()
    fig.add_trace(
        go.Bar(
            x=quality_counts.index,
            y=quality_counts.values,
            name='Quality Counts',
            marker_color='crimson',
            text=quality_counts.values,
            textposition='auto'
        ),
        row=1, col=1
    )
    
    # Subplot 2: Pie chart
    fig.add_trace(
        go.Pie(
            labels=quality_counts.index,
            values=quality_counts.values,
            name='Quality Distribution',
            textinfo='label+percent'
        ),
        row=1, col=2
    )
    
    # Subplot 3: Quality categories
    quality_categories = pd.cut(df['quality'], bins=[0, 5, 6, 7, 10], 
                               labels=['Poor (3-5)', 'Average (6)', 'Good (7)', 'Excellent (8-9)'])
    category_counts = quality_categories.value_counts()
    
    fig.add_trace(
        go.Bar(
            x=list(category_counts.index),
            y=category_counts.values,
            name='Quality Categories',
            marker_color=['#ff6b6b', '#feca57', '#48dbfb', '#0abde3'],
            text=category_counts.values,
            textposition='auto'
        ),
        row=1, col=3
    )
    
    # Update layout
    fig.update_layout(
        title_text="Wine Quality Distribution Analysis",
        showlegend=False,
        height=500
    )
    
    fig.show()
    
    # Insights
    print("\n🔍 INSIGHTS FROM QUALITY DISTRIBUTION:")
    print(f"• Most wines have quality score 6 ({quality_counts[6]} wines, {quality_counts[6]/len(df)*100:.1f}%)")
    print(f"• Quality scores range from {df['quality'].min()} to {df['quality'].max()}")
    print(f"• {len(df[df['quality'] >= 7])} wines ({len(df[df['quality'] >= 7])/len(df)*100:.1f}%) are high quality (≥7)")
    print(f"• {len(df[df['quality'] <= 5])} wines ({len(df[df['quality'] <= 5])/len(df)*100:.1f}%) are low quality (≤5)")
    
    return quality_counts

def analyze_feature_distributions(df):
    """Analyze the distribution of each chemical property using Plotly."""
    print("\n" + "="*50)
    print("2. 📈 FEATURE DISTRIBUTIONS")
    print("="*50)
    
    features = df.drop('quality', axis=1).columns
    n_features = len(features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    # Create subplots
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[f'{feature.title()} Distribution' for feature in features],
        specs=[[{"secondary_y": False} for _ in range(n_cols)] for _ in range(n_rows)]
    )
    
    for i, feature in enumerate(features):
        row = (i // n_cols) + 1
        col = (i % n_cols) + 1
        
        # Histogram with KDE
        fig.add_trace(
            go.Histogram(
                x=df[feature],
                name=feature,
                nbinsx=30,
                marker_color='skyblue',
                opacity=0.7
            ),
            row=row, col=col
        )
        
        # Add mean and median lines
        mean_val = df[feature].mean()
        median_val = df[feature].median()
        
        fig.add_vline(
            x=mean_val, line_dash="dash", line_color="red",
            annotation_text=f"Mean: {mean_val:.2f}",
            row=row, col=col
        )
        
        fig.add_vline(
            x=median_val, line_dash="dash", line_color="green",
            annotation_text=f"Median: {median_val:.2f}",
            row=row, col=col
        )
    
    # Update layout
    fig.update_layout(
        title_text="Feature Distributions",
        showlegend=False,
        height=300 * n_rows
    )
    
    fig.show()
    
    print("\n📊 FEATURE DISTRIBUTION INSIGHTS:")
    for feature in features:
        skewness = df[feature].skew()
        print(f"• {feature}: Skewness = {skewness:.2f} ({'Right-skewed' if skewness > 0.5 else 'Left-skewed' if skewness < -0.5 else 'Normal'})")

def analyze_correlations(df):
    """Analyze correlations between features and quality using Plotly."""
    print("\n" + "="*50)
    print("3. 🔗 CORRELATION ANALYSIS")
    print("="*50)
    
    # Correlation matrix
    correlation_matrix = df.corr()
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=np.round(correlation_matrix.values, 3),
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title='Feature Correlation Matrix',
        xaxis_title='Features',
        yaxis_title='Features',
        height=600
    )
    
    fig.show()
    
    # Quality correlations bar chart
    quality_correlations = correlation_matrix['quality'].sort_values(ascending=False)
    
    fig = go.Figure(data=go.Bar(
        x=quality_correlations.index,
        y=quality_correlations.values,
        marker_color=['green' if x > 0 else 'red' for x in quality_correlations.values],
        text=np.round(quality_correlations.values, 3),
        textposition='auto'
    ))
    
    fig.update_layout(
        title='Correlation with Quality',
        xaxis_title='Features',
        yaxis_title='Correlation Coefficient',
        height=500
    )
    
    fig.show()
    
    # Quality correlations
    print("\n🔗 CORRELATION WITH QUALITY:")
    for feature, corr in quality_correlations.items():
        if feature != 'quality':
            strength = 'Strong' if abs(corr) > 0.5 else 'Moderate' if abs(corr) > 0.3 else 'Weak'
            direction = 'Positive' if corr > 0 else 'Negative'
            print(f"• {feature}: {corr:.3f} ({strength} {direction})")
    
    return correlation_matrix

def analyze_quality_vs_features(df):
    """Analyze how each feature relates to wine quality using Plotly."""
    print("\n" + "="*50)
    print("4. 📊 QUALITY VS FEATURES ANALYSIS")
    print("="*50)
    
    features = df.drop('quality', axis=1).columns
    n_features = len(features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    # Create subplots
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[f'{feature.title()} vs Quality' for feature in features],
        specs=[[{"secondary_y": False} for _ in range(n_cols)] for _ in range(n_rows)]
    )
    
    for i, feature in enumerate(features):
        row = (i // n_cols) + 1
        col = (i % n_cols) + 1
        
        # Box plot
        for quality in sorted(df['quality'].unique()):
            subset = df[df['quality'] == quality][feature]
            fig.add_trace(
                go.Box(
                    y=subset,
                    name=f'Quality {quality}',
                    boxpoints='outliers',
                    jitter=0.3,
                    pointpos=-1.8
                ),
                row=row, col=col
            )
        
        # Add trend line
        quality_means = df.groupby('quality')[feature].mean()
        fig.add_trace(
            go.Scatter(
                x=quality_means.index,
                y=quality_means.values,
                mode='lines+markers',
                name=f'{feature} Trend',
                line=dict(color='red', dash='dash'),
                showlegend=False
            ),
            row=row, col=col
        )
    
    # Update layout
    fig.update_layout(
        title_text="Quality vs Features Analysis",
        showlegend=False,
        height=300 * n_rows
    )
    
    fig.show()
    
    # Statistical analysis
    print("\n📈 QUALITY VS FEATURES INSIGHTS:")
    for feature in features:
        # Calculate correlation
        corr = df[feature].corr(df['quality'])
        
        # Calculate mean by quality
        quality_means = df.groupby('quality')[feature].mean()
        
        print(f"\n• {feature.title()}:")
        print(f"  - Correlation with quality: {corr:.3f}")
        print(f"  - Mean values by quality: {dict(quality_means)}")

def analyze_feature_importance(correlation_matrix):
    """Analyze feature importance based on correlation with quality using Plotly."""
    print("\n" + "="*50)
    print("5. 🎯 FEATURE IMPORTANCE ANALYSIS")
    print("="*50)
    
    # Feature importance based on correlation
    feature_importance = abs(correlation_matrix['quality']).sort_values(ascending=False)
    feature_importance = feature_importance.drop('quality')
    
    # Create horizontal bar chart
    colors = ['green' if correlation_matrix['quality'][f] > 0 else 'red' for f in feature_importance.index]
    
    fig = go.Figure(data=go.Bar(
        y=feature_importance.index,
        x=feature_importance.values,
        orientation='h',
        marker_color=colors,
        text=[f"{correlation_matrix['quality'][f]:.3f}" for f in feature_importance.index],
        textposition='auto'
    ))
    
    fig.update_layout(
        title='Feature Importance (Absolute Correlation with Quality)',
        xaxis_title='Absolute Correlation Coefficient',
        yaxis_title='Features',
        height=500
    )
    
    fig.show()
    
    print("\n🎯 FEATURE IMPORTANCE INSIGHTS:")
    print("Top 5 most important features:")
    for i, (feature, importance) in enumerate(feature_importance.head().items(), 1):
        corr_val = correlation_matrix['quality'][feature]
        direction = "increases" if corr_val > 0 else "decreases"
        print(f"{i}. {feature}: {importance:.3f} (quality {direction} with higher {feature})")
    
    return feature_importance

def analyze_pairwise_relationships(df, feature_importance):
    """Analyze pairwise relationships between top features using Plotly."""
    print("\n" + "="*50)
    print("6. 📊 PAIRWISE RELATIONSHIPS")
    print("="*50)
    
    # Top correlations
    top_correlations = feature_importance.head(4).index.tolist()
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[f'{feature.title()} vs Quality' for feature in top_correlations],
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
                 [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    for i, feature in enumerate(top_correlations, 1):
        row = (i + 1) // 3 + 1
        col = (i - 1) % 2 + 1
        
        # Scatter plot with quality as color
        fig.add_trace(
            go.Scatter(
                x=df[feature],
                y=df['quality'],
                mode='markers',
                marker=dict(
                    color=df['quality'],
                    colorscale='viridis',
                    size=8,
                    opacity=0.6,
                    colorbar=dict(title="Quality Score")
                ),
                name=feature,
                text=f"Quality: {df['quality']}<br>{feature}: {df[feature]}",
                hovertemplate="<b>%{text}</b><extra></extra>"
            ),
            row=row, col=col
        )
        
        # Add trend line
        z = np.polyfit(df[feature], df['quality'], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(df[feature].min(), df[feature].max(), 100)
        y_trend = p(x_trend)
        
        fig.add_trace(
            go.Scatter(
                x=x_trend,
                y=y_trend,
                mode='lines',
                line=dict(color='red', dash='dash'),
                name=f'{feature} Trend',
                showlegend=False
            ),
            row=row, col=col
        )
    
    fig.update_layout(
        title_text="Top Features vs Quality Relationships",
        showlegend=False,
        height=800
    )
    
    fig.show()

def analyze_quality_categories(df):
    """Analyze wine characteristics by quality category using Plotly."""
    print("\n" + "="*50)
    print("7. 📈 QUALITY PREDICTION INSIGHTS")
    print("="*50)
    
    features = df.drop('quality', axis=1).columns
    
    # High vs Low quality analysis
    df['quality_category'] = pd.cut(df['quality'], bins=[0, 5, 6, 10], 
                                   labels=['Low (3-5)', 'Medium (6)', 'High (7-9)'])
    
    n_features = len(features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    # Create subplots
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[f'{feature.title()} by Quality Category' for feature in features],
        specs=[[{"secondary_y": False} for _ in range(n_cols)] for _ in range(n_rows)]
    )
    
    for i, feature in enumerate(features):
        row = (i // n_cols) + 1
        col = (i % n_cols) + 1
        
        # Violin plot
        for category in df['quality_category'].unique():
            subset = df[df['quality_category'] == category][feature]
            fig.add_trace(
                go.Violin(
                    x=[category] * len(subset),
                    y=subset,
                    name=category,
                    box_visible=True,
                    meanline_visible=True
                ),
                row=row, col=col
            )
    
    fig.update_layout(
        title_text="Feature Distributions by Quality Category",
        showlegend=False,
        height=300 * n_rows
    )
    
    fig.show()
    
    # Statistical comparison
    print("\n🍷 HIGH QUALITY WINE CHARACTERISTICS:")
    high_quality = df[df['quality'] >= 7]
    low_quality = df[df['quality'] <= 5]
    
    for feature in features:
        high_mean = high_quality[feature].mean()
        low_mean = low_quality[feature].mean()
        diff = high_mean - low_mean
        
        print(f"\n• {feature.title()}:")
        print(f"  - High quality wines: {high_mean:.3f}")
        print(f"  - Low quality wines: {low_mean:.3f}")
        print(f"  - Difference: {diff:.3f} ({'Higher' if diff > 0 else 'Lower'} in high quality)")

def analyze_outliers(df):
    """Analyze outliers in the dataset using Plotly."""
    print("\n" + "="*50)
    print("8. 📊 OUTLIER ANALYSIS")
    print("="*50)
    
    features = df.drop('quality', axis=1).columns
    n_features = len(features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    # Create subplots
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[f'{feature.title()} - Outlier Analysis' for feature in features],
        specs=[[{"secondary_y": False} for _ in range(n_cols)] for _ in range(n_rows)]
    )
    
    for i, feature in enumerate(features):
        row = (i // n_cols) + 1
        col = (i % n_cols) + 1
        
        # Box plot to show outliers
        fig.add_trace(
            go.Box(
                y=df[feature],
                name=feature,
                boxpoints='outliers',
                jitter=0.3,
                pointpos=-1.8
            ),
            row=row, col=col
        )
        
        # Add statistics
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[feature] < Q1 - 1.5*IQR) | (df[feature] > Q3 + 1.5*IQR)]
        
        fig.add_annotation(
            text=f'Outliers: {len(outliers)}',
            xref=f"x{i+1}", yref=f"y{i+1}",
            x=0.02, y=0.98,
            showarrow=False,
            bgcolor="yellow",
            bordercolor="black",
            borderwidth=1
        )
    
    fig.update_layout(
        title_text="Outlier Analysis",
        showlegend=False,
        height=300 * n_rows
    )
    
    fig.show()
    
    print("\n🔍 OUTLIER ANALYSIS INSIGHTS:")
    for feature in features:
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[feature] < Q1 - 1.5*IQR) | (df[feature] > Q3 + 1.5*IQR)]
        
        print(f"• {feature}: {len(outliers)} outliers ({len(outliers)/len(df)*100:.1f}% of data)")

def create_summary_dashboard(df, quality_counts, feature_importance, correlation_matrix):
    """Create a comprehensive summary dashboard using Plotly."""
    print("\n" + "="*50)
    print("9. 📈 SUMMARY DASHBOARD")
    print("="*50)
    
    features = df.drop('quality', axis=1).columns
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=('Quality Distribution', 'Top 5 Quality Correlations', 'Missing Values',
                       'Feature Ranges', 'Top 3 Features vs Quality', 'Dataset Summary'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # 1. Quality distribution
    fig.add_trace(
        go.Bar(
            x=quality_counts.index,
            y=quality_counts.values,
            name='Quality Counts',
            marker_color='crimson'
        ),
        row=1, col=1
    )
    
    # 2. Top correlations
    top_corr_features = feature_importance.head(5).index
    top_corr_values = [correlation_matrix['quality'][f] for f in top_corr_features]
    colors = ['green' if x > 0 else 'red' for x in top_corr_values]
    
    fig.add_trace(
        go.Bar(
            y=top_corr_features,
            x=top_corr_values,
            orientation='h',
            name='Correlations',
            marker_color=colors
        ),
        row=1, col=2
    )
    
    # 3. Missing values
    missing_data = df.isnull().sum()
    fig.add_trace(
        go.Bar(
            x=missing_data.index,
            y=missing_data.values,
            name='Missing Values',
            marker_color='orange'
        ),
        row=1, col=3
    )
    
    # 4. Feature ranges
    feature_ranges = df[features].max() - df[features].min()
    fig.add_trace(
        go.Bar(
            x=feature_ranges.index,
            y=feature_ranges.values,
            name='Feature Ranges',
            marker_color='purple'
        ),
        row=2, col=1
    )
    
    # 5. Quality by feature (top 3)
    top_3_features = feature_importance.head(3).index
    for i, feature in enumerate(top_3_features):
        quality_means = df.groupby('quality')[feature].mean()
        fig.add_trace(
            go.Scatter(
                x=quality_means.index,
                y=quality_means.values,
                mode='lines+markers',
                name=feature,
                line=dict(width=2)
            ),
            row=2, col=2
        )
    
    # 6. Data summary (text)
    summary_text = f"""
    DATASET SUMMARY
    ===============
    • Total Wines: {len(df):,}
    • Features: {len(features)}
    • Quality Range: {df['quality'].min()}-{df['quality'].max()}
    • High Quality (≥7): {len(df[df['quality'] >= 7])} ({len(df[df['quality'] >= 7])/len(df)*100:.1f}%)
    • Low Quality (≤5): {len(df[df['quality'] <= 5])} ({len(df[df['quality'] <= 5])/len(df)*100:.1f}%)
    • Missing Values: {df.isnull().sum().sum()}
    • Duplicates: {df.duplicated().sum()}

    TOP INSIGHTS
    ============
    • Most important feature: {feature_importance.index[0]}
    • Strongest correlation: {correlation_matrix['quality'].drop('quality').abs().idxmax()}
    • Quality distribution: {quality_counts.idxmax()} is most common
    """
    
    fig.add_annotation(
        text=summary_text,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=12, family="monospace"),
        align="left",
        bgcolor="white",
        bordercolor="black",
        borderwidth=1
    )
    
    fig.update_layout(
        title_text="Wine Quality Dataset - Comprehensive Dashboard",
        showlegend=True,
        height=800
    )
    
    fig.show()
    
    print("\n🎯 KEY INSIGHTS SUMMARY:")
    print(f"1. Dataset contains {len(df)} wines with {len(features)} chemical properties")
    print(f"2. Quality scores range from {df['quality'].min()} to {df['quality'].max()}")
    print(f"3. Most wines have quality score 6 ({quality_counts[6]} wines)")
    print(f"4. {feature_importance.index[0]} is the most important feature for quality prediction")
    print(f"5. {len(df[df['quality'] >= 7])} wines ({len(df[df['quality'] >= 7])/len(df)*100:.1f}%) are high quality")
    print(f"6. The dataset has {df.isnull().sum().sum()} missing values and {df.duplicated().sum()} duplicates")

def main():
    """Main function to run all analyses."""
    print("🍷 WINE QUALITY DATASET - COMPREHENSIVE ANALYSIS WITH PLOTLY")
    print("="*60)
    
    # Load and explore data
    df = load_and_explore_data()
    
    # Run all analyses
    quality_counts = analyze_quality_distribution(df)
    analyze_feature_distributions(df)
    correlation_matrix = analyze_correlations(df)
    analyze_quality_vs_features(df)
    feature_importance = analyze_feature_importance(correlation_matrix)
    analyze_pairwise_relationships(df, feature_importance)
    analyze_quality_categories(df)
    analyze_outliers(df)
    create_summary_dashboard(df, quality_counts, feature_importance, correlation_matrix)
    
    print("\n" + "="*60)
    print("✅ ANALYSIS COMPLETED SUCCESSFULLY!")
    print("="*60)

if __name__ == "__main__":
    main() 