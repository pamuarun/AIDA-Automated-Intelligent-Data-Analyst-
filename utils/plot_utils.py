# utils/plot_utils.py
import io
import matplotlib.pyplot as plt
import seaborn as sns

# Set global styling for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def fig_to_png_bytes(fig):
    """Convert Matplotlib figure to PNG byte stream."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=300)
    buf.seek(0)
    data = buf.read()
    plt.close(fig)
    return data


def distribution_plot(df, col):
    """Create distribution + KDE plot and return BOTH fig and PNG bytes."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df[col], kde=True, ax=ax, alpha=0.7)
    ax.set_title(f"Distribution of {col}", fontsize=14, pad=15)
    ax.set_xlabel(col, fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3)
    
    png_bytes = fig_to_png_bytes(fig)
    return fig, png_bytes


def box_plot(df, col):
    """Create box plot and return BOTH fig and PNG bytes."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(x=df[col], ax=ax, palette="viridis")
    ax.set_title(f"Boxplot of {col}", fontsize=14, pad=15)
    ax.set_xlabel(col, fontsize=12)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3)
    
    png_bytes = fig_to_png_bytes(fig)
    return fig, png_bytes


def correlation_heatmap(df):
    """Create correlation heatmap and return BOTH fig and PNG bytes."""
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        return None, None

    fig, ax = plt.subplots(figsize=(10, 8))
    corr_matrix = numeric_df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax, 
                center=0, square=True, fmt='.2f', cbar_kws={"shrink": .8})
    ax.set_title("Correlation Heatmap", fontsize=16, pad=20)
    
    # Rotate labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    png_bytes = fig_to_png_bytes(fig)
    return fig, png_bytes