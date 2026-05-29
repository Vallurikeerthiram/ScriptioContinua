import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
df = pd.read_excel("simple_wikipedia_sentence_level_scriptio_continua.xlsx")

def plot_numeric_bins(column, step, title, xlabel):
    max_val = int(df[column].max())
    
    # Create bins
    bins = np.arange(0, max_val + step, step)
    
    # Histogram
    counts, edges = np.histogram(df[column], bins=bins)
    centers = edges[:-1] + step / 2

    plt.figure(figsize=(10, 5))
    plt.bar(centers, counts, width=step * 0.9)

    # X-axis: ONLY start (0) and end (max)
    plt.xticks([0, max_val])

    # Y-axis: AUTO SCALE (do NOT touch)
    plt.xlabel(xlabel)
    plt.ylabel("Number of sentences")
    plt.title(title)

    plt.tight_layout()
    plt.show()


# 1️⃣ Stopwords → 0,1,2,3,...
plot_numeric_bins(
    column="stopwords",
    step=1,
    title="Stopword Count Distribution",
    xlabel="Stopword count"
)

# 2️⃣ Space count → 0,2,4,6,...
plot_numeric_bins(
    column="space_count",
    step=2,
    title="Space Count Distribution",
    xlabel="Space count"
)

# 3️⃣ Word count → 0,3,6,9,...
plot_numeric_bins(
    column="wordcount",
    step=3,
    title="Word Count Distribution",
    xlabel="Word count"
)

# 4️⃣ Character count → 0,10,20,30,...
plot_numeric_bins(
    column="charactercount",
    step=10,
    title="Character Count Distribution",
    xlabel="Character count (excluding spaces)"
)

# 5️⃣ Numbers → 0,1,2,3,...
plot_numeric_bins(
    column="numbers",
    step=1,
    title="Digit Count Distribution",
    xlabel="Digit count"
)

# 6️⃣ Letters → 0,40,80,120,...
plot_numeric_bins(
    column="letters",
    step=40,
    title="Letter Count Distribution",
    xlabel="Letter count"
)

# 7️⃣ Special characters → 0,1,2,3,...
plot_numeric_bins(
    column="special_characters",
    step=1,
    title="Special Character Count Distribution",
    xlabel="Special character count"
)
