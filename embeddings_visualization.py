import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from sklearn.manifold import TSNE
import seaborn as sns
# Make sure to adjust the path as needed
sys.path.append('D:/CodeProject/LLM-Detect-AI-Generated-Text/sentence-transformers')
from sentence_transformers import SentenceTransformer

# Load the model
model = SentenceTransformer('D:/CodeProject/LLM-Detect-AI-Generated-Text/all-MiniLM-L6-v2')

# Load the data
df = pd.read_csv(r'D:\CodeProject\LLM-Detect-AI-Generated-Text\code\train1.csv')
df2 = df[df['label'] == 0].reset_index(drop=True)
df = df[df['label'] == 1].reset_index(drop=True)

# Encode the texts
vector1 = model.encode(list(df["text"]), batch_size=512, show_progress_bar=True, device="cuda", convert_to_tensor=False)
vector2 = model.encode(list(df2["text"]), batch_size=512, show_progress_bar=True, device="cuda", convert_to_tensor=False)

# Concatenate the data
show_df = pd.concat([df, df2], ignore_index=True)
show_emb = np.concatenate([vector1, vector2])

# Perform t-SNE
tsne = TSNE(n_components=2, random_state=42)  # Set random_state for reproducibility
X_embedded = tsne.fit_transform(show_emb)

# Create labels for the sources
show_df['src'] = ['LLM'] * df.shape[0] + ['student'] * df2.shape[0]

# Create the scatter plot
plt.figure(figsize=(12, 8))
sns.scatterplot(x=X_embedded[:, 0], y=X_embedded[:, 1], hue=show_df['src'], legend='full', palette='viridis')
plt.title('t-SNE Visualization of Text Embeddings')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.legend(title='Source')
plt.grid(True)
plt.show()