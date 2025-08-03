import os
from src.utils.Drives import drives
import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset, DataLoader
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting

# -----------------------------
# 1. Load the Data
# -----------------------------
with open('data/460631.pkl', 'rb') as file:
    data = pickle.load(file)

print("Pattern groups:", data.splitted_ts_groups.keys())
print("Example trip DataFrame:")
print(data.splitted_ts_groups[1][0])
print("Columns:", data.splitted_ts_groups[1][0].columns)

# -----------------------------
# 2. Preprocess Data with Padding
# -----------------------------
def preprocess_data(data):
    """
    Converts `data.splitted_ts_groups` dictionary into padded tensors.
    Returns:
      X_padded: (num_trips, max_seq_length, input_dim)
      y_padded: (num_trips, max_seq_length, 2)
      group_ids_tensor: (num_trips)
      road_speeds_padded: (num_trips, max_seq_length)
      hours_padded: (num_trips, max_seq_length)
      lengths_tensor: (num_trips) - original (unpadded) sequence lengths
    """
    X_list, y_list = [], []
    group_ids_list, road_speeds_list, hours_list, lengths = [], [], [], []

    for group_id, time_series_list in data.splitted_ts_groups.items():
        for df in time_series_list:
            df = df.sort_values("orig_time")  # Ensure correct time order

            # Extract features and targets
            features = df[['longitude', 'latitude', 'speed', 'acceleration_est_1', 'angular_acc']].values
            road_speed = df['road_speed'].values
            hour = pd.to_datetime(df['orig_time']).dt.hour.values
            target = df[['speed', 'acceleration_est_1']].values  # Next step prediction

            # Skip trips that are too short
            if len(features) < 2:
                continue

            # --- Clip road_speed values to valid range [0, 99] ---
            road_speed = np.clip(road_speed, 0, 99)

            # Create tensors: input is features[:-1] and target is target[1:]
            X_tensor = torch.tensor(features[:-1], dtype=torch.float32)
            y_tensor = torch.tensor(target[1:], dtype=torch.float32)
            road_speed_tensor = torch.tensor(road_speed[:-1], dtype=torch.long)
            hour_tensor = torch.tensor(hour[:-1], dtype=torch.long)

            X_list.append(X_tensor)
            y_list.append(y_tensor)
            # Convert group IDs to zero-indexed (subtract 1)
            group_ids_list.append(torch.tensor(group_id - 1, dtype=torch.long))
            road_speeds_list.append(road_speed_tensor)
            hours_list.append(hour_tensor)
            lengths.append(X_tensor.shape[0])

    # Pad the sequences to the same length
    X_padded = pad_sequence(X_list, batch_first=True, padding_value=0)
    y_padded = pad_sequence(y_list, batch_first=True, padding_value=0)
    road_speeds_padded = pad_sequence(road_speeds_list, batch_first=True, padding_value=0)
    hours_padded = pad_sequence(hours_list, batch_first=True, padding_value=0)
    group_ids_tensor = torch.stack(group_ids_list)
    lengths_tensor = torch.tensor(lengths, dtype=torch.long)

    return X_padded, y_padded, group_ids_tensor, road_speeds_padded, hours_padded, lengths_tensor

# Preprocess the data
X_train, y_train, group_ids, road_speeds, hours, lengths = preprocess_data(data)

# -----------------------------
# 3. Define the Transformer Model with Masked Time-Step Prediction and Coordinate MF Interaction
# -----------------------------
class DriverTransformer(nn.Module):
    def __init__(self, num_groups, input_dim, latent_dim, n_heads=4, n_layers=3, max_seq_len=100, dropout_rate=0.1):
        super(DriverTransformer, self).__init__()
        # Group embedding for latent driver style
        self.group_embedding = nn.Embedding(num_groups, latent_dim)
        # Bias embeddings (road speed: indices 0-99; hour: indices 0-23)
        self.road_speed_bias = nn.Embedding(100, 1)
        self.time_bias = nn.Embedding(24, 1)
        # Learned positional encoding for up to max_seq_len steps
        self.positional_encoding = nn.Parameter(torch.randn(1, max_seq_len, latent_dim))
        # Transformer Encoder with dropout
        encoder_layer = TransformerEncoderLayer(
            d_model=latent_dim, nhead=n_heads, dropout=dropout_rate, batch_first=True
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=n_layers)
        # Dropout before regression head
        self.dropout = nn.Dropout(dropout_rate)
        # Regression head to predict next-step speed & acceleration
        self.regression_head = nn.Linear(latent_dim, 2)
        # New: Linear layer to project the coordinate (lon, lat) into latent space (MF style)
        self.coord_proj = nn.Linear(2, latent_dim)

    def forward(self, x, group_ids, road_speeds, hours, lengths, pred_indices=None):
        """
        x: (batch_size, seq_len, input_dim) -- used for obtaining coordinate features
        group_ids: (batch_size)
        road_speeds: (batch_size, seq_len)
        hours: (batch_size, seq_len)
        lengths: (batch_size) - original (unpadded) lengths of each sequence
        pred_indices: (batch_size) - indices at which to predict (if None, defaults to lengths-1)
        """
        batch_size, seq_len, _ = x.size()

        # Get and repeat group embedding
        group_embed_rep = self.group_embedding(group_ids)  # (batch_size, latent_dim)
        group_embed = group_embed_rep.unsqueeze(1).repeat(1, seq_len, 1)  # (batch_size, seq_len, latent_dim)

        # Compute bias terms for each time step
        road_bias = self.road_speed_bias(road_speeds).squeeze(-1)  # (batch_size, seq_len)
        time_bias = self.time_bias(hours).squeeze(-1)  # (batch_size, seq_len)

        # Prepare transformer input: add group embedding + positional encoding
        transformer_input = group_embed + self.positional_encoding[:, :seq_len, :]
        transformer_output = self.transformer(transformer_input)  # (batch_size, seq_len, latent_dim)

        # If no prediction indices are provided, default to last valid index per sample
        if pred_indices is None:
            pred_indices = (lengths - 1)
        # Expand pred_indices for gathering hidden states
        pred_indices_expanded = pred_indices.unsqueeze(1).unsqueeze(2).expand(batch_size, 1, transformer_output.size(2))
        final_hidden = transformer_output.gather(1, pred_indices_expanded).squeeze(1)
        final_hidden = self.dropout(final_hidden)

        # Gather corresponding bias values using pred_indices
        road_bias_final = road_bias.gather(1, pred_indices.unsqueeze(1)).squeeze(1)
        time_bias_final = time_bias.gather(1, pred_indices.unsqueeze(1)).squeeze(1)

        # --- New: Incorporate coordinate MF interaction ---
        # Extract coordinate features (longitude and latitude) from x at the masked time-step.
        # x is of shape (batch_size, seq_len, input_dim); we take the first 2 features.
        x_coord = x.gather(1, pred_indices.unsqueeze(1).unsqueeze(2).expand(batch_size, 1, 2)).squeeze(1)  # (batch_size, 2)
        coord_latent = self.coord_proj(x_coord)  # (batch_size, latent_dim)
        # For MF, compute dot product between the (original) group embedding and the coordinate latent vector.
        mf_term = (group_embed_rep * coord_latent).sum(dim=1, keepdim=True)  # (batch_size, 1)

        # Final prediction
        prediction = (self.regression_head(final_hidden) +
                      road_bias_final.unsqueeze(-1) +
                      time_bias_final.unsqueeze(-1) +
                      mf_term)
        return prediction, final_hidden

# -----------------------------
# 4. Training Setup (Using GPU with CUDA)
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

num_groups = 7       # Groups indexed 0-6
input_dim = 5        # Features: longitude, latitude, speed, acceleration, angular_acc
latent_dim = 64      # Dimension of latent embeddings

# Use padded sequence length as max_seq_len
max_seq_len = X_train.size(1)

model = DriverTransformer(num_groups, input_dim, latent_dim, max_seq_len=max_seq_len, dropout_rate=0.1).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
criterion = nn.MSELoss()
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

# Create TensorDataset and DataLoader for batching
batch_size = 16  # Adjust based on GPU memory
dataset = TensorDataset(X_train, y_train, group_ids, road_speeds, hours, lengths)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# -----------------------------
# 5. Checkpoint Setup: Load model if exists
# -----------------------------
checkpoint_file = "src/modeling/models/driver_transformer_coor_checkpoint.pth"
start_epoch = 0
loss_history = []

if os.path.exists(checkpoint_file):
    print("Loading checkpoint from", checkpoint_file)
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    loss_history = checkpoint['loss_history']
    print(f"Resuming training from epoch {start_epoch}")

# -----------------------------
# 6. Training Loop with Random Masked (Time-Step) Prediction
# -----------------------------
num_epochs = 10  # Total epochs to train
for epoch in range(start_epoch, num_epochs):
    model.train()
    epoch_loss = 0.0
    for batch in dataloader:
        X_batch, y_batch, group_ids_batch, road_speeds_batch, hours_batch, lengths_batch = [item.to(device) for item in batch]
        optimizer.zero_grad()

        # For each sample in the batch, sample a random valid prediction index (simulate masking)
        pred_indices_list = []
        for l in lengths_batch:
            idx = torch.randint(low=0, high=l.item(), size=(1,))
            pred_indices_list.append(idx)
        pred_indices = torch.stack(pred_indices_list).to(device).squeeze()  # (batch_size,)

        # Forward pass with randomly chosen prediction indices
        y_pred, _ = model(X_batch, group_ids_batch, road_speeds_batch, hours_batch, lengths_batch, pred_indices)

        # Gather the corresponding target values using the same indices
        pred_indices_expanded = pred_indices.unsqueeze(1).unsqueeze(2).expand(y_batch.size(0), 1, y_batch.size(2))
        target = y_batch.gather(1, pred_indices_expanded).squeeze(1)

        loss = criterion(y_pred, target)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * X_batch.size(0)
    epoch_loss /= len(dataset)
    scheduler.step()
    loss_history.append(epoch_loss)
    print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {epoch_loss:.4f}")

    # Save checkpoint at the end of each epoch
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss_history': loss_history,
    }, checkpoint_file)

# Plot training loss
plt.figure(figsize=(8, 4))
plt.plot(loss_history, marker="o", label="Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.legend()
plt.show()

# -----------------------------
# 7. Extract and Visualize Group Embeddings & PCA (2D)
# -----------------------------
group_embeddings = model.group_embedding.weight.data.cpu().numpy()  # Shape: (num_groups, latent_dim)
pca = PCA(n_components=2)
group_embeddings_2d = pca.fit_transform(group_embeddings)
eigen_values = pca.explained_variance_

plt.figure(figsize=(8, 6))
for i in range(num_groups):
    plt.scatter(group_embeddings_2d[i, 0], group_embeddings_2d[i, 1], label=f"Group {i + 1}", s=100)
    plt.text(group_embeddings_2d[i, 0], group_embeddings_2d[i, 1], f" {i + 1}", fontsize=12)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA of Learned Group Embeddings (2D)")
plt.legend()
plt.show()

print("First two eigenvalues from PCA on group embeddings (2D):")
print(eigen_values)

# -----------------------------
# 8. 3D Visualization of Group Embeddings using the First Three PCA Components
# -----------------------------
pca_3d = PCA(n_components=3)
group_embeddings_3d = pca_3d.fit_transform(group_embeddings)
eigen_values_3d = pca_3d.explained_variance_

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
for i in range(num_groups):
    ax.scatter(group_embeddings_3d[i, 0], group_embeddings_3d[i, 1], group_embeddings_3d[i, 2],
               label=f"Group {i + 1}", s=100)
    ax.text(group_embeddings_3d[i, 0], group_embeddings_3d[i, 1], group_embeddings_3d[i, 2],
            f" {i + 1}", fontsize=12)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
ax.set_title("PCA of Learned Group Embeddings (3D)")
plt.legend()
plt.show()

print("First three eigenvalues from PCA on group embeddings (3D):")
print(eigen_values_3d)

