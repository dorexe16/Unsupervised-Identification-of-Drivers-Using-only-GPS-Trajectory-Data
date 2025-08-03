# import os
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA
# from mpl_toolkits.mplot3d import Axes3D  # ensures 3D projection is available
#
# # Define the directory containing the models
# models_dir = "src/modeling/models/"
#
# # Loop over all model files in the directory
# for model_filename in os.listdir(models_dir):
#     # Adjust the file extension check as needed (e.g., .pth, .pt, etc.)
#     if model_filename.endswith(".pth") or model_filename.endswith(".pt"):
#         model_path = os.path.join(models_dir, model_filename)
#         print(f"Processing model: {model_filename}")
#
#         # Load the model (map_location='cpu' ensures compatibility if CUDA isn't available)
#         model = torch.load(model_path, map_location='cpu')
#
#         # Ensure the model has the required attribute
#         if hasattr(model, "group_embedding"):
#             # Extract the first 7 group embeddings
#             group_embeddings_np = model.group_embedding.weight.data.cpu().numpy()[:7, :]
#
#             # Apply PCA to reduce embeddings to 3 dimensions
#             pca_3d = PCA(n_components=3)
#             group_embeddings_3d = pca_3d.fit_transform(group_embeddings_np)
#             eigen_values_3d = pca_3d.explained_variance_
#
#             # Create a 3D scatter plot
#             fig = plt.figure(figsize=(10, 8))
#             ax = fig.add_subplot(111, projection='3d')
#             for i in range(7):
#                 ax.scatter(group_embeddings_3d[i, 0], group_embeddings_3d[i, 1], group_embeddings_3d[i, 2],
#                            label=f"Group {i + 1}", s=100)
#                 ax.text(group_embeddings_3d[i, 0], group_embeddings_3d[i, 1], group_embeddings_3d[i, 2],
#                         f" {i + 1}", fontsize=12)
#
#             ax.set_xlabel("PC1")
#             ax.set_ylabel("PC2")
#             ax.set_zlabel("PC3")
#             ax.set_title(f"PCA of Learned Group Embeddings (3D) - {model_filename}")
#             plt.legend()
#             plt.show()
#         else:
#             print(f"Model {model_filename} does not have a 'group_embedding' attribute.")
import torch

# טוענים את ה-checkpoint
checkpoint = torch.load(r"C:\Users\dorex\Desktop\Unsupervised Identification of Drivers Using only GPS Trajectory Data\Unsupervised-Identification-of-Drivers-Using-only-GPS-Trajectory-Data\src\modeling\models\driver_transformer_checkpoint.pth")

print(checkpoint.keys())
