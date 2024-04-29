
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
import pandas as pd



class VectorQuantizerKMeans(nn.Module):
    """
    Args:
        embedding_dim (int): the dimensionality of the tensors in the
          quantized space. Inputs to the modules must be in this format as well.
        num_embeddings (int): the number of vectors in the quantized space.
        commitment_cost (float): scalar which controls the weighting of the loss terms (see
          equation 4 in the paper - this variable is Beta).
    """

    def __init__(self, features, k=100):
        super().__init__()

        # initialize embeddings
        
        
        self.k = k
        self.centroids = torch.empty(0)
        self.labels = torch.empty(0)
        #print(self.centroids.min(), self.centroids.max(), '---')
        


    def forward(self, features, max_iters=10, assign=False, save=False):
        """
        K-Means clustering function.

        """
        #features_ = features.detach()
        # Randomly initialize cluster centers
        #print(max_iters.shape, max_iters, )
        #print(max_iters)

        if not assign:
            #print(assign)
            if len(self.centroids) == 0:
                #print(features.shape, '---',features[torch.randperm(features.size(0))[:self.k]].shape)
                self.centroids = features[torch.randperm(features.size(0))[:self.k]]
                
            for _ in range(max_iters):
                #print(max_iters, tt, self.k)
                distances = torch.cdist(features, self.centroids)
                self.labels = torch.argmin(distances, dim=1)
                
                #new_centroids = torch.stack([features[self.labels == i].mean(dim=0) for i in range(self.k)])
                #if torch.all(new_centroids == self.centroids):
                #    break
                new_centroids = []

                for i in range(self.k):
                    # Get the features belonging to the current class i
                    features_i = features[self.labels == i]
                    
                    if features_i.size(0) == 0:  # If there are no samples in the current class
                        # Here's an example: if the features are empty, set the centroid to a zero vector
                        centroid_i = torch.zeros(features.size(1)).to("cuda")  # Keep the same dimensionality as the original features
                    else:
                        # Calculate the new centroid for the current class
                        centroid_i = features_i.mean(dim=0)
                    
                    # Append the new centroid to the list
                    new_centroids.append(centroid_i)

                # Stack the new centroids into a tensor
                self.centroids = torch.stack(new_centroids)

            final_features = self.centroids[self.labels.long()]
            final_features_ = features - features.detach() + final_features
            differences = F.mse_loss(features, final_features_)
            
        else:
            final_features = self.centroids[self.labels.long()]
            final_features_ =  final_features
            differences = 0.0
        
        if save:
            df_bc = pd.DataFrame(final_features_.cpu().detach().numpy())
            df_b1 = pd.DataFrame(self.labels.cpu().detach().numpy())
            df_combined = pd.concat([df_bc, df_b1], axis=1)
            xls_filename_combined = "combined_tensor.xlsx"
            df_combined.to_excel(xls_filename_combined, index=False, header=False)
        
        
        return final_features_, self.labels, differences
