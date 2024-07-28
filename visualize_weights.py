import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def extract_and_plot_weights(state_dict):
    # Extract all weights
    weights = []
    for name, param in state_dict.items():
        if param.dim() > 1:  # Exclude biases which are 1D
            weights.extend(param.detach().cpu().numpy().flatten())
    
    weights = np.array(weights)
    weights = weights[(weights >= -1) & (weights <= 1)]


    
    # Smart bin selection using Freedman-Diaconis rule
    iqr = np.percentile(weights, 75) - np.percentile(weights, 25)
    bin_width = 2 * iqr / (len(weights) ** (1/3))
    num_bins = int((weights.max() - weights.min()) / bin_width)
    
    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(weights, bins=num_bins, edgecolor='black')
    plt.title('Histogram of Model Weights')
    plt.xlabel('Weight Value')
    plt.ylabel('Frequency')
    
    # Add some statistics
    plt.text(0.95, 0.95, f'Mean: {weights.mean():.4f}\nStd: {weights.std():.4f}',
             verticalalignment='top', horizontalalignment='right',
             transform=plt.gca().transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    plt.xlim(-2, 2)
    
    plt.savefig("our_weights_hist.png")

    #plt.savefig('vm_weights_hist.png')

state_dict = torch.load('./bad_weights_7_24.pth')
#state_dict = torch.load('/home/fvidal/weights/VM/videomamba_m16_5M_f8_res224.pth')
state_dict = {key: value.half() for key, value in state_dict.items()}
#print(state_dict.keys())
extract_and_plot_weights(state_dict)
# Example usage:
# Assuming you have a PyTorch model named 'my_model'
# extract_and_plot_weights(my_model)
