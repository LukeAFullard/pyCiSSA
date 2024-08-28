import numpy as np
import matplotlib.pyplot as plt

def plot_grouped_components(t: np.ndarray,
                            x: np.ndarray,
                            x_trend: np.ndarray,
                            x_periodic: np.ndarray,
                            x_noise: np.ndarray):
    # Create a figure and a set of subplots
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Plot for the top subplot
    axs[0].plot(t, x, 'k', label='original time-series')
    axs[0].plot(t, x_trend, 'r', label='trend')
    axs[0].plot(t, x_periodic, 'g', label='periodic')
    axs[0].plot(t, x_trend + x_periodic, 'b', label='trend + periodic')
    
    # Set labels and title for the top subplot
    axs[0].set_ylabel('value')
    axs[0].legend(loc='upper left')
    axs[0].set_title('Time Series Components')
    
    # Plot for the bottom subplot
    axs[1].plot(t, x_noise, 'c', label='noise')
    axs[1].plot(t, x_periodic, 'g', label='periodic')
    
    # Set labels and title for the bottom subplot
    axs[1].set_xlabel('t')
    axs[1].set_ylabel('value')
    axs[1].legend(loc='upper left')
    axs[1].set_title('Noise and Periodic Components')
    
    # Adjust layout
    plt.tight_layout()
    
    # Show the plot
    plt.show()
    
    # Return the figure object
    return fig
