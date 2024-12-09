import pandas as pd
import matplotlib.pyplot as plt

# Load data from CSV file
data = pd.read_csv('/lustre/orion/stf218/proj-shared/brave/transfusion-pytorch/transfusion_pytorch/results/acc2.csv', header=None, names=['Value'])

# Calculate moving average
window_size = 5
data['Moving Average'] = data['Value'].rolling(window=window_size).mean()

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(data.index, data['Value'], marker='o', linestyle='-', label='Actual Values')
plt.plot(data.index, data['Moving Average'], linestyle='--', label=f'{window_size}-Point Moving Average')
plt.title('Line Plot from CSV Data with Moving Average')
plt.xlabel('Index')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.savefig('plot_with_moving_average2.png', format='png', dpi=300)  # Save as PNG with high resolution
