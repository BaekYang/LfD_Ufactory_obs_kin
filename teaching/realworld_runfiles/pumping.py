import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the original CSV file
file_path = './data/test.csv'
data = pd.read_csv(file_path)

# Define the target sampling rate
target_rate = 360  # 360 samples per second
interval_ms = 1000 / target_rate  # Time interval in milliseconds for each sample

# Create new timestamps for interpolation based on the target rate
start_time = data['timestamp'].iloc[0]
end_time = data['timestamp'].iloc[-1]
new_timestamps = np.arange(start_time, end_time, interval_ms)

# Interpolate all columns
interpolated_data = pd.DataFrame({'timestamp': new_timestamps})
for column in data.columns[1:]:  # Skip 'timestamp' for interpolation
    interpolated_data[column] = np.interp(new_timestamps, data['timestamp'], data[column])

# Save the interpolated data to a new CSV file
output_path = './data/pumping_interpolated_trajectory.csv'
interpolated_data.to_csv(output_path, index=False)

print(f"Interpolated CSV saved to: {output_path}")

# 3D visualization
fig = plt.figure(figsize=(10, 8))

# Extract positions for plotting
original_positions = data[['x', 'y', 'z']].values
interpolated_positions = interpolated_data[['x', 'y', 'z']].values

# 3D trajectory plot
ax = fig.add_subplot(111, projection='3d')
ax.plot(original_positions[:, 0], original_positions[:, 1], original_positions[:, 2], 'r-', label='Original Trajectory')
ax.plot(interpolated_positions[:, 0], interpolated_positions[:, 1], interpolated_positions[:, 2],
        '--', label='Interpolated Trajectory')
ax.set_xlabel('X Position (mm)')
ax.set_ylabel('Y Position (mm)')
ax.set_zlabel('Z Position (mm)')
ax.legend()
plt.title("Interpolated 3D Trajectory")
plt.show()
