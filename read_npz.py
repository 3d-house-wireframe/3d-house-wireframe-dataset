import open3d as o3d
import numpy as np

# Define the file path
file_path = 'sample.npz'

# Attempt to load the data
try:
    data = np.load(file_path)
except FileNotFoundError:
    print(f"File not found: {file_path}")
    exit()
except Exception as e:
    print(f"Error loading file: {e}")
    exit()

# Extract vertices and lines
vertices = data.get('vertices')
lines = data.get('lines')

# Check if the data exists
if vertices is None or lines is None:
    print("Vertices or lines data is missing in the file")
    exit()

# Create a LineSet object
line_set = o3d.geometry.LineSet()
line_set.points = o3d.utility.Vector3dVector(vertices)
line_set.lines = o3d.utility.Vector2iVector(lines)

# Visualize the LineSet
o3d.visualization.draw_geometries([line_set])
