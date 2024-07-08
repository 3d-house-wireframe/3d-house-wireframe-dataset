# 3D HOUSE WIREFRAME DATASET

## Description

This project is dedicated to working with 3d house wireframe datasets stored in NPZ files. It includes a Python script for reading and visualizing 3D wireframe data using the Open3D library. The dataset and related scripts are organized to facilitate easy loading, processing, and visualization of 3D wireframes.

## Directory Structure

```
3D_HOUSE_WIREFRAME_DATASET/
│
├── npz/
│   ├── 00000.npz
│   ├── 00001.npz
│   ├── ...
│   └── [more NPZ files]
│
├── read_npz.py
│
└── README.md
```

- **npz/**: This directory contains the NPZ files with the 3D house wireframe data.
- **read_npz.py**: A Python script for reading and visualizing the wireframe data from NPZ files.
- **README.md**: This file, providing an overview of the project.

## Usage

### Prerequisites

Make sure you have Python installed along with the necessary libraries:
- `open3d`
- `numpy`

You can install these libraries using pip:
```bash
pip install open3d numpy
```

### Running the Script

To visualize a 3d house wireframe from an NPZ file, follow these steps:

1. Update the `file_path` variable in `read_npz.py` to point to the desired NPZ file.
2. Run the script:
   ```bash
   python read_npz.py
   ```

### Script Explanation

The `read_npz.py` script performs the following tasks:
1. Loads the vertices and lines data from the specified NPZ file.
2. Checks if the data is correctly loaded.
3. Creates a `LineSet` object using Open3D.
4. Visualizes the wireframe using Open3D's visualization tools.


## Acknowledgments

- Thanks to the Open3D and NumPy communities for providing the essential tools for this project.

```
