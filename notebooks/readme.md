## Research Notebooks

The [`notebooks/`](notebooks/) directory contains Jupyter notebooks demonstrating various applications and experiments:

- **QST_SNN/**: Quantum State Tomography with Spiking Neural Networks demonstrations
  - See [`notebooks/readme.md`](notebooks/readme.md) for more details

### Running the Notebooks

To run the research notebooks, you'll need to install the Python dependencies:

1. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
2. Launch Jupyter:
    ```bash
   jupyter notebook
   ```
3. Navigate to the notebooks directory and open any notebook to get started.

Note: Some notebooks may require additional Julia packages. Ensure you have activated the project environment in Project.toml before running Julia-based notebooks:
   ```bash
    using Pkg
    Pkg.activate(".")
    Pkg.instantiate()
   ```