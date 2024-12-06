# hackaton_bio_ai

- run scPRINT and scDataloader
- Downloader: HEST to spatial data files in lamindb (visium / )
- DataLoader for spatialdata
- extract embeddings from TITAN and store in .obsm() and ask questions
- add textbox to napari to ask questions
-





### 1. Clone the repository
```bash
#TODO
```

### 2. Install the dependencies
```bash
conda env create -f environment.yaml
conda activate theremia
```

### 3. Install the package
```bash
pip install -e .
```

### 4. View an image
```bash
conda activate theremia
python -m st_challenge.scripts.viz_slice INT1
```

# st_challenge



# HEST

https://github.com/mahmoodlab/HEST

```
git clone https://github.com/mahmoodlab/HEST.git
cd HEST
conda create -n "hest" python=3.9
conda activate hest
pip install -e .
```

# CONCH
