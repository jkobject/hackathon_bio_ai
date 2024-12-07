# hackaton_bio_ai

- run scPRINT and scDataloader
- Downloader: HEST to spatial data files in lamindb (visium / )
- DataLoader for spatialdata
- extract embeddings from TITAN and store in .obsm() and ask questions
- add textbox to napari to ask questions
-



### 1. Install the package in editable mode
```bash
git clone git@github.com:jkobject/hackathon_bio_ai.git
cd hackathon_bio_ai
conda create -n st_challenge_dev python=3.10
conda activate st_challenge_dev
pip install -e .
```

### Install the package
```bash
conda create -n st_challenge_prod python=3.10
conda activate st_challenge_prod
pip install git+https://github.com/jkobject/hackathon_bio_ai.git
```

```bash
conda deactivate
conda remove --name st_challenge_prod --all
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
