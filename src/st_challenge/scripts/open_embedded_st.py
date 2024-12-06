import os
import tempfile
import datasets
from huggingface_hub import login
import shutil
from pathlib import Path
from hest import iter_hest
import numpy as np
from PIL import Image
from transformers import AutoModel
from torchvision import transforms
import torch
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATCH_SIZE_LV0 = 1024
MODEL_PATH = "/mnt/HDD8TO/models/"

def make_embeddings(patchs):

    titan = AutoModel.from_pretrained('MahmoodLab/TITAN', trust_remote_code=True)
    titan = titan.to(device)
    encoder_model = titan.return_conch()[0]

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    features = []
    for i in tqdm(range(patchs.shape[0])):
        res = transform(Image.fromarray(patchs[i])).unsqueeze(dim=0)

        with torch.inference_mode():
            features_per_patch = encoder_model(res)
            features.append(features_per_patch)

    return torch.cat(features, dim=0).numpy()

def convert_center_to_slicing(coords_center, dimensions, reverse_order=False):
    """
    Convert a center coordinate and dimensions to a slicing.
    If reverse_order is True, the slicing is returned in the reverse order.
    example
        input: np.array([150, 350, 550]), np.array([100, 100, 100])
        output: (slice(100,200), slice(300,400), slice(500,600))
    """
    dimensions = np.array(dimensions).astype(int)
    coords_center = np.array(coords_center).astype(int)
    half_dimensions = np.array(dimensions) // 2
    coords_min = coords_center - half_dimensions
    coords_max = coords_center - half_dimensions + dimensions
    slicing = tuple([slice(min_, max_) for min_, max_ in zip(coords_min, coords_max)])
    if reverse_order:
        return slicing[::-1]
    return slicing

class HEST:
    def __init__(self, hf_token=None, cache_dir=None):
        self.hf_token = hf_token
        self.cache_dir = Path(cache_dir)
        if self.hf_token == "from_env":
            self.hf_token = os.getenv("HF_TOKEN")
        if self.cache_dir is None:
            self.cache_dir = Path(tempfile.gettempdir(), "hest")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if hf_token is not None:
            login(self.hf_token)

    def empty_cache(self):
        shutil.rmtree(self.cache_dir)

    def cache_dataset(self, dataset_id, ):
        datasets.load_dataset(
            'MahmoodLab/hest',
            cache_dir=self.cache_dir,
            patterns=[f"*{dataset_id}[_.]**"],
        )

    def load_dataset(self, dataset_id, fullres=True):
        if not dataset_id in os.listdir(self.cache_dir):
            self.cache_dataset(dataset_id)
        for st in iter_hest(self.cache_dir, id_list=['INT1']): # Replaced by one that is present
            sdata = st.to_spatial_data(fullres=fullres)
        sdata = self.repatch(sdata)
        return sdata

    def repatch(self, sdata):
        image = np.array(sdata.images["ST_fullres_image"]["scale0"].image)
        DIMENSIONS = (128, 128)
        geometries = sdata.shapes["locations"]["geometry"]
        slicings = geometries.apply(lambda x: (slice(None),) + convert_center_to_slicing((int(x.x), int(x.y)), DIMENSIONS)).values
        patchs = []
        for slicing in slicings:
            patchs.append(image[slicing].swapaxes(0, 2))
        sdata.tables['table'].obsm['embedding'] = make_embeddings(np.array(patchs))
        return sdata

def open_hest_ids(
        id_to_query :list,
        hf_token=os.environ.get('HF_TOKEN'),
        cache_dir=os.environ.get('CACHE_DIR')
        ):

    sdata_list = []
    for id in id_to_query:
        hest = HEST(hf_token=hf_token, cache_dir=cache_dir)
        sdata_list.append(hest.load_dataset(id))
    hest.empty_cache()
    return sdata_list

if __name__ == "__main__":
    print(open_hest_ids(["INT1", "INT2"]))
