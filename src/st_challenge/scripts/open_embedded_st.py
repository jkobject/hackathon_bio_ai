import os
import tempfile
import datasets
from huggingface_hub import login
import shutil
from pathlib import Path
from hest import iter_hest
import numpy as np


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
            from huggingface_hub import login
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
        sdata.tables['table'].obsm['embeddings'] = np.array(patchs)
        return sdata


def open_hest_ids(
        id_to_query :list,
        hf_token=os.environ.get('HF_TOKEN'),
        cache_dir=os.environ.get('CACHE_DIR')
        ):

    sdata_list = []
    for id in ["INT1", "INT2"]:
        hest = HEST(hf_token=hf_token, cache_dir=cache_dir)
        sdata_list.append(hest.load_dataset(id))
        hest.empty_cache()
    return sdata_list


if __name__ == "__main__":
    print(open_hest_ids(["INT1", "INT2"]))
