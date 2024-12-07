import lamindb as ln
from scdataloader import DataModule, Preprocessor, utils
from scdataloader.preprocess import additional_postprocess, additional_preprocess

import os
import urllib.request
import torch

import dask

dask.config.set({"dataframe.query-planning": False})

from scprint import scPrint
from scprint.tasks import Denoiser, Embedder

import datasets
from hest.HESTData import load_hest
from spatialdata import read_zarr

from huggingface_hub import login

import scanpy as sc
import squidpy as sq
import matplotlib.pyplot as plt
import seaborn as sns
import anndata as ad
import pandas as pd
from scipy.sparse import csr_matrix
import numpy as np


class STDenoiser:
    def __init__(self, scprint_size, genelist, batch_size=12, num_workers=8):
        ckpt_path = scprint_size + ".ckpt"
        if not os.path.exists(ckpt_path):
            url = (
                "https://huggingface.co/jkobject/scPRINT/resolve/main/"
                + scprint_size
                + ".ckpt"
            )
            urllib.request.urlretrieve(url, ckpt_path)
        self.model = scPrint.load_from_checkpoint(
            ckpt_path,
            precpt_gene_emb=None,
            # triton gets installed so it must think it has cuda enabled
            transformer="fast",  # else normal, without flashattention
        )
        self.denoiser = Denoiser(
            max_cells=200_000,  # number of cells which will be processed
            batch_size=batch_size,
            num_workers=num_workers,
            how="some",
            genelist=genelist,
            max_len=len(genelist),  # we will work on 2000 genes (input and output)
            downsample=False,  # we are removing 70% of the counts,
            # should be modified to make the data look more like st
            predict_depth_mult=20,  # how much to increase expression
            dtype=torch.float32,
        )
        self.embed = Embedder(
            how="most var",
            max_len=500,
            add_zero_genes=0,
            num_workers=16,
            pred_embedding=["cell_type_ontology_term_id"],
            keep_all_cls_pred=False,
            output_expression="none",
            batch_size=64,
        )

    def __call__(self, stdata):
        if "organism_ontology_term_id" not in stdata.obs.columns:
            raise ValueError("organism_ontology_term_id not in stdata.obs")
        stdata = stdata[
            :,
            [
                "BLANK" not in i and "NegControl" not in i
                for i in stdata.var.index.tolist()
            ],
        ]

        # set to 300, 1 for visium
        sc.pp.filter_cells(
            stdata,
            min_counts=50,
        )
        sc.pp.filter_genes(stdata, min_cells=5)

        stdata.layers["counts"] = stdata.X.copy()
        sc.pp.normalize_total(stdata, inplace=True)
        sc.pp.log1p(stdata)
        sc.pp.pca(stdata)
        sc.pp.neighbors(stdata)
        sc.tl.umap(stdata)
        sc.tl.leiden(stdata)

        n_adata, metrics = self.embed(self.model, stdata.copy(), cache=False)
        metrics, random_indices, genes, expr_pred = self.denoiser(
            model=self.model,
            adata=stdata,
        )
        n_adata.X[
            :, n_adata.var.index.isin(np.array(self.model.genes)[genes[0]])
        ] = expr_pred
        return n_adata
