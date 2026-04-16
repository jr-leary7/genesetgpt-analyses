import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Libraries 

    Here we load all the libraries we'll need for our analysis.
    """
    )
    return


@app.cell
def _():
    import os
    import time
    import json
    import openai
    import shutil
    import pickle
    import getpass
    import warnings
    import numpy as np
    import scanpy as sc
    import pandas as pd
    import marimo as mo
    import session_info
    import scvelo as scv
    import anndata as ad
    import cellrank as cr
    import genesetgpt as gpt 
    from openai import OpenAI
    from datetime import datetime
    from functools import partial
    from pydantic import BaseModel
    from dotenv import load_dotenv
    import matplotlib.pyplot as plt
    from pandarallel import pandarallel
    from concurrent.futures import ThreadPoolExecutor
    return cr, load_dotenv, mo, os, pd, plt, sc, session_info, shutil, warnings


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Settings

    Here we set some global settings and read in some environment variables.
    """
    )
    return


@app.cell
def _(cr, sc, warnings):
    sc.settings.verbosity = 0
    cr.settings.verbosity = 0
    warnings.simplefilter('ignore', category=UserWarning)
    warnings.simplefilter('ignore', category=FutureWarning)
    return


@app.cell
def _(load_dotenv):
    load_dotenv(dotenv_path='../.env')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Data

    We begin by loading in the human bone marrow dataset included in the `cellrank` package. For simplicity's sake, we remove the common lymphoid progenitor (CLP) celltype as it isn't relevant to any of the other lineages.
    """
    )
    return


@app.cell
def _(cr):
    ad_bone = cr.datasets.bone_marrow()
    ad_bone.layers['counts'] = ad_bone.layers['spliced'] + ad_bone.layers['unspliced']
    ad_bone.obs['cell'] = ad_bone.obs.index.to_list()
    ad_bone.var['gene'] = ad_bone.var.index.to_list()
    ad_bone.obs.rename(columns={'clusters': 'celltype'}, inplace=True)
    ad_bone = ad_bone[~ad_bone.obs['celltype'].str.contains('CLP', na=False), :].copy()
    ad_bone.raw = ad_bone
    return (ad_bone,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Since downloading the dataset using `cellrank` creates a cache directory called `datasets/` in our current directory, we remove it (if the directory exists). This is done because we don't want to accidentally commit a large data file to our GitHub repository.""")
    return


@app.cell
def _(os, shutil):
    if os.path.isdir('datasets/'):
        try: 
            shutil.rmtree('datasets/')
        except Exception as e:
            print('Error removing the datasets/ directory.')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Analysis 

    ## Preprocessing the scRNA-seq data 

    We start by performing some very basic QC on our genes and cells.
    """
    )
    return


@app.cell
def _(ad_bone, sc):
    sc.pp.filter_cells(ad_bone, min_counts=1000)
    sc.pp.filter_genes(ad_bone, min_cells=5)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Next we identify a set of 3,000 highly variable genes (HVGs).""")
    return


@app.cell
def _(ad_bone, sc):
    sc.pp.highly_variable_genes(
        ad_bone, 
        n_top_genes=3000, 
        flavor='seurat_v3', 
        subset=False
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Continuing on, we depth-normalize and log1p-transform the raw counts, making sure to save the normalized counts matrix in a new `layer`.""")
    return


@app.cell
def _(ad_bone, sc):
    ad_bone.X = sc.pp.normalize_total(ad_bone, target_sum=1e4, inplace=False)['X']
    sc.pp.log1p(ad_bone)
    ad_bone.layers['norm'] = ad_bone.X.copy()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""After scaling the normalized counts, we run PCA using the HVG set we identified earlier.""")
    return


@app.cell
def _(ad_bone, sc):
    sc.pp.scale(ad_bone)
    sc.tl.pca(
        ad_bone, 
        n_comps=50, 
        random_state=312, 
        mask_var='highly_variable'
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Next, we generate a shared nearest neighbors (SNN) graph in PCA using $k = 20$ neighbors per cell, then partition the graph into clusters via the Leiden algorithm.""")
    return


@app.cell
def _(ad_bone, sc):
    sc.pp.neighbors(
        ad_bone, 
        n_neighbors=20,
        n_pcs=30,  
        use_rep='X_pca', 
        metric='cosine', 
        random_state=312
    )
    sc.tl.leiden(
        ad_bone, 
        resolution=0.3, 
        flavor='igraph',
        n_iterations=2, 
        random_state=312
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""We check out the clustering results in comparison to the ground-truth celltypes using our PCA embedding. The PCA embedding clearly picked up on some of the structure in our dataset, but the first 2 PCs don't show clear directionality within the myeloid and erythroid lineages which isn't ideal.""")
    return


@app.cell
def _(ad_bone, plt, sc):
    _fig, _axes = plt.subplots(
        nrows=1, 
        ncols=2, 
        figsize = (15, 5), 
        sharex=True, 
        sharey=True
    )
    sc.pl.embedding(
        ad_bone, 
        basis='pca', 
        color='leiden',
        title='Leiden',
        frameon=True, 
        size=30, 
        alpha=0.75,
        show=False, 
        ax=_axes[0]
    )
    sc.pl.embedding(
        ad_bone, 
        basis='pca', 
        color='celltype',
        title='Celltype',
        frameon=True, 
        size=30, 
        alpha=0.75,
        show=False, 
        ax=_axes[1]
    )
    for _ax in _axes:
        _ax.set_xlabel('PC 1')
        _ax.set_ylabel('PC 2')
    _fig.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Next, we compute a PAGA embedding of the connectivities between the celltypes.""")
    return


@app.cell
def _(ad_bone, sc):
    sc.tl.paga(ad_bone, groups='celltype')
    return


@app.cell
def _():
    return


@app.cell
def _(ad_bone, sc):
    sc.pl.paga(
        ad_bone, 
        random_state=312, 
        frameon=True
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Next, we further reduce dimensionality down to 2D via the UMAP algorithm.""")
    return


@app.cell
def _(ad_bone, sc):
    sc.tl.umap(
        ad_bone, 
        init_pos='paga', 
        random_state=312
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The UMAP embedding places the celltypes in a pretty uninspiring blob, which isn't ideal for trajectory analysis.""")
    return


@app.cell
def _(ad_bone, plt, sc):
    _fig, _axes = plt.subplots(
        nrows=1, 
        ncols=2, 
        figsize = (15, 5), 
        sharex=True, 
        sharey=True
    )
    sc.pl.embedding(
        ad_bone, 
        basis='umap', 
        color='leiden',
        title='Leiden',
        frameon=True, 
        size=30, 
        alpha=0.75,
        show=False, 
        ax=_axes[0]
    )
    sc.pl.embedding(
        ad_bone, 
        basis='umap', 
        color='celltype',
        title='Celltype',
        frameon=True, 
        size=30, 
        alpha=0.75,
        show=False, 
        ax=_axes[1]
    )
    for _ax in _axes:
        _ax.set_xlabel('UMAP 1')
        _ax.set_ylabel('UMAP 2')
    _fig.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Lastly, we visualize the Leiden clusters and ground-truth celltype labels on the t-SNE embedding that came with the dataset. This embedding is exactly what we want - the lineages are distinct but still mostly connected.""")
    return


@app.cell
def _(ad_bone, plt, sc):
    _fig, _axes = plt.subplots(
        nrows=1, 
        ncols=2, 
        figsize = (15, 5), 
        sharex=True, 
        sharey=True
    )
    sc.pl.embedding(
        ad_bone, 
        basis='tsne', 
        color='leiden',
        title='Leiden',
        frameon=True, 
        size=30, 
        alpha=0.75,
        show=False, 
        ax=_axes[0]
    )
    sc.pl.embedding(
        ad_bone, 
        basis='tsne', 
        color='celltype',
        title='Celltype',
        frameon=True, 
        size=30, 
        alpha=0.75,
        show=False, 
        ax=_axes[1]
    )
    for _ax in _axes:
        _ax.set_xlabel('FA 1')
        _ax.set_ylabel('FA 2')
    _fig.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Estimating terminal cell fates""")
    return


@app.cell
def _(ad_bone, cr):
    pk = cr.kernels.PseudotimeKernel(ad_bone, time_key='palantir_pseudotime')
    pk.compute_transition_matrix(
        threshold_scheme='soft', 
        n_jobs=2, 
        show_progress_bar=False
    )
    return (pk,)


@app.cell
def _(pk, plt):
    pk.plot_projection(
        basis='tsne', 
        recompute=True, 
        color='celltype', 
        frameon=True, 
        show=False, 
        legend_loc='right margin'
    )
    plt.gca().set_xlabel('t-SNE 1')
    plt.gca().set_ylabel('t-SNE 2')
    plt.show()
    return


@app.cell
def _(cr, pk):
    g = cr.estimators.GPCCA(pk)
    g.compute_schur()
    return (g,)


@app.cell
def _(g):
    g.compute_macrostates(n_states=12, cluster_key='celltype')
    return


@app.cell
def _(g, plt):
    g.plot_macrostates(
        which='all', 
        same_plot=True, 
        basis='tsne', 
        frameon=True, 
        show=False
    )
    plt.gca().set_xlabel('t-SNE 1')
    plt.gca().set_xlabel('t-SNE 2')
    plt.show()
    return


@app.cell
def _(g):
    g.set_terminal_states(states=['DCs_2', 'Mono_1_1', 'Ery_2', 'Mega'])
    g.set_initial_states(states=['HSC_1'])
    return


@app.cell
def _(g):
    g.compute_fate_probabilities(n_jobs=2, show_progress_bar=False)
    return


@app.cell
def _(g):
    g.plot_fate_probabilities(
        same_plot=False, 
        basis='tsne', 
        frameon=True
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Next, we compute a set of genes that drive commitment to each terminal cell fate (AKA driver genes). We add the resulting `DataFrame` to a `list` object that is grouped by terminal cell fate. """)
    return


@app.cell
def _(g):
    driver_dfs = []
    cell_fates = ['DCs_2', 'Mono_1_1', 'Ery_2', 'Mega']
    for _fate in cell_fates:
        _df = g.compute_lineage_drivers(
            lineages=_fate, 
            cluster_key='celltype', 
            seed=312, 
            n_jobs=2, 
            show_progress_bar=False
        )
        _df['gene'] = _df.index.to_list()
        cor_column = _df.columns[0]
        _df.query(f'{cor_column} > 0', inplace=True)
        driver_dfs.append(_df)
    return cell_fates, driver_dfs


@app.cell
def _(cell_fates, driver_dfs, pd):
    top30_drivers = pd.DataFrame()
    for _df, _fate in zip(driver_dfs, cell_fates):
        df_top30 = _df.head(n=30)
        drivers_per_celltype = pd.DataFrame({
            'celltype': df_top30.columns[0].replace('_corr', ''), 
            'driver_gene': df_top30['gene'].to_list()
        })
        top30_drivers = pd.concat([top30_drivers, drivers_per_celltype])
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Session information""")
    return


@app.cell
def _(session_info):
    session_info.show()
    return


if __name__ == "__main__":
    app.run()
