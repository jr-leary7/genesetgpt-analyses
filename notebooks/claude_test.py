import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo


@app.cell
def _():
    import os
    import re
    import json
    import shutil
    import pickle
    import getpass
    import warnings
    import anthropic
    import numpy as np
    import igraph as ig
    import scanpy as sc
    import pandas as pd
    import session_info
    import squidpy as sq
    import anndata as ad
    import matplotlib.style
    import genesetgpt as gpt
    from datetime import datetime
    from functools import partial
    from pydantic import BaseModel
    from dotenv import load_dotenv
    import matplotlib.pyplot as plt
    from pandarallel import pandarallel
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler
    from concurrent.futures import ThreadPoolExecutor

    return (
        BaseModel,
        NearestNeighbors,
        PCA,
        StandardScaler,
        anthropic,
        gpt,
        ig,
        load_dotenv,
        matplotlib,
        np,
        os,
        pandarallel,
        pd,
        plt,
        sc,
        shutil,
        sq,
        warnings,
    )


@app.cell
def _(sc, warnings):
    sc.settings.verbosity = 0
    warnings.simplefilter(action='ignore')
    mo._runtime.context.get_context().marimo_config['runtime']['output_max_bytes'] = 100_000_000
    return


@app.cell
def _(matplotlib, plt):
    matplotlib.style.use('default')
    plt.rcParams.update({
        'font.size': 12, 
        'axes.linewidth': 1.5, 
        'legend.frameon': False, 
        'figure.dpi': 320, 
        'font.family': 'Arial'
    })
    return


@app.cell
def _(load_dotenv):
    load_dotenv()
    return


@app.cell
def _():
    return


@app.cell
def _(sq):
    ad_brain = sq.datasets.visium(sample_id='V1_Human_Brain_Section_1')
    ad_brain.layers['counts'] = ad_brain.X.copy()
    ad_brain.var_names_make_unique()
    ad_brain.var['gene'] = ad_brain.var.index.to_list()
    ad_brain.raw = ad_brain
    return (ad_brain,)


@app.cell
def _(os, shutil):
    if os.path.isdir('data/V1_Human_Brain_Section_1/'):
        try: 
            shutil.rmtree('data/V1_Human_Brain_Section_1/')
        except Exception as e:
            print('Error removing the data/V1_Human_Brain_Section_1/ directory.')
    return


@app.cell
def _(ad_brain, sc):
    sc.pp.filter_cells(data=ad_brain, min_counts=1000)
    sc.pp.filter_genes(data=ad_brain, min_cells=5)
    return


@app.cell
def _(ad_brain, sc):
    sc.pp.highly_variable_genes(
        adata=ad_brain, 
        n_top_genes=3000, 
        flavor='seurat_v3', 
        subset=False
    )
    return


@app.cell
def _(ad_brain, sc):
    ad_brain.X = sc.pp.normalize_total(adata=ad_brain, target_sum=1e4, inplace=False)['X']
    sc.pp.log1p(ad_brain)
    ad_brain.layers['norm'] = ad_brain.X.copy()
    return


@app.cell
def _(ad_brain, sc):
    sc.pp.scale(ad_brain)
    sc.pp.pca(
        data=ad_brain, 
        n_comps=50, 
        random_state=312, 
        mask_var='highly_variable'
    )
    return


@app.cell
def _(ad_brain, sc):
    sc.pp.neighbors(
        adata=ad_brain, 
        n_neighbors=20,
        n_pcs=30,  
        use_rep='X_pca', 
        metric='cosine', 
        random_state=312
    )
    sc.tl.leiden(
        adata=ad_brain, 
        resolution=0.5, 
        flavor='igraph',
        n_iterations=2, 
        random_state=312
    )
    return


@app.cell
def _(ad_brain, sc):
    sc.tl.umap(adata=ad_brain, random_state=312)
    return


@app.cell
def _(ad_brain, sq):
    sq.gr.spatial_neighbors(adata=ad_brain, n_neighs=10)
    return


@app.cell
def _(ad_brain, sq):
    top3k_hvgs = ad_brain.var[ad_brain.var['highly_variable']]['gene'].to_list()
    sq.gr.spatial_autocorr(
        adata=ad_brain,
        mode='moran',
        genes=top3k_hvgs, 
        use_raw=False, 
        layer='norm', 
        n_perms=100,
        n_jobs=4, 
        seed=312
    )
    return


@app.cell
def _(ad_brain):
    moran_df = ad_brain.uns['moranI'].copy()
    moran_df.query('pval_sim_fdr_bh < 0.05', inplace=True)
    moran_df.sort_values(
        by='I',
        key=lambda col: col.abs(),
        ascending=False, 
        inplace=True
    )
    top1k_svgs = moran_df.index.to_list()[:1000]
    ad_brain.var['spatially_variable'] = ad_brain.var_names.isin(top1k_svgs)
    return (top1k_svgs,)


@app.cell
def _(StandardScaler, ad_brain, top1k_svgs):
    expr_mtx = ad_brain[:, top1k_svgs].layers['norm'].T.toarray()
    scaler = StandardScaler(with_mean=True, with_std=True)
    expr_mtx_scaled = scaler.fit_transform(X=expr_mtx)
    return (expr_mtx_scaled,)


@app.cell
def _(PCA, expr_mtx_scaled):
    pca = PCA(n_components=30, random_state=312)
    pc_mtx = pca.fit_transform(X=expr_mtx_scaled)
    return (pc_mtx,)


@app.cell
def _(NearestNeighbors, ig, np, pc_mtx, pd, top1k_svgs):
    nns = NearestNeighbors(n_neighbors=20, metric='cosine').fit(X=pc_mtx)
    knn_graph = nns.kneighbors_graph(X=pc_mtx, mode='connectivity')
    adj_mtx = knn_graph.toarray()
    adj_mtx = np.maximum(adj_mtx, adj_mtx.T)
    g = ig.Graph.Adjacency((adj_mtx > 0).tolist(), mode=ig.ADJ_UNDIRECTED)
    partition = g.community_leiden(resolution=0.01)
    cluster_df = pd.DataFrame({
        'gene': top1k_svgs, 
        'leiden': np.array(partition.membership)
    })
    return (cluster_df,)


@app.cell
def _(cluster_df):
    module_gene_dict = {
        cl: cluster_df.query(f'leiden == {cl}')['gene'].to_list()
        for cl in cluster_df['leiden'].unique()
    }
    return (module_gene_dict,)


@app.cell
def _(ad_brain, module_gene_dict, sc):
    for cl, genes in module_gene_dict.items():
        sc.tl.score_genes(
            adata=ad_brain,
            gene_list=genes,
            score_name=f'svg_module{cl}',
            random_state=312,
            use_raw=False,
            layer='norm'
        )
    return


@app.cell
def _():
    return


@app.cell
def _(gpt):
    all_hs_genes = gpt.fetch_gene_table()
    mim_table = gpt.fetch_mim_table()
    return all_hs_genes, mim_table


@app.cell
def _(all_hs_genes):
    svg_gene_ids = all_hs_genes.query('hgnc_symbol in @top1k_svgs').copy()
    svg_gene_ids.dropna(inplace=True)
    return (svg_gene_ids,)


@app.cell
def _(pandarallel):
    pandarallel.initialize(
        progress_bar=True, 
        nb_workers=2, 
        verbose=0
    )
    return


@app.cell
def _(gpt, mim_table, os, svg_gene_ids):
    mim_key = os.getenv('MIM_API_KEY')
    svg_gene_ids['prompt_user'] = svg_gene_ids.parallel_apply(
        lambda row: 
        gpt.build_user_prompt(
            ensembl_id=row['ensembl_id'], 
            hgnc_symbol=row['hgnc_symbol'], 
            entrez_id=row['entrez_id'], 
            entrez_email='j.leary@ufl.edu', 
            mim_mapping_table=mim_table, 
            mim_api_key=mim_key, 
            include_aliases=True
        ), 
        axis=1
    )
    return


@app.cell
def _():
    return


@app.cell
def _(anthropic, os):
    claude_client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
    return (claude_client,)


@app.cell
def _(claude_client):
    claude_client.__class__
    return


@app.cell
def _():
    prompt_system = 'You are an experienced computational biologist with advanced knowledge of transcriptomics analyses such as single-cell RNA-seq and spatially-resolved transcriptomics. When generating responses, you consider the statistical, computational, and biological angles of the question at hand. Your responses are detailed without being too overly technical. The system being studied is the healthy human cortex, and the data were assayed using 10X Genomics Visium V1.'
    return (prompt_system,)


@app.cell
def _(svg_gene_ids):
    svg_gene_ids['prompt_user'].to_list()[0]
    return


@app.cell
def _(anthropic):
    def get_completion(prompt_user: str = None,
                       prompt_system: str = None,
                       model: str = 'claude-haiku-4-5', 
                       n_max_tokens: int = 2000, 
                       client: anthropic.Anthropic = None) -> str:
        if prompt_user is None:
            raise ValueError('The prompt_user string argument must be supplied.')
        if client is None:
            raise ValueError('The Anthropic client argument must be supplied.')
        message = client.messages.create(
            model=model,
            max_tokens=n_max_tokens,
            system=prompt_system,
            messages=[
              {'role': 'user', 'content': prompt_user}
            ]
        )
        res = message.content[0].text
        return res

    return (get_completion,)


@app.cell
def _(claude_client, get_completion, prompt_system, svg_gene_ids):
    sumy_test = get_completion(
        prompt_user=svg_gene_ids['prompt_user'].to_list()[0], 
        prompt_system=prompt_system, 
        client=claude_client
    )
    return (sumy_test,)


@app.cell
def _(sumy_test):
    mo.md(sumy_test)
    return


@app.cell
def _():
    return


@app.cell
def _(BaseModel):
    class GeneSummary(BaseModel):
        gene_summary: str
        confidence_score: float
        score_rationale: str

    return (GeneSummary,)


@app.cell
def _(anthropic):
    def get_completion_struct(prompt_user: str = None,
                              prompt_system: str = None,
                              model: str = 'claude-haiku-4-5', 
                              n_max_tokens: int = 2000, 
                              client: anthropic.Anthropic = None, 
                              output_struct = None):
        if prompt_user is None:
            raise ValueError('The prompt_user string argument must be supplied.')
        if client is None:
            raise ValueError('The Anthropic client argument must be supplied.')
        response = client.messages.parse(
            model=model, 
            max_tokens=n_max_tokens, 
            system=prompt_system, 
            messages=[
                {'role': 'user', 'content': prompt_user}
            ], 
            output_format=output_struct
        )
        return response  

    return (get_completion_struct,)


@app.cell
def _(sumy_test_struct):
    sumy_test_struct.__class__
    return


@app.cell
def _(sumy_test_struct):
    type(sumy_test_struct)
    return


@app.cell
def _(
    GeneSummary,
    claude_client,
    get_completion_struct,
    prompt_system,
    svg_gene_ids,
):
    sumy_test_struct = get_completion_struct(
        prompt_user=svg_gene_ids['prompt_user'].to_list()[0], 
        prompt_system=prompt_system, 
        client=claude_client, 
        output_struct=GeneSummary
    )
    return (sumy_test_struct,)


@app.cell
def _(sumy_test_struct):
    mo.md(sumy_test_struct.parsed_output.gene_summary)
    return


@app.cell
def _(sumy_test_struct):
    sumy_test_struct.parsed_output.confidence_score
    return


@app.cell
def _(sumy_test_struct):
    mo.md(sumy_test_struct.parsed_output.score_rationale)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
