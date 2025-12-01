import marimo

__generated_with = "0.18.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Libraries

    Here we import all the packages we'll need to load our data and perform our analysis.
    """)
    return


@app.cell
def _():
    import os
    import re
    import time
    import json
    import openai
    import shutil
    import pickle
    import getpass
    import warnings
    import numpy as np
    import igraph as ig
    import scanpy as sc
    import pandas as pd
    import marimo as mo
    import session_info
    import squidpy as sq
    import anndata as ad
    import matplotlib.style
    import genesetgpt as gpt
    from openai import OpenAI
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
        OpenAI,
        PCA,
        StandardScaler,
        ThreadPoolExecutor,
        datetime,
        getpass,
        gpt,
        ig,
        json,
        load_dotenv,
        matplotlib,
        mo,
        np,
        os,
        pandarallel,
        partial,
        pd,
        plt,
        sc,
        session_info,
        shutil,
        sq,
        warnings,
    )


@app.cell
def _(matplotlib, sc, warnings):
    sc.settings.verbosity = 0
    matplotlib.style.use('default')
    warnings.simplefilter('ignore', category=UserWarning)
    warnings.simplefilter('ignore', category=FutureWarning)
    return


@app.cell
def _(plt):
    plt.rcParams.update({
        'font.size': 8, 
        'axes.linewidth': 1.5, 
        'legend.frameon': False, 
        'figure.dpi': 320, 
        'font.family': 'Arial'
    })
    palette_leiden = plt.colormaps['Accent']
    return (palette_leiden,)


@app.cell
def _(load_dotenv):
    load_dotenv(dotenv_path='.env')
    return


@app.cell
def _(sq):
    ad_brain = sq.datasets.visium(sample_id='V1_Human_Brain_Section_1')
    ad_brain.layers['counts'] = ad_brain.X.copy()
    ad_brain.var['gene'] = ad_brain.var.index.to_list()
    ad_brain.var_names_make_unique()
    ad_brain.raw = ad_brain
    return (ad_brain,)


@app.cell
def _(os, shutil):
    if os.path.isdir('data/'):
        try: 
            shutil.rmtree('data/')
        except Exception as e:
            print('Error removing the data/ directory.')
    return


@app.cell
def _(ad_brain, sc):
    sc.pp.filter_cells(ad_brain, min_counts=1000)
    sc.pp.filter_genes(ad_brain, min_cells=5)
    return


@app.cell
def _(ad_brain, sc):
    sc.pp.highly_variable_genes(
        ad_brain, 
        n_top_genes=3000, 
        flavor='seurat_v3', 
        subset=False
    )
    return


@app.cell
def _(ad_brain, sc):
    ad_brain.X = sc.pp.normalize_total(ad_brain, target_sum=1e4, inplace=False)['X']
    sc.pp.log1p(ad_brain)
    ad_brain.layers['norm'] = ad_brain.X.copy()
    return


@app.cell
def _(ad_brain, sc):
    sc.pp.scale(ad_brain)
    sc.tl.pca(
        ad_brain, 
        n_comps=50, 
        random_state=312, 
        mask_var='highly_variable'
    )
    return


@app.cell
def _(ad_brain, sc):
    sc.pp.neighbors(
        ad_brain, 
        n_neighbors=20,
        n_pcs=30,  
        use_rep='X_pca', 
        metric='cosine', 
        random_state=312
    )
    sc.tl.leiden(
        ad_brain, 
        resolution=0.5, 
        flavor='igraph',
        n_iterations=2, 
        random_state=312
    )
    return


@app.cell
def _(ad_brain, plt, sc):
    sc.pl.embedding(
        ad_brain, 
        basis='pca', 
        color='leiden',
        title='Leiden',
        frameon=True, 
        size=30, 
        alpha=0.75,
        show=False
    )
    plt.gca().set_xlabel('PC 1')
    plt.gca().set_ylabel('PC 2')
    plt.show()
    return


@app.cell
def _(ad_brain, sc):
    sc.tl.umap(ad_brain, random_state=312)
    return


@app.cell
def _(ad_brain):
    ad_brain.layers
    return


@app.cell
def _(ad_brain, np, palette_leiden, pd, plt):
    _plot_df = pd.DataFrame({
        'umap1': ad_brain.obsm['X_umap'][:,0], 
        'umap2': ad_brain.obsm['X_umap'][:,1],
        'leiden': ad_brain.obs['leiden']
    })
    _clusters = np.sort(_plot_df['leiden'].unique().astype(int))
    _cluster_to_color = {
        cl: palette_leiden(i / max(1, len(_clusters) - 1))
        for i, cl in enumerate(_clusters)
    }
    _plot_df['color'] = _plot_df['leiden'].astype(int).map(_cluster_to_color)
    _fig, _ax = plt.subplots(figsize=(4.0, 3.0))
    _fig.subplots_adjust(left=0.01, right=0.79, bottom=0.01, top=0.99)
    _ax.scatter(
        _plot_df['umap1'],
        _plot_df['umap2'],
        c=_plot_df['color'], 
        s=10,
        linewidths=0.0, 
        alpha=0.8
    )
    _ax.set_xlabel('UMAP 1')
    _ax.set_ylabel('UMAP 2')
    _ax.set_xticks([])
    _ax.set_yticks([])
    _ax.set_yticklabels([])
    _ax.set_xticklabels([])
    _ax.spines[['right', 'top']].set_visible(False)
    for _spine in _ax.spines.values():
        _spine.set_linewidth(1.5)
    plt.tight_layout()
    plt.savefig('figures/brain_umap_scatter.png', bbox_inches='tight')
    plt.show()
    return


@app.cell
def _(ad_brain, plt, sc):
    with plt.rc_context({'figure.figsize': (4, 3)}):
        sc.pl.embedding(
            ad_brain, 
            basis='umap', 
            color='leiden',
            title='',
            frameon=True, 
            size=30, 
            alpha=0.75,
            show=False
        )
        plt.gca().spines[['right', 'top']].set_visible(False)
        plt.gca().set_xlabel('UMAP 1')
        plt.gca().set_ylabel('UMAP 2')
        plt.gca().get_legend().remove()
        plt.gcf().legend(
            title='Leiden', 
            loc='outside right', 
            borderaxespad=0.75
        )
        plt.savefig('figures/brain_umap_scatter.png', bbox_inches='tight')
        plt.show()
    return


@app.cell
def _(ad_brain, plt, sq):
    with plt.rc_context({'figure.figsize': (4, 3.33), 'axes.linewidth': 1.5}):
        sq.pl.spatial_scatter(
            ad_brain, 
            shape='hex',
            color='leiden', 
            title='', 
            img=False, 
            size=1.5
        )
        plt.gca().set_xlabel('Spatial 1')
        plt.gca().set_ylabel('Spatial 2')
        plt.gca().get_legend().remove()
        plt.gcf().legend(
            title='Leiden', 
            loc='outside right', 
            borderaxespad=1
        )
        plt.savefig('figures/brain_spatial_scatter.png', bbox_inches='tight')
        plt.show()
    return


@app.cell
def _(ad_brain, np, palette_leiden, pd, plt):
    _plot_df = pd.DataFrame({
        'x': ad_brain.obsm['spatial'][:,0], 
        'y': ad_brain.obsm['spatial'][:,1],
        'leiden': ad_brain.obs['leiden']
    })
    _clusters = np.sort(_plot_df['leiden'].unique().astype(int))
    _cluster_to_color = {
        cl: palette_leiden(i / max(1, len(_clusters) - 1))
        for i, cl in enumerate(_clusters)
    }
    _plot_df['color'] = _plot_df['leiden'].astype(int).map(_cluster_to_color)
    _fig, _ax = plt.subplots(figsize=(3.3, 3.0))
    _fig.subplots_adjust(left=0.01, right=0.79, bottom=0.01, top=0.99)
    _ax.scatter(
        _plot_df['x'],
        _plot_df['y'],
        c=_plot_df['color'], 
        s=8,
        linewidths=0.0, 
        marker='H'
    )
    _ax.invert_yaxis()
    _ax.set_xlabel('Spatial 1')
    _ax.set_ylabel('Spatial 2')
    _ax.set_xticks([])
    _ax.set_yticks([])
    _ax.set_yticklabels([])
    _ax.set_xticklabels([])
    for spine in _ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
    for cl in _clusters:
        _ax.scatter([], [], color=_cluster_to_color[cl], label=cl, s=30)
    leg = _ax.legend(
        title='Leiden',
        bbox_to_anchor=(1, 0.65),
        loc='upper left',
        frameon=False,
        borderpad=0.2
    )
    # plt.tight_layout()
    plt.savefig('figures/brain_spatial_scatter.png', bbox_inches='tight')
    plt.show()
    return


@app.cell
def _(ad_brain, sq):
    sq.gr.spatial_neighbors(ad_brain, n_neighs=10)
    return


@app.cell
def _(ad_brain, sq):
    top3k_hvgs = ad_brain.var[ad_brain.var['highly_variable']]['gene'].to_list()
    sq.gr.spatial_autocorr(
        ad_brain,
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
    expr_mtx_scaled = scaler.fit_transform(expr_mtx)
    return (expr_mtx_scaled,)


@app.cell
def _(PCA, expr_mtx_scaled):
    pca = PCA(n_components=30, random_state=312)
    pc_mtx = pca.fit_transform(expr_mtx_scaled)
    return (pc_mtx,)


@app.cell
def _(NearestNeighbors, ig, np, pc_mtx):
    nns = NearestNeighbors(n_neighbors=20, metric='cosine').fit(pc_mtx)
    knn_graph = nns.kneighbors_graph(pc_mtx, mode='connectivity')
    adj_mtx = knn_graph.toarray()
    adj_mtx = np.maximum(adj_mtx, adj_mtx.T)
    g = ig.Graph.Adjacency((adj_mtx > 0).tolist(), mode=ig.ADJ_UNDIRECTED)
    partition = g.community_leiden(resolution=0.02)
    return (partition,)


@app.cell
def _(np, partition, pd, top1k_svgs):
    cluster_df = pd.DataFrame({
        'gene': top1k_svgs, 
        'leiden': np.array(partition.membership)
    })
    return (cluster_df,)


@app.cell
def _(cluster_df):
    genes_clust0 = cluster_df.query('leiden == 0')['gene'].to_list()
    genes_clust1 = cluster_df.query('leiden == 1')['gene'].to_list()
    genes_clust2 = cluster_df.query('leiden == 2')['gene'].to_list()
    genes_clust3 = cluster_df.query('leiden == 3')['gene'].to_list()
    return genes_clust0, genes_clust1, genes_clust2, genes_clust3


@app.cell
def _(ad_brain, genes_clust0, genes_clust1, genes_clust2, genes_clust3, sc):
    sc.tl.score_genes(
        ad_brain, 
        gene_list=genes_clust0,
        score_name='svg_cluster0', 
        random_state=312, 
        use_raw=False
    )
    sc.tl.score_genes(
        ad_brain, 
        gene_list=genes_clust1,
        score_name='svg_cluster1', 
        random_state=312, 
        use_raw=False
    )
    sc.tl.score_genes(
        ad_brain, 
        gene_list=genes_clust2,
        score_name='svg_cluster2', 
        random_state=312, 
        use_raw=False
    )
    sc.tl.score_genes(
        ad_brain, 
        gene_list=genes_clust3,
        score_name='svg_cluster3', 
        random_state=312, 
        use_raw=False
    )
    return


@app.cell
def _(ad_brain, plt, sq):
    svg_clusters = [0, 1, 2, 3]
    plot_titles = [f'SVG Cluster {c}' for c in svg_clusters]
    fig, axes = plt.subplots(
        nrows=1, 
        ncols=4, 
        figsize = (15, 5), 
        sharex=True, 
        sharey=True
    )
    for ax, c, title in zip(axes, svg_clusters, plot_titles):
        sq.pl.spatial_scatter(
            ad_brain,
            shape='hex',
            color=f'svg_cluster{c}',
            title=title,
            ax=ax
        )
        ax.set_xlabel('Spatial 1')
        ax.set_ylabel('Spatial 2')
    fig.tight_layout()
    plt.show()
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
        nb_workers=3, 
        verbose=0
    )
    return


@app.cell
def _(gpt, mim_table, os, svg_gene_ids):
    svg_gene_ids['prompt_user'] = svg_gene_ids.parallel_apply(
        lambda row: 
        gpt.build_user_prompt(
            ensembl_id=row['ensembl_id'], 
            hgnc_symbol=row['hgnc_symbol'], 
            entrez_id=row['entrez_id'], 
            entrez_email='j.leary@ufl.edu', 
            mim_mapping_table=mim_table, 
            mim_api_key=os.getenv('MIM_API_KEY'), 
            include_aliases=True
        ), 
        axis=1
    )
    return


@app.cell
def _():
    prompt_dev = 'You are an experienced computational biologist with advanced knowledge of transcriptomics analyses such as single-cell RNA-seq and spatial transcriptomics. When generating responses, you consider the statistical, computational, and biological angles of the question at hand. Your responses are detailed without being too overly technical. The system being studied is the human brain, and the data were assayed using 10X Genomics Visium V1.'
    return (prompt_dev,)


@app.cell
def _(OpenAI, os):
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    return (client,)


@app.cell
def _(ThreadPoolExecutor, client, gpt, partial, prompt_dev, svg_gene_ids):
    summarize_one = partial(
        gpt.summarize_genes,
        prompt_dev=prompt_dev,
        openai_client=client,
        openai_model='gpt-5-mini'
    )
    user_prompts = svg_gene_ids['prompt_user'].to_list()
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(summarize_one, user_prompts))
    llm_summaries, llm_scores = zip(*results)
    return llm_scores, llm_summaries


@app.cell
def _(llm_scores, llm_summaries, svg_gene_ids):
    svg_gene_ids['llm_summary'] = llm_summaries
    svg_gene_ids['llm_confidence_score'] = llm_scores
    return


@app.cell
def _(BaseModel):
    class GeneSetSummary(BaseModel):
        summary: str
        name: str
        confidence: float
    return (GeneSetSummary,)


@app.cell
def _(cluster_df):
    unique_svg_clusters = list(set(cluster_df['leiden']))
    return (unique_svg_clusters,)


@app.cell
def _(
    GeneSetSummary,
    client,
    cluster_df,
    prompt_dev,
    svg_gene_ids,
    unique_svg_clusters,
):
    cluster_summaries = []
    cluster_names = []
    cluster_scores = []
    model_jsons = []
    for clust in unique_svg_clusters:
        cluster_genes = cluster_df.query(f'leiden == {clust}')['gene'].to_list()
        cluster_genes_str = ', '.join(cluster_genes)
        cluster_gene_ids = svg_gene_ids.query('hgnc_symbol in @cluster_genes').copy()
        cluster_user_prompts = cluster_gene_ids['prompt_user'].to_list()
        cluster_llm_summaries_bulleted = '\n'.join(f'- {s}' for s in cluster_user_prompts)
        summary_prompt = f"""
        Below are brief, independent descriptions of genes in a set:

        {cluster_llm_summaries_bulleted}

        Please write a concise (5–7 sentence) paragraph summarizing the common function(s) of this gene set. In addition, please provide a robust, 3-decimal score ranging from 0-1 estimating how confident you are in your overall annotation. Lastly, provide a short 2-5 word name for the gene set based on your annotation.
        """
        summary_response = client.responses.parse(
            model='gpt-5-mini', 
            input=[
                {'role': 'developer', 'content': prompt_dev}, 
                {'role': 'user', 'content': summary_prompt}
            ], 
            text_format=GeneSetSummary
        )
        cluster_summaries.append(summary_response.output_parsed.summary)
        cluster_names.append(summary_response.output_parsed.name)
        cluster_scores.append(summary_response.output_parsed.confidence)
        model_jsons.append(summary_response.model_dump_json())
    return cluster_names, cluster_scores, cluster_summaries, model_jsons


@app.cell
def _(
    cluster_names,
    cluster_scores,
    cluster_summaries,
    pd,
    unique_svg_clusters,
):
    final_summary_df = pd.DataFrame({
        'cluster': unique_svg_clusters, 
        'summary': cluster_summaries, 
        'name': cluster_names, 
        'score': cluster_scores
    })
    return (final_summary_df,)


@app.cell
def _(final_summary_df, mo):
    mo.md(final_summary_df['summary'].to_list()[0])
    return


@app.cell
def _(final_summary_df, mo):
    mo.md(final_summary_df['summary'].to_list()[1])
    return


@app.cell
def _(final_summary_df, mo):
    mo.md(final_summary_df['summary'].to_list()[2])
    return


@app.cell
def _(final_summary_df, mo):
    mo.md(final_summary_df['summary'].to_list()[3])
    return


@app.cell
def _(ad_brain):
    ad_brain.write_h5ad('data/human-brain-spatial/ad_brain.h5ad')
    return


@app.cell
def _(datetime, getpass, svg_gene_ids):
    svg_gene_ids['timestamp'] = datetime.now()
    svg_gene_ids['model'] = 'gpt-5-mini'
    svg_gene_ids['author'] = getpass.getuser()
    svg_gene_ids.to_pickle('data/human-brain-spatial/svg_gene_ids.pkl')
    return


@app.cell
def _(final_summary_df):
    final_summary_df.to_pickle('data/human-brain-spatial/final_summary_df.pkl')
    return


@app.cell
def _(json, model_jsons):
    with open('data/human-brain-spatial/model_jsons.json', 'w') as f:
        json.dump(model_jsons, f, indent=2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Session information
    """)
    return


@app.cell
def _(session_info):
    session_info.show()
    return


if __name__ == "__main__":
    app.run()
