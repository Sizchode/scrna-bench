import scanpy as sc
import pandas as pd
import numpy as np
import openai
import re
import scipy.sparse
import os
import subprocess
from .tool_base import MultiToolBase


def ensure_highly_variable_genes(adata):
    # Copy the original data matrix
    original_X = adata.X.copy()
    # Check if 'highly_variable' column exists in adata.var
    if 'highly_variable' not in adata.var.columns:
        # Normalize the total counts per cell
        sc.pp.normalize_total(adata, target_sum=1e4)
        # Logarithmize the data
        sc.pp.log1p(adata)
        # Identify highly variable genes
        sc.pp.highly_variable_genes(adata)
    # Restore the original data matrix
    adata.X = original_X


class SinglecellSequencing_toolkit(MultiToolBase):
    def __init__(self, cfg=None):
        """Initialize the Single-cell Sequencing Toolkit."""
        super().__init__()
        self.tool_name = "SingleCellSequencing"
        self.brief_description = "A series of methods for Single-cell RNA Sequencing data analysis."
        self.detailed_description = ""
        self.cfg = cfg

    # Batch Integration API
    def batch_integration(self, adata, batch_key, method=["liger"]): 
        # ["harmony", "liger", "scvi"]
        
        def run_harmony(adata, batch):
            import scanpy.external as sce
            if 'X_pca' not in adata.obsm:
                sc.tl.pca(adata, svd_solver='arpack')
            sce.pp.harmony_integrate(adata, key=batch)
            print("Harmony integration finished.")
            adata.obsm["X_harmony"] = adata.obsm["X_pca_harmony"]
            return adata

        def run_liger(adata, batch):
            import pyliger
            
            bdata = adata.copy()
            bdata.obs[batch] = bdata.obs[batch].astype('category')
            batch_cats = bdata.obs[batch].cat.categories

            # Ensure data matrix is sparse
            if isinstance(bdata.X, np.ndarray):
                bdata.X = scipy.sparse.csr_matrix(bdata.X)

            adata_list = [bdata[bdata.obs[batch] == b].copy() for b in batch_cats]
            for i, ad in enumerate(adata_list):
                ad.uns["sample_name"] = batch_cats[i]
                ad.uns["var_gene_idx"] = np.arange(bdata.n_vars)
            
            liger_data = pyliger.create_liger(adata_list, remove_missing=False, make_sparse=False)
            liger_data.var_genes = bdata.var_names
            pyliger.normalize(liger_data)
            pyliger.scale_not_center(liger_data)
            pyliger.optimize_ALS(liger_data, k=30, max_iters=30)
            pyliger.quantile_norm(liger_data)
            
            adata.obsm["X_pca_liger"] = np.zeros((adata.shape[0], liger_data.adata_list[0].obsm["H_norm"].shape[1]))
            for i, b in enumerate(batch_cats):
                adata.obsm["X_pca_liger"][adata.obs[batch] == b] = liger_data.adata_list[i].obsm["H_norm"]
            
            adata.obsm["X_liger"] = adata.obsm["X_pca_liger"]
            print("LIGER integration finished.")
            return adata

        def run_scvi(adata, batch):
            import scvi
            if 'counts' not in adata.layers:
                # Get the indices of the current adata.var_names in raw.var_names
                raw_index = [np.where(adata.raw.var_names == v)[0][0] for v in adata.var_names]
                adata.layers['counts'] = adata.raw.X[:, raw_index]

            scvi.model.SCVI.setup_anndata(adata, layer="counts", batch_key=batch)
            vae = scvi.model.SCVI(adata, gene_likelihood="nb", n_layers=2, n_latent=30)
            vae.train()
            adata.obsm["X_pca_scVI"] = vae.get_latent_representation()
            adata.obsm["X_scVI"] = adata.obsm["X_pca_scVI"]

        if 'X_pca' not in adata.obsm:
            sc.tl.pca(adata, svd_solver='arpack')

        # Robustly handle the 'method' input parameter
        if isinstance(method, str):
            method = [method]
        elif not isinstance(method, (list, tuple)):
            raise ValueError("method must be str or list/tuple of str")

        # Run the required methods sequentially
        for m in method:
            m_lower = m.lower()
            if m_lower == "harmony":
                adata = run_harmony(adata, batch_key)
            elif m_lower == "liger":
                adata = run_liger(adata, batch_key)
            elif m_lower == "scvi":
                adata = run_scvi(adata, batch_key)
            else:
                print(f"Unknown method: {m}, skipped.")

        return adata

    # Cell Type Annotation API
    def celltype_annotation(self, adata, species='', tissue_type='', cancer_type='Normal', obs_cluster='', method=["gpt4"], openai_api_key=None): 
        # Can be a single string or a list, e.g., ["gpt4", "cellmarker", "act"]
        
        def gpt4_method(adata, species, tissue_type, cancer_type, obs_cluster, openai_api_key):
            if "rank_genes" not in adata.uns.keys():
                sc.tl.rank_genes_groups(adata, obs_cluster, method='wilcoxon', key_added='rank_genes')
            
            result = adata.uns['rank_genes']
            groups = result['names'].dtype.names
            dat = pd.DataFrame({group: result['names'][group] for group in groups})
            df_first_10_rows = dat.head(10)
            rows_as_strings = df_first_10_rows.T.apply(lambda x: ','.join(x.dropna().astype(str)), axis=1)
            gene_list = '\n'.join([f"{i + 1}.{row}" for i, row in enumerate(rows_as_strings)])
            
            parts = []
            if tissue_type:
                parts.append(f"of {tissue_type} cells")
            
            meta = []
            if species:
                meta.append(f"species: {species}")
            if cancer_type:
                meta.append(f"cancer type: {cancer_type}")
            meta_str = f" ({', '.join(meta)})" if meta else ""
            
            prompt = (
                f"Markers for each cluster ({obs_cluster}):\n"
                f"{gene_list}\n\n"
                f"Identify each cluster cell types {' '.join(parts)}{meta_str} using these markers separately for each row. "
                "Only return the cell type name. Do not show numbers before the cell types name. Some can be a mixture of multiple cell types.\n"
            )
            
            if openai_api_key is not None:
                client = openai.OpenAI(api_key=openai_api_key)
            else:
                client = openai.OpenAI()
            
            completion = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ]
            )
            
            lines = completion.choices[0].message.content.split('\n')
            names = [re.sub(r'^\s*\d+\s*[\.\-:]?\s*', '', line).strip() for line in lines if line.strip()]
            n_cluster = len(groups)
            cell_types = (names + [names[-1]] * n_cluster)[:n_cluster] if names else ["Unknown"] * n_cluster
            cluster2type = dict(zip(groups, cell_types))
            adata.obs['gpt4_predict_type'] = adata.obs[obs_cluster].map(cluster2type).astype('category')
            return adata

        def cellmarker_method(adata, species, tissue_type, cancer_type, obs_cluster):
            
            marker = pd.read_excel("/root/CellAgent/scOmni/maker_database/Cell_marker_Seq.xlsx")
            filtered_df = marker[
                (marker['species'] == species) &
                (marker['tissue_type'] == tissue_type) &
                (marker['cancer_type'] == cancer_type)
            ]
            result_dict = {}
            for cell_name in filtered_df['cell_name'].unique():
                cell_df = filtered_df[filtered_df['cell_name'] == cell_name]
                markers = cell_df['marker'].tolist()
                result_dict[cell_name] = markers

            if "rank_genes" not in adata.uns.keys():
                sc.tl.rank_genes_groups(adata, obs_cluster, method='wilcoxon', key_added='rank_genes')

            cell_annotation_norm = sc.tl.marker_gene_overlap(
                adata, result_dict,
                key='rank_genes', 
                normalize='reference',
                adj_pval_threshold=0.05
            )
            max_indexes = cell_annotation_norm.idxmax()
            final = max_indexes.to_dict()
            adata.obs['CellMarker_predict_type'] = adata.obs[obs_cluster].map(final).astype('category')
            return adata

        def act_method(adata, species, tissue_type, obs_cluster):
            marker = pd.read_csv("/root/CellAgent/scOmni/maker_database/ACT.csv")
            filtered_df = marker[
                (marker['Species'] == species) &
                (marker['Tissue'] == tissue_type)
            ]
            result_dict = {}
            for cell_name in filtered_df['CellType'].unique():
                cell_df = filtered_df[filtered_df['CellType'] == cell_name]
                markers = cell_df['Marker'].tolist()
                result_dict[cell_name] = markers
            
            if "rank_genes" not in adata.uns.keys():
                sc.tl.rank_genes_groups(adata, obs_cluster, method='wilcoxon', key_added='rank_genes')
            
            cell_annotation_norm = sc.tl.marker_gene_overlap(
                adata, result_dict,
                key='rank_genes', 
                normalize='reference',
                adj_pval_threshold=0.05
            )
            max_indexes = cell_annotation_norm.idxmax()
            final = max_indexes.to_dict()
            adata.obs['ACT_predict_type'] = adata.obs[obs_cluster].map(final).astype('category')
            return adata

        if isinstance(method, str):
            methods = [method]
        else:
            methods = method

        for m in methods:
            m_lower = m.lower()
            if m_lower == "gpt4":
                adata = gpt4_method(adata, species, tissue_type, cancer_type, obs_cluster, openai_api_key)
            elif m_lower == "cellmarker":
                adata = cellmarker_method(adata, species, tissue_type, cancer_type, obs_cluster)
            elif m_lower == "act":
                adata = act_method(adata, species, tissue_type, obs_cluster)
            else:
                print(f"Unknown annotation method: {m}, skipped.")
        return adata
    
    # Trajectory Inference: Get Top K Methods
    def trajectory_top_k_methods(
        self,
        adata,
        time_limit='20m',
        memory_limit='25GB',
        fixed_n_methods='3',
        expected_topology=None,
        start_id=None,
        end_id=None,
        groups_id=None,
        PCA=None,
        X_umap=None
    ):
        # Initialization
        ti_output_path = "./temp/trajectory_inference"
        if not os.path.exists(ti_output_path):
            os.makedirs(ti_output_path, exist_ok=True)
            if 'counts' not in adata.layers:
                # Find indices of current cells in raw.obs_names
                cell_indices = [adata.raw.obs_names.get_loc(x) for x in adata.obs_names]
                gene_indices = [adata.raw.var_names.get_loc(v) for v in adata.var_names]
                adata.layers['counts'] = adata.raw.X[cell_indices, :][:, gene_indices]

            # Ensure adata.X and adata.layers['counts'] are csc_matrix for R compatibility
            if not isinstance(adata.X, np.ndarray):
                if scipy.sparse.isspmatrix_csr(adata.X):
                    adata.X = adata.X.tocsc()
                elif scipy.sparse.isspmatrix(adata.X) and not scipy.sparse.isspmatrix_csc(adata.X):
                    adata.X = adata.X.tocsc()
            if 'counts' in adata.layers:
                if scipy.sparse.isspmatrix_csr(adata.layers['counts']):
                    adata.layers['counts'] = adata.layers['counts'].tocsc()
                elif scipy.sparse.isspmatrix(adata.layers['counts']) and not scipy.sparse.isspmatrix_csc(adata.layers['counts']):
                    adata.layers['counts'] = adata.layers['counts'].tocsc()

            # Save h5ad for R script access
            sc.write(os.path.join(ti_output_path, 'adata.h5ad'), adata)

        # Automatically assemble command and call R script to return method list
        
        # Basic command list
        command = [
            "conda", "run", "-p", "/opt/conda/envs/DYNO_R",
            "Rscript", "/root/CellAgent/scOmni/TI_R/Ti_method.R"
        ]
        # Dynamically build command list based on parameter values
        params = [
            ("time_limit", time_limit),
            ("memory_limit", memory_limit),
            ("fixed_n_methods", fixed_n_methods),
            ("expected_topology", expected_topology),
            ("start_id", start_id),
            ("end_id", end_id),
            ("groups_id", groups_id),
            ("dimred", PCA),
            ("X_umap", X_umap),
            ("adata_path", ti_output_path),
        ]

        for param, value in params:
            if value is not None:
                command.append(f"{param}={value}")

        # Print the command to be executed
        print("Executing command:", " ".join(command))

        try:
            result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            # Use regex to find all content within double quotes
            print("STDOUT:")
            print(result.stdout)
            methods = re.findall(r'"([^"]+)"', re.search(r'\[\d+\]\s+("[^"]+"(\s+)*)+', result.stdout).group()) if re.search(r'\[\d+\]\s+("[^"]+"(\s+)*)+', result.stdout) else print("No match found")
            
            print("Methods found:", methods)
            return methods
        except subprocess.CalledProcessError as e:
            print("ERROR:")
            print(e.stderr)
            raise RuntimeError(f"Command '{command}' failed with error:\n{e.stderr}") from e 

    # Trajectory Inference API (single step execution)
    def trajectory_inference(self, adata, obs_cluster=None, obsm_embedding=None, start_id=None, expected_topology=None, fixed_n_methods=3):
        
        def run_method(method_name=None):
            import subprocess
            # Basic command list
            command = ["conda", "run", "-p", "/opt/conda/envs/DYNO_R", "Rscript", "/root/CellAgent/scOmni/TI_R/Ti_Run.R"]
            # Dynamically build command list
            if method_name is not None:
                command.append(f"Methods_selected={method_name}")
            ti_output_path = "./temp/trajectory_inference"
            command.append(f"adata_path={ti_output_path}")

            # Execute the R script and capture output
            print("Executing command:", " ".join(command))
            try:
                result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                print("STDOUT:")
                print(result.stdout)
                return result
            except subprocess.CalledProcessError as e:
                print("ERROR:")
                print(e.stderr)
                raise RuntimeError(f"Command '{command}' failed with error:\n{e.stderr}") from e
        
        # Get the recommended methods first
        method_list = self.trajectory_top_k_methods(
            adata,
            start_id=start_id, 
            expected_topology=expected_topology, 
            fixed_n_methods=fixed_n_methods,
            X_umap=obsm_embedding,
            groups_id=obs_cluster
        )

        # Run each recommended method
        for method in method_list:
            try:
                run_method(method_name=method)
            except Exception as e:
                print(f"Error running method {method}: {e}")


SinglecellSequencing_toolkit = SinglecellSequencing_toolkit()