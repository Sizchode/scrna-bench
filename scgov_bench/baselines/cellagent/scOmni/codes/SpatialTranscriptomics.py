# Spatial Transcriptomics API
# import stlearn as st (stlearn is not included in the main environment with squidpy due to conflicts caused by stlearn's dependency on TensorFlow)
import scanpy as sc
try:
    import squidpy as sq
except ImportError as e:
    print(f"Exception: Error importing squidpy: {e}")
import subprocess
import os
import numpy as np
import random
import scanpy as sc
from .tool_base import MultiToolBase
import pandas as pd

def save_filtered_raw_data(adata, train_genes, predict_genes, value_genes, filename):
    """
    Save filtered raw data to a CSV file.
    Parameters:
    adata (AnnData): AnnData object containing the raw data.
    train_genes (list): List of genes used for training.
    predict_genes (list): List of genes to be predicted.
    value_genes (list): List of genes used for validation.
    filename (str): The file path where the filtered data will be saved.

    Returns:
    None: This function saves the filtered data to a CSV file.
    """
    try:
        # Extract raw data and convert to DataFrame
        raw_data = pd.DataFrame(adata.X, columns=adata.var_names, index=adata.obs_names)
        # Filter genes
        genes = train_genes + predict_genes + value_genes
        filtered_data = raw_data[genes].copy()
        filtered_data.to_csv(filename, sep=',', index=True, header=True)
        print(f"Filtered raw data saved to {filename}")
    except Exception as e: 
        raise ValueError("No raw data available in adata.")
        
def save_filtered_raw_data_st(adata, train_genes, predict_genes,value_genes,filename):
    """
    Save filtered raw data for spatial transcriptomics to a CSV file.

    Parameters:
    adata (AnnData): AnnData object containing the raw data.
    train_genes (list): List of genes used for training.
    predict_genes (list): List of genes to be predicted.
    value_genes (list): List of genes used for validation.
    filename (str): The file path where the filtered data will be saved.

    Returns:
    None: This function saves the filtered data to a CSV file.
    """
    # Check if adata has raw attribute
    try:
        # Extract raw data and convert to DataFrame
        raw_data = pd.DataFrame(adata.X, columns=adata.var_names, index=adata.obs_names)
        # Filter genes
        genes = train_genes + value_genes
        filtered_data = raw_data[genes].copy()
        # Set columns corresponding to predict_genes to -1
        for gene in predict_genes:
            filtered_data.loc[:, gene] = -1.0
        filtered_data.to_csv(filename, sep=',', index=True, header=True)
        print(f"Filtered raw data saved to {filename}")
    except Exception as e:
        raise ValueError("No raw data available in adata.")
            
def save_spatial_coordinates_as_txt(adata, filename):
    """
    Save the spatial coordinates from adata.obsm['spatial'] into a txt file.
    The first row will contain column headers 'x' and 'y'.

    Parameters:
    - adata: AnnData object containing spatial coordinates in adata.obsm['spatial']
    - filename: The file path where the spatial coordinates will be saved

    Returns:
    None: This function saves the spatial coordinates to a text file.
    """
    # Check if spatial coordinates exist in adata.obsm
    if 'spatial' not in adata.obsm:
        print("Error: Spatial data not found in adata.obsm['spatial']")
        return

    # Extract the spatial coordinates from adata.obsm['spatial']
    spatial_coords = adata.obsm['spatial']

    # Convert to a DataFrame with 'x' and 'y' as column names
    spatial_df = pd.DataFrame(spatial_coords, columns=['x', 'y'])

    # Save the DataFrame to a tab-separated text file (.txt)
    spatial_df.to_csv(filename, sep='\t', index=False, header=True)
    print(f"Spatial coordinates saved to {filename}")

def to_adata_raw(adata):
    """
    Creates a new AnnData object from adata.raw if available; otherwise uses adata.X.
    Keeps the 'spatial' data in adata.obsm if present.
    """
    import pandas as pd
    import scanpy as sc

    # Decide which data (raw or X) to use
    if adata.raw is not None:
        data_source = adata.raw.X
        var_names = adata.raw.var_names
        obs_names = adata.raw.obs_names
    else:
        data_source = adata.X
        var_names = adata.var_names
        obs_names = adata.obs_names

    # Convert sparse matrix (if present) to a dense array
    if hasattr(data_source, "toarray"):
        data_source = data_source.toarray()

    # Build pandas DataFrame and then create the AnnData object
    raw_data = pd.DataFrame(data_source, columns=var_names, index=obs_names)
    adata_raw = sc.AnnData(raw_data)

    # Retain 'spatial' data from the original AnnData's .obsm
    if "spatial" in adata.obsm:
        adata_raw.obsm["spatial"] = adata.obsm["spatial"]

    return adata_raw

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

class SpatialTranscriptomics_toolkit(MultiToolBase):
    def __init__(self):
        """Initialize the Spatial Transcriptomics Toolkit."""
        super().__init__()
        self.tool_name = "SpatialTranscriptomics"
        self.brief_description = "A series of methods for Spatial Transcriptomics analysis."
        self.detailed_description = ""
        self.temp_path = "./temp" 
        self.first_impute_method_run = True 
        
    # Spatial Domain Identification (Spatial Clustering) API
    def spatial_domain_identify(self, adata_st, n_domains=None, methods=None):
        """
        Identify spatial domains in spatial transcriptomics data using specified methods.

        This function takes an AnnData object containing spatial transcriptomics data
        and applies one or more domain identification methods to it. It also saves
        relevant visualizations and intermediate results.

        Parameters:
        adata_st (anndata.AnnData): The AnnData object containing spatial transcriptomics data.
                                 It should have the necessary information such as the spatial coordinates
                                 and gene expression values.
        n_domains (int, optional): The number of domains to identify. If not provided, it will be
                                 determined by the individual methods. Defaults to None.
        methods (list of str, optional): A list of methods to use for spatial domain identification.
                                         Supported methods include 'DeepST', 'stLearn', 'SEDR'.
                                         If not provided, the default list ['DeepST', 'stLearn', 'SEDR'] will be used.
                                         Defaults to None.

        Returns:
                None: This function does not return a value directly. However, it saves the following:
                    - A histopathological plot showing the spatial data to './tmp/spatial_domains/histopathological_plot'.
            Example:
                >>> # specify the number of domains and methods to use
                >>> # here assume the user specifies n_domains as 7, but if usr do not specify the number of domains, n_domains should be None !
                >>> spatial_domain_identify(adata, n_domains=7, methods=['DeepST', 'stLearn', 'SEDR']) # specify the number of domains and methods to use 
                >>> # Note: 
                >>> try:
                ...     adata_DeepST = sc.read('./tmp/spatial_domains/DeepST_results/adata.h5ad')
                ...     sc.pl.spatial(adata_DeepST, color='DeepST_clusters', title='Spatial Domains by DeepST')
                ... except Exception as e:
                ...     print(f"An error occurred: {e}")
                >>> try:
                ...     adata_stLearn = sc.read('./tmp/spatial_domains/stLearn_results/adata.h5ad')
                ...     sc.pl.spatial(adata_stLearn, color='stLearn_clusters', title='Spatial Domains by stLearn')
                ... except Exception as e:
                ...     print(f"An error occurred: {e}")
                >>> try:
                ...     adata_SEDR = sc.read('./tmp/spatial_domains/SEDR_results/adata.h5ad')
                ...     sc.pl.spatial(adata_SEDR, color='SEDR_clusters', title='Spatial Domains by SEDR')
                ... except Exception as e:
                ...     print(f"An error occurred: {e}")
        """
        adata_st.write('./adata_spatial_domain.h5ad')
        import matplotlib.pyplot as plt
        import os 
        # Create folder if it doesn't exist
        os.makedirs('./tmp/spatial_domains', exist_ok=True)
        sc.pl.spatial(adata_st, img_key="hires",show=False)
        plt.savefig('./tmp/spatial_domains/histopathological_plot', bbox_inches='tight')
        plt.close() 
        if methods is None:
            methods = ['DeepST', 'stLearn', 'SEDR']
            
        import subprocess
        python_interpreters = {
            'DeepST': '/data/cellana/anaconda3/envs/deepst_env/bin/python',
            'stLearn': '/home/cellana/anaconda3/envs/cellana/envs/stlearn/bin/python',
            'SEDR': '/home/cellana/anaconda3/envs/SEDR/bin/python',
        }
        # Paths remain hardcoded as they refer to the execution environment
        script_paths = {
            'DeepST': '/data/cellana/single_cell_spatial_analysis/spatial_scripts/DeepST_cluster.py',
            'stLearn': '/data/cellana/single_cell_spatial_analysis/spatial_scripts/stLearn_cluster.py',
            'SEDR': '/data/cellana/single_cell_spatial_analysis/spatial_scripts/SEDR_cluster.py',
        }

        for method in methods:
            if method not in python_interpreters or method not in script_paths:
                print(f"Method {method} is not recognized.")
                continue
            try:
                # Construct the command
                command = [python_interpreters[method], script_paths[method]]
                command.append(str(n_domains))
                # Execute the script using subprocess
                result = subprocess.run(
                    command,
                    text=True, 
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=True,
                )

                # print(result.stdout.strip())

            except subprocess.CalledProcessError as e:
                print(f"Error running the script for method {method}:\n{e.stderr}")
            except FileNotFoundError as e:
                print(f"File not found: {e}")
            except Exception as e:
                print(f"An unexpected error occurred: {e}")

    def spatial_domain_plot(self, method=None):
        """
        Function to display the image visualization results corresponding to specific spatial domain methods.

        Parameters:
        method (str or None): Specifies the name of the method for which the image is to be displayed. Valid options are 'DeepST', 'stLearn', 'SEDR'. If it is None, the function will iterate through and display the images for all available methods.
        """
        import os
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        plots_paths = {
            'DeepST': './tmp/spatial_domains/DeepST_results/DeepST_refine_plot.png',
            'stLearn': './tmp/spatial_domains/stLearn_results/stLearn_plot.png',
            'SEDR': './tmp/spatial_domains/SEDR_results/SEDR_leiden_plot.png',
        }

        # If method is None, iterate through all methods
        if method is None:
            methods = plots_paths.keys()
        else:
            methods = [method] if method in plots_paths else []

        # Iterate and display images, skipping if file does not exist
        for method in methods:
            img_path = plots_paths[method]
            if os.path.exists(img_path):
                img = mpimg.imread(img_path)
                plt.imshow(img)
                # plt.title(f'{method} Visualization')
                plt.axis('off')
                plt.show()
            else:
                print(f"Image for {method} does not exist, skipping.")

    # Spatial Trajectory API
    def stLearn_spatial_trajectory(self, adata, use_label="louvain_morphology", cluster=0):
        """
        Perform spatial trajectory analysis using stLearn.
        
        Parameters:
        adata (AnnData): The input data.
        use_label (str): The label to use for clustering. Defaults to "louvain_morphology".
        cluster (int): The cluster to set as root for trajectory analysis. Defaults to 0.
        
        Returns:
        dict: A dictionary of available paths for trajectory analysis.

        Example:
        >>> stLearn_spatial_trajectory(adata, use_label="louvain", cluster=0) 
        >>> # The function will automatically calculate the trajectory and plot the trajectory, so there is no need to generate code to plot further.
        """
        
        import subprocess
        import os 
        # Create temp folder for spatial trajectory if it doesn't exist
        temp_path = "./tmp/spatial_trajectory" 
        os.makedirs(temp_path, exist_ok=True)
        adata.write(f'{temp_path}/adata.h5ad')

        # Create TI_plot folder if it doesn't exist
        os.makedirs("./tmp/spatial_trajectory/TI_plot" , exist_ok=True)
        ti_plot_path = "./tmp/spatial_trajectory/TI_plot"

        # Specify the virtual environment's Python interpreter
        python_interpreter = "/home/cellana/anaconda3/envs/cellana/envs/stlearn/bin/python"
        # Path remains hardcoded as it refers to the execution environment
        script_path = '/data/cellana/single_cell_spatial_analysis/spatial_scripts/stLearn_spatial_trajectory.py'
        
        # Call the script using subprocess
        subprocess.run([
            python_interpreter, 
            script_path, 
            temp_path, 
            use_label, 
            str(cluster),
            ti_plot_path,
        ])
        # Display images in the TI_plot folder
        import matplotlib.pyplot as plt
        from IPython.display import Image, display
        
        for img_file in os.listdir(ti_plot_path):
            if img_file.endswith(".png"):
                display(Image(filename=os.path.join(ti_plot_path, img_file)))

    # Squidpy API - Interaction Matrix
    def Compute_interaction_matrix(self, adata, cluster_key):
        """
        Compute and visualize spatial neighborhood richness analysis for clusters.

        Parameters
        ----------
        adata : AnnData
            Annotated data object containing single-cell data.
        cluster_key : str
            Key in `adata.obs` where clustering information is stored.

        details of this function:
        def Compute_interaction_matrixCompute_interaction_matrix(self, adata, cluster_key):
            sq.gr.spatial_neighbors(adata)
            sq.gr.interaction_matrix(adata, cluster_key=cluster_key)
            sq.pl.interaction_matrix(adata, cluster_key=cluster_key)
        -------
        None
            Modifies the `adata` object to include the interaction matrix,
            stored in `adata.uns['{cluster_key}_interactions']`,
            and generates a plot of the interaction matrix.

        Example:
        >>> # The function will automatically calculate interaction_matrix and plot the image, so there is no need to generate code further.
        >>> Compute_interaction_matrix(adata, cluster_key="louvain") 
        """
        # Compute spatial neighbors
        sq.gr.spatial_neighbors(adata)
        # Compute interaction matrix for clusters
        sq.gr.interaction_matrix(adata, cluster_key=cluster_key)
        # Plot the interaction matrix
        sq.pl.interaction_matrix(adata, cluster_key=cluster_key)

    def Compute_co_occurrence_probability(self, adata, cluster_key, chosen_clusters):
        """
        Compute the co-occurrence probability for specified clusters and visualize the results.

        Parameters
        ----------
        adata : AnnData
            Annotated data object containing single-cell data and spatial information.
        cluster_key : str
            Key in `adata.obs` where clustering information is stored.
        chosen_clusters : str | Sequence[str]
            Specific cluster instances to plot, can be a single cluster or a list of multiple clusters.

        details of this function:
        def Compute_co_occurrence_probability(self, adata, cluster_key, chosen_clusters):
            sq.gr.co_occurrence(adata, cluster_key=cluster_key)
            sq.pl.co_occurrence(adata, cluster_key=cluster_key, clusters=chosen_clusters)
            sq.pl.spatial_scatter(adata, color=cluster_key, shape=None)

        Returns
        -------
        None
            This function directly displays the results in the plotting interface and does not return any value.
        """
        sq.gr.co_occurrence(adata, cluster_key=cluster_key)
        sq.pl.co_occurrence(adata, cluster_key=cluster_key, clusters=chosen_clusters)
        sq.pl.spatial_scatter(adata, color=cluster_key, shape=None)

    def receptor_ligand_analysis(self, adata, cluster_key, source_groups, n_perms=1000, threshold=0, alpha=0.005):
        """
        Perform receptor-ligand analysis on the given AnnData object.
        
        Parameters:
        - adata: AnnData object containing the spatial transcriptomics data.
        - cluster_key: Key to identify the clusters in the data.
        - source_groups: Groups to visualize in the ligand-receptor analysis.
        - n_perms: Number of permutations for significance testing (default is 1000).
        - threshold: Minimum significance threshold for receptor-ligand interactions (default is 0).
        - alpha: Significance level for the analysis (default is 0.005).
        
        Returns:
        - res: Results of the receptor-ligand analysis as a new AnnData object.

        details of this function:
        def receptor_ligand_analysis(self, adata, cluster_key, source_groups, n_perms=1000, threshold=0, alpha=0.005):
            res = sq.gr.ligrec(
            adata,
            n_perms=n_perms, 
            cluster_key=cluster_key, 
            copy=True, 
            use_raw=True, 
            transmitter_params={"categories": "ligand"}, 
            receiver_params={"categories": "receptor"}, 
            threshold=threshold, 
            )
            sq.pl.ligrec(res, source_groups=source_groups, alpha=alpha)
            return res
        
        """
        # Perform ligand-receptor analysis on the data
        res = sq.gr.ligrec(
            adata,
            n_perms=n_perms,
            cluster_key=cluster_key,
            copy=True,
            use_raw=True,
            transmitter_params={"categories": "ligand"},
            receiver_params={"categories": "receptor"},
            threshold=threshold,
        )
        # Visualize the results of the receptor-ligand analysis
        sq.pl.ligrec(res, source_groups=source_groups, alpha=alpha)
        
        return res

    def analyze_spatial_autocorr(self, adata):
        """
        Analyze spatial autocorrelation using Moran's I score and identify the most spatially correlated genes.
        
        Parameters:
        - adata: AnnData object containing the spatial transcriptomics data.
                
        Returns:
        -  Moran's I score will save in adata.uns["moranI"].

        This function will compute the Moran's I score to identify genes with strong spatial autocorrelation,
        similar to how spatially variable genes are identified.

        details of this function:
        def analyze_spatial_autocorr(self, adata):
            ...
            sq.gr.spatial_autocorr(
                adata,
                mode="moran", 
                genes=genes, 
                n_perms=100, 
                n_jobs=4, 
            )
            adata.uns["moranI"].head(10)
            print('top 10 genes with the highest Moran\'s I score\n',adata.uns["moranI"].head(10))

        Example:
        >>> ...
        >>> SpatialTranscriptomics.analyze_spatial_autocorr(adata_st)
        >>> top_genes = adata_st.uns["moranI"].head(10).index
        >>> sq.pl.spatial_scatter(adata_st, color=top_genes[:3])

                
        """
        if 'highly_variable' not in adata.var.columns:
            sc.pp.highly_variable_genes(adata, n_top_genes=2000)
        # Select the top 100 highly variable genes
        genes = adata[:, adata.var.highly_variable].var_names.values[:100]
        # Compute spatial neighbors for the AnnData object
        sq.gr.spatial_neighbors(adata)
        # Calculate Moran's I score for the selected genes
        sq.gr.spatial_autocorr(
            adata,
            mode="moran",
            genes=genes,
            n_perms=100,
            n_jobs=4,
        )
        # Display the top 10 genes with the highest Moran's I scores
        adata.uns["moranI"].head(10)
        print('top 10 genes with the highest Moran\'s I score\n',adata.uns["moranI"].head(10))
        # Visualize the spatial distribution of the top genes based on Moran's I score
        #top_genes = adata.uns["moranI"].head(10).index
        #print('adata.uns["moranI"].head(10).index\n',top_genes)
        #sq.pl.spatial_scatter(adata, color=top_genes[:3]) 

    def compute_ripley(self, adata, cluster_key, mode="L"):
        """
        Compute Ripley's statistics to analyze the spatial distribution patterns of cells.
        
        Parameters:
        - adata: AnnData object containing the spatial transcriptomics data.
        - cluster_key: Key in `adata.obs` that indicates the clustering of cells.
        - mode: The mode of Ripley's statistic to compute. Default is "L", which is commonly used.
        
        This function calculates Ripley's K-function or L-function to assess the spatial distribution
        of cells and determine whether they exhibit randomness, clustering, or regularity.

        details of this function:
        def compute_ripley(self, adata, cluster_key, mode="L"):
            sq.gr.ripley(adata, cluster_key=cluster_key, mode=mode)
            sq.pl.ripley(adata, cluster_key=cluster_key, mode=mode)
        """
        # Calculate Ripley's statistics for the specified clustering
        sq.gr.ripley(adata, cluster_key=cluster_key, mode=mode)
        
        # Visualize the results of Ripley's statistics
        sq.pl.ripley(adata, cluster_key=cluster_key, mode=mode)

    # Gene Imputation API
    def impute_method_run(self, adata_sc,adata_st,train_genes=None,predict_genes=None,value_genes=None,methods=['Tangram', 'gimVI', 'SpaGE']):
        """
        Run imputation methods on the given single-cell and spatial transcriptomics data.

        Parameters:
        adata_sc (AnnData): AnnData object containing single-cell data.
        adata_st (AnnData): AnnData object containing spatial transcriptomics data.
        train_genes (list, optional): List of genes used for training. Defaults to None.
        predict_genes (list, optional): List of genes to be predicted. Defaults to None.
        value_genes (list, optional): List of genes used for validation. Defaults to None.
        methods (list of str, optional): List of imputation methods to use. Defaults to ['Tangram', 'gimVI', 'SpaGE']. 

        Returns:
        None: This function runs the imputation methods and saves the results.
        Example:
        >>> methods=['Tangram','gimVI']
        >>> # specify the methods to run , if ues require these methods, otherwise the default methods should be run.
        >>> # if train_genes and predict_genes are provided, use them, otherwise, the function will automatically select genes.
        >>> SpatialTranscriptomics.impute_method_run(adata_sc,adata_st,methods=methods) 
        >>> # Attempt to visualize prediction results
        >>> try: 
                # if Tangram imputation is available
                adata_Tangram = SpatialTranscriptomics.get_imputed_anndata(adata_st=adata_st,method='Tangram')
                genes = list(adata_Tangram.var_names)[:3]
                # Visualize these three genes
                for gene in genes:
                    sc.pl.spatial(adata_Tangram, color=gene, title=f'Expression of {gene}', show=True)
                    # if user specifies spot_size = 50 , use sc.pl.spatial(adata_Tangram, color=gene, title=f'Expression of {gene}', spot_size = 50,show=True)
            except Exception as e:
                print(f"An error occurred: {e}") 
        
                try:
                    adata_gimVI = SpatialTranscriptomics.get_imputed_anndata(adata_st=adata_st,method='gimVI')
                    genes = list(adata_gimVI.var_names)[:3]
                    for gene in genes:
                        sc.pl.spatial(adata_gimVI, color=gene, title=f'Expression of {gene}', show=True)
            except Exception as e:
                print(f"An error occurred: {e}")
            ...
        """
        adata_sc = to_adata_raw(adata_sc)
        adata_st = to_adata_raw(adata_st) 

        # global first_impute_method_run
        if self.first_impute_method_run:
            # Mark as False to ensure subsequent calls don't re-run initialization
            self.first_impute_method_run = False
            if train_genes is None:
                ensure_highly_variable_genes(adata_st)
                ensure_highly_variable_genes(adata_sc)

                # Get highly variable genes for adata_st and adata_sc
                highly_variable_genes_st = set(adata_st.var[adata_st.var['highly_variable']].index)
                highly_variable_genes_sc = set(adata_sc.var[adata_sc.var['highly_variable']].index)

                # Find intersection
                highly_variable_genes_intersection = highly_variable_genes_st & highly_variable_genes_sc

                # Get intersection of raw data genes
                raw_genes_intersection = set(adata_sc.var_names) & set(adata_st.var_names)

                # Combine highly variable gene intersection and raw data gene intersection
                combined_genes = highly_variable_genes_intersection | raw_genes_intersection

                # If combined genes exceed 1000, prioritize highly variable genes
                if len(combined_genes) > 1000:
                    combined_genes = list(highly_variable_genes_intersection)[:1000] + list(raw_genes_intersection - highly_variable_genes_intersection)[:1000-len(highly_variable_genes_intersection)]

                # Final selected 1000 genes
                train_genes = list(combined_genes)[:1000]

            if predict_genes is None:
                predict_genes = list(set(adata_sc.var_names)-set(adata_st.var_names))[:10]

            if value_genes is None:
                value_genes_ = list(set(adata_sc.var_names) & set(adata_st.var_names) - set(train_genes))
                value_genes_len = int(len(predict_genes)*0.2)+1 
                value_genes = value_genes_[:value_genes_len]

            input_dir='./tmp/imputation'
            if not os.path.exists(input_dir):
                os.makedirs(input_dir)

            save_filtered_raw_data(adata_sc,train_genes, predict_genes,value_genes, input_dir + '/Rna_data.csv')
            save_filtered_raw_data_st(adata_st,train_genes, predict_genes,value_genes,input_dir + '/Spatial_data.csv')
            save_spatial_coordinates_as_txt(adata_st, input_dir+'/Locations.txt')
            np.save(input_dir + '/train_list.npy', np.array(train_genes))
            np.save(input_dir + '/test_list.npy', np.array(predict_genes+value_genes))
            np.save(input_dir + '/value_list.npy', np.array(value_genes))
            
        methods = str(methods)

        import subprocess
        python_interpreter = "/home/cellana/anaconda3/envs/Benchmarking/bin/python"
        # Path remains hardcoded as it refers to the execution environment
        script_path = "/data/cellana/single_cell_spatial_analysis/spatial_scripts/gene_imputation_run_script-Copy1.py"
        try:
            # Construct the command
            command = [python_interpreter, script_path, methods]
            
            # Execute the script using subprocess
            result = subprocess.run(
                command,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )

        except subprocess.CalledProcessError as e:
            print(f"Error running the script:\n{e.stderr}")
            raise
        except FileNotFoundError as e:
            print(f"File not found: {e}")
            raise
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            raise

    def get_imputed_anndata(self, adata_st,method):
        """
        Get the imputed AnnData object for the specified method.

        Parameters:
        adata_st (AnnData): AnnData object containing spatial transcriptomics data.
        method (str): The imputation method to use.

        Returns:AnnData
            The imputed AnnData object. In this object:
            - `adata.X` contains the imputed expression matrix. where all genes are imputed (predicted) unknown genes.
        
        Example:
        >>> try: 
            # if Tangram imputation is available
            adata_Tangram = SpatialTranscriptomics.get_imputed_anndata(adata_st=adata_st,method='Tangram')
            genes = adata_Tangram.var_names
            random_genes = random.sample(list(genes), 3)
            # Visualize these three genes
            for gene in random_genes:
                sc.pl.spatial(adata_Tangram, color=gene, title=f'Expression of {gene}', show=True)
                except Exception as e:
                print(f"An error occurred: {e}") 
        """
        impute_genes_paths = {
            'Tangram': './tmp/imputation/Tangram_impute.csv',
            'gimVI': './tmp/imputation/gimVI_impute.csv',
            'SpaGE': './tmp/imputation/SpaGE_impute.csv',
            'stPlus': './tmp/imputation/stPlus_impute.csv',
            'novoSpaRc': './tmp/imputation/novoSpaRc_impute.csv',
        }
        
        if method in impute_genes_paths:
            file_path = impute_genes_paths[method]
            if os.path.exists(file_path):
                imputed_data = pd.read_csv(file_path,index_col=0)
                adata_imputation = sc.AnnData(imputed_data) 
                try:
                    # Attempt to copy spatial data
                    adata_imputation.uns['spatial'] = adata_st.uns['spatial'].copy()
                except Exception as e:
                    print(f"KeyError: 'spatial' not found in adata_st.uns: {e}")
                try:
                    adata_imputation.obsm['spatial'] = adata_st.obsm['spatial'].copy()
                except Exception as e:
                    print(f"KeyError: 'spatial' not found in adata_st.obsm: {e}")
            else:
                print(f"Imputed data file not found for method: {method}")
                raise FileNotFoundError(f"Imputed data file not found at {file_path}")
        else:
            print(f"Unknown method: {method}")
            raise ValueError(f"Unknown imputation method: {method}")
        print('imputed adata info:')
        # Print the imputation AnnData object summary
        print(adata_imputation)
        return adata_imputation