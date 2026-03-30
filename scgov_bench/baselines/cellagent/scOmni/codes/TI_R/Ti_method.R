library(dyno)
library(tidyverse)
library(anndata)

# Initialize the guide list and directly specify default values
guide <- list(
        adata_path='/home/cellana/Automation/test_framework/Frame_xyh/R_xyh',
        time_limit = '10m', 
        memory_limit = "24GB",
        fixed_n_methods = '3',
        expected_topology = NULL, 
        start_id = NULL,
        end_id = NULL,
        groups_id = NULL,  # In real datasets, 'grouping' is used
        dimred = NULL, # 'X_pca', # Use PCA components, a dimensionality reduction of the cells (not used here)
        X_umap = NULL,
       prior_information = c(
          # "groups_id"   # Not used in the full pipeline by default
          # "dimred"      # Not used in the full pipeline by default
          # start_id
          # end_id
                   )
       )

#print(guide)

# Retrieve command-line arguments
args <- commandArgs(trailingOnly = TRUE)

# Function: Parse command-line arguments and update the guide list
parse_args <- function(args, guide) {
  for (arg in args) {
  key_value <- strsplit(arg, "=")[[1]]
  if (length(key_value) == 2) {
    key <- key_value[1]
    value <- key_value[2]
    # Depending on the key, the value may need to be converted to a specific type
    # For simplicity, we assume all input values are strings
    guide[[key]] <- value
  }
  }
  return(guide)
}

########################################
# Call the function to parse command-line arguments and update the guide list
guide <- parse_args(args, guide)
print(str(guide))

#print(getwd())
#print(guide$adata_path)
#q()
setwd(guide$adata_path)
#print(getwd())
###############################################

################################
# Check guide parameters and set the working directory (file save path)
#print(str(guide))
#setwd(guide$adata_path)

adata <- read_h5ad("adata.h5ad")
#print(adata)
#q()
################################

# Basic information: count expression information
dataset <- wrap_expression(
  #counts = adata$X,
  counts = adata$layers['counts'],
  expression = adata$X
)

# Add prior information for start_id if provided
if (!is.null(guide$start_id)){
  guide$prior_information <- c(guide$prior_information, "start_id")
}

# Add prior information for end_id if provided
if (!is.null(guide$end_id)){
  guide$prior_information <- c(guide$prior_information, "end_id")
}

group_id_final = NULL
# Add prior information for groups_id if provided
if (!is.null(guide$groups_id)){
  guide$prior_information <- c(guide$prior_information, "groups_id")
  # Preprocessing for cell group information
  adata$obs['group_id'] = adata$obs[guide$groups_id]
  group_id_final <- rownames_to_column(adata$obs['group_id'], var = "cell_id")
}

# Add prior information to the dataset
dataset <- add_prior_information(
  dataset,
  groups_id = group_id_final,
  start_id = guide$start_id,
  end_id = guide$end_id,
)

# Add dimensionality reduction data if provided
if (!is.null(guide$dimred)){
  guide$prior_information <- c(guide$prior_information, "dimred")
  X_pca <- adata$obsm[guide$dimred][[1]] # A matrix: a × b of type dbl
  cell_names <- rownames(adata$obs)
  rownames(X_pca) <- cell_names
  dataset <- add_dimred(
  dataset,
  X_pca
)
}

X_umap = NULL
# Add UMAP information for visualization if provided
if (!is.null(guide$X_umap)){
  X_umap = as.data.frame(adata$obsm[guide$X_umap])
  cell_names <- rownames(adata$obs)
  X_umap <- cbind(cell_id = cell_names, X_umap)    
}
save(X_umap, file = "X_umap.RData")

# Retrieve dimension information
dims <- dim(adata)

# Reproduce the guidelines as created in the shiny app
answers <- dynguidelines::answer_questions(
  multiple_disconnected = FALSE, 
  n_cells = dims[1],
  n_features = dims[2],
  expect_topology = TRUE, 
  expected_topology = guide$expected_topology,  # angle, cycle, linear, convergence, bifurcation, tree
  time = guide$time_limit, 
  memory = guide$memory_limit, 
  prior_information = guide$prior_information,
  method_selection = "fixed_n_methods", 
  fixed_n_methods = as.integer(guide$fixed_n_methods), # Top_k TI methods
  docker = TRUE
)

guidelines <- dynguidelines::guidelines(answers = answers)
methods_selected <- guidelines$methods_selected

# Output TI methods and the dataset information passed to the TI methods
#####################################

#print('TI methods rank:')
print(methods_selected)
#print('Final dataset information:')
#print(str(dataset))

save(dataset, file = "dataset.RData")
q()
#####################################
