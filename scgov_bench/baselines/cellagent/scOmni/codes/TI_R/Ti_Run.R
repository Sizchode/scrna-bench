# Ensure the directory is created and has permissions
# Sys.setenv(TMPDIR  = "/path/to/tmp")  
# Sys.setenv(TMP  = "/path/to/tmp")
# Sys.setenv(TEMP  = "/path/to/tmp")
# unlink("/path/to/tmp", recursive = TRUE)  # Ensure the directory is cleared
# dir.create("/path/to/tmp", recursive = TRUE, mode = "0777")
# Use tempdir() function to get the current temporary directory
# temp_directory <- tempdir()

# Output the temporary directory
# print(temp_directory)

library(anndata)
library(dyno)
library(tidyverse)

# Initialize the guide list with default values
guide <- list(
        adata_path='./temp/trajectory_inference',
        Methods_selected=NULL
       )

#print(guide)

# Get command-line arguments
args <- commandArgs(trailingOnly = TRUE)
# Function: Parse command-line arguments and update the guide list
parse_args <- function(args, guide) {
  for (arg in args) {
  key_value <- strsplit(arg, "=")[[1]]
  if (length(key_value) == 2) {
    key <- key_value[1]
    value <- key_value[2]
    # Depending on the key, the value may need type conversion
    # For simplicity, we assume all values are strings
    guide[[key]] <- value
  }
  }
  return(guide)
}

########################################
# Call the function to parse arguments and update the guide list
guide <- parse_args(args, guide)
setwd(guide$adata_path)
load("X_umap.RData")
load("dataset.RData")

Methods_selected <- guide$Methods_selected
print('Methods_selected:')
print(Methods_selected)
# print('Final dataset information:')
# print(str(dataset))
#q()
#####################################

# Run trajectory inference algorithm
options(dynwrap_backend = 'container')
model <- infer_trajectory(dataset, Methods_selected)
print('infer_trajectory is done!')
# save(model, file = "model.RData")
save(model, file=paste0(Methods_selected, "_model.RData"))
#q()
# print('Start running visualization plotting function:')
#####################################

# Run plotting
Ti_dirmed_coler <- function(model, dataset, feature_name = NULL) {
  library(cowplot)
  library(ggplot2)

  # Ensure plot_dimred does not throw errors
  safe_plot_dimred <- function(..., default_title = "Plot Failed") {
  tryCatch({
    plot_dimred(...)
  }, error = function(e) {
    message("⚠️ Warning: plot_dimred failed - ", e$message)
    ggplot() + 
    annotate("text", x = 0, y = 0, label = default_title, size = 6, color = "red") +
    theme_void()
  })
  }

  plots <- list()

  # If dataset$prior_information$groups_id$group_id exists, plot P1
  if (!is.null(dataset$prior_information$groups_id$group_id)) {
  plots[[length(plots) + 1]] <- safe_plot_dimred(
    model, dimred = X_umap, grouping = dataset$prior_information$groups_id$group_id,
    default_title = "Cell grouping Failed"
  ) + ggtitle("Cell grouping")
  }

  # Always plot P2 and P3
  plots[[length(plots) + 1]] <- safe_plot_dimred(
  model, dimred = X_umap, grouping = group_onto_nearest_milestones(model),
  default_title = "Milestones Failed"
  ) + ggtitle("group_onto_nearest_milestones")

  plots[[length(plots) + 1]] <- safe_plot_dimred(
  model, dimred = X_umap, color_cells = "pseudotime",
  default_title = "Pseudotime Failed"
  ) + ggtitle("Pseudotime")

  # If feature_name exists, plot P4
  if (!is.null(feature_name)) {
  plots[[length(plots) + 1]] <- safe_plot_dimred(
    model, dimred = X_umap, feature_oi = feature_name, expression_source = dataset,
    default_title = "Feature Expression Failed"
  ) + ggtitle("Feature expression")
  }

  # Count the number of successful plots and arrange them horizontally
  num_plots <- length(plots)
  combined_plot <- plot_grid(plotlist = plots, ncol = num_plots, align = "hv")

  # Automatically calculate width (each subplot 4 inches wide)
  plot_width <- num_plots * 4

  # Save the plot
  ggsave(
  paste0(Methods_selected, "_Ti_dirmed_coler.png"),
  combined_plot,
  width = plot_width,  # Adjust width automatically
  height = 4,          # Fixed height
  dpi = 150
  )
}
Ti_dirmed_coler(model, dataset)

# Visualizing many genes along a trajectory
plot <- plot_heatmap(model, expression_source = dataset)
ggsave(paste0(Methods_selected, "_Genes_along_trajectory.png"), plot, width = 10, height = 8, dpi = 300)
print('trajectory plotting is done!')
