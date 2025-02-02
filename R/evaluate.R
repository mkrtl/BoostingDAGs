library("igraph")
library("pcalg")
library("graph")
library("SID")

BASE_PATH = "<BASE_PATH>"
scaled <- "scaled"
additive <- "True"
graph.type <- "ER"

main_directory <- paste0(BASE_PATH, scaled, "/graph_", N, "_", p, "_additive_", additive, "_", graph.type)


target_filename <- "result_cam.csv" # Can for example also be target_filename <- "boosting_cam_edges.csv"

# Function to load the graph from a CSV file and create a graphNEL object
create_graph <- function(filepath, num_nodes=100) {
  if (grepl("result_cam",filepath)){
    sep <- " "}else{
      sep <- ","
    }
  # Read the edges from the CSV file
  edges <- read.csv(filepath, header = FALSE, sep=sep)
  edges <- edges + 1
  g <- make_empty_graph(n = 100, directed = TRUE)
  g <- add_edges(g, t(as.matrix(edges)))
  g <- as_graphnel(g)
  return(g)
}


# Main function to loop through directories
process_directory <- function(main_directory, target_filename, ref_graph_path, num_nodes) {
  # Initialize result arrays
  shd_results <- c()
  sid_results <- c()
  true_positive_results <- c()
  recall_results <- c()
  
  # Loop through subdirectories
  subdirectories <- list.dirs(main_directory, recursive = FALSE)
  for (subdir in subdirectories) {
    target_filepath <- file.path(subdir, target_filename)
    true_graph_filepath <- file.path(subdir, ref_graph_path)
    print(target_filepath)
    # Check if the file exists
    if (file.exists(target_filepath)) {
      # Load reference graph
      estimated_graph <- create_graph(target_filepath)
      # Create graphNEL object
      true_graph <- create_graph(true_graph_filepath)
      
      # Apply comparison functions
      shd_to_add <- shd(true_graph, estimated_graph)
      cat("SHD: ", shd_to_add, "\n")
      #sid_to_add <- structIntervDist(true_graph, estimated_graph)$sid
      # True positives
      adjacency_estimated <- as(estimated_graph, "matrix")
      adjacency_true <- as(true_graph, "matrix")
      matching_ones <- (adjacency_estimated == 1) & (adjacency_true == 1)
      
      # Count and return the number of TRUE values
      true_positives <- sum(matching_ones, na.rm = TRUE)
      tpr <- true_positives / sum(adjacency_estimated) 
      
      # Recall 
      recall <- true_positives / sum(adjacency_true)
      
      
      # cat("SID ", sid_to_add, "\n")
      cat("--------------------- \n")
      shd_results <- c(shd_results, shd_to_add)
      recall_results <- c(recall_results, recall)
      true_positive_results = c(true_positive_results, tpr)
      # sid_results <- c(sid_results, sid_to_add)
    }
  }
  
  # Return the results
  return(list(shd = shd_results, sid = sid_results, recall_results = recall_results, 
              true_positive_results = true_positive_results))
}

# Parameters
num_nodes <- 100                           # Total number of nodes in the graph

# Process the directory
results <- process_directory(main_directory, target_filename, ref_graph_path, num_nodes)

# Print results
print(results)

print("TPR")
print(mean(results$true_positive_results))
print(sd((results$true_positive_results)))

print("Recall")
print(mean(results$recall_results))
print(sd((results$recall_results)))

print("SHD")
print(mean(results$shd))
print(sd((results$shd)))

print("SID")
print(mean(results$sid))
print(sd((results$sid)))
