library("CAM")
library("igraph")
library("pcalg")
library("graph")
library("parallel")
# This script loops through the data and runs CAM on each dataset. Additionally, it stores the estimated graph in this directory. 
BASE_PATH <- "<BASE_PATH>"
numCores <- detectCores() - 1
scaled <- "scaled"
additive <- "False"
graph.type <- "ER"
N <- 100
p <- 200

path.data <- paste0(BASE_PATH, scaled, "/graph_", N, "_", p, "_additive_", additive, "_", graph.type)

subdirs <- list.dirs(path = path.data, full.names = TRUE, recursive = FALSE)
target.file.name.edges <- "result_cam.csv"
# Function to process a single directory
process_directory <- function(directory) {
  print(directory)
  path.data <- paste0(directory, "/data.csv")
  path.edges <- paste0(directory, "/edges.csv")
  path.edges.result <- file.path(directory, target.file.name.edges)
  print(path.edges.result)
  
  if (file.exists(path.edges.result)) {
    return(NULL)
  }
  
  dta <- read.csv(path.data, header=FALSE)
  r <- CAM(dta, output=F, variableSel = TRUE, pruning = TRUE, numCores = 1)
  g.estimated <- graph_from_adjacency_matrix(r$Adj)
  g.estimated <- as_graphnel(g.estimated)
  
  edges <- read.csv(path.edges, header = FALSE)
  g.true <- make_empty_graph(n = 100, directed = TRUE)
  g.true <- add_edges(g.true, t(as.matrix(edges) + 1))
  g.true <- as_graphnel(g.true)
  
  seed <- strsplit(directory, "/")[[1]][length(strsplit(directory, "/")[[1]])]
  shd_value <- shd(g.estimated, g.true)
  print(paste0("Seed: ", seed, ", SHD: ", shd_value))
  
  write_graph(graph_from_graphnel(g.estimated), path.edges.result, format="edgelist")
}

# Setup parallel cluster
cl <- makeCluster(numCores, type = "PSOCK") # PSOCK works on Windows
clusterExport(cl, varlist = c("process_directory", "target.file.name.edges", "CAM", "numCores", 
                              "path.data", "make_empty_graph", "add_edges", "shd", 
                              "graph_from_adjacency_matrix", "graph_from_graphnel", "as_graphnel"))
clusterEvalQ(cl, {
  library("CAM")
  library("igraph")
  library("pcalg")
  library("graph")
})
parLapply(cl, subdirs, process_directory)
stopCluster(cl)
