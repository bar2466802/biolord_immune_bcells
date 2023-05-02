remotes::install_github("YuLab-SMU/clusterProfiler", force = TRUE) 
# install the packages
remotes::install_github("YuLab-SMU/createKEGGdb", force = TRUE)
# import the library and create a KEGG database locally 
library(createKEGGdb)
species <-c("ath","hsa","mmu", "rno","dre","dme","cel")
createKEGGdb::create_kegg_db(species)
# You will get KEGG.db_1.0.tar.gz file in your working directory


library(clusterProfiler)
library(org.Hs.eg.db)

data(geneList, package="DOSE")
gene <- names(geneList)[abs(geneList) > 2]

run1 <- gseKEGG(geneList     = geneList,
                                organism      = 'hsa',
                                eps           = 0.0,
                                minGSSize     = 10,
                                maxGSSize     = 500,
                                pAdjustMethod = "none",
                                pvalueCutoff  = 1,
                                verbose       = FALSE,
                                seed          = TRUE)
