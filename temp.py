import os
# genes = ["MYOD1", "P53", "CDK2"]
gene_string = "CCND1, CCND2, CCND3, CDK1, CDK2, CDK4, CDK6, CDKN1A (p21), CDKN1B (p27), CDKN2A, CDKN2B, CCNE1, CCNE2, E2F1, E2F2, E2F3, RB1, AURKA, AURKB, PLK1, PLK4, BUB1, BUBR1, MAD2L1, CDC20, CDC25A, CDC25B, CDC25C, CHEK1, CHEK2, MYC, RRM2, MCM2, MCM3, MCM4, MCM5, MCM6, MCM7, RFC1, RFC2, RFC3, RFC4, RFC5, RAD51, RAD54, RAD17, TTK, PTTG1, CDH1, CDC14A, CDC14B, SKP2, FBXW7, WEE1, CDK7, CDK9, MCL1, PRC1, KIF11, KIF14, TTK, PLK2, PLK3, CDK8, CCAT1, FZR1"
genes = [gene.strip() for gene in gene_string.split(',')]
output_file = "enrichment_results.xlsx"
verbose = True

THRESHOLD_P_VALUE = 0.05
MAXIMUM_ENRICHMENT_TERMS = 10


import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment


import gget
import sys
import os
sys.path.append('../BRAD')
from rich import print
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# Imports for BRAD library
from BRAD.agent import Agent, AgentFactory
# from BRAD.utils import delete_dirs_without_log, strip_root_path
from BRAD.endpoints import parse_log_for_one_query

# For the Video RAG
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

SYSTEM_ENRICHMENT_PROMPT = """You are a bioinformatics specialist tasked with summarizing gene enrichment results.  
Provide a clear, concise, and scientifically accurate 1-5 sentence explanation of the biological significance of the given pathway.  
Focus solely on the pathway's function, relevance, and associated biological processes.  
Do not include information about yourself or the nature of this request.  

Pathway: """

SYSTEM_ENRICHMENT_TYPE_PROMPT = """You are a bioinformatics specialist tasked with summarizing gene enrichment results.  
Provide a clear, concise, and scientifically accurate 2-5 sentence explanation of the following enrichment
dataframe. Focus primarily on the relationship between the pathways with one another and on the high level themes
observed accross the enrichment results. Do not include information about yourself or the nature of this request.  

Enrichment Results: """


def process_pathway(pathway):
    """Initialize a separate BRAD instance for each pathway and invoke it."""
    brad = Agent(
        tools=['RAG'],
        gui=True,
        config='config.json',
        interactive=False
    )
    
    return brad.invoke(SYSTEM_ENRICHMENT_PROMPT + pathway)

def summarize_enrichment_type(enrichment_df):
    """Summarize the results of an enrichment dataframe"""
    brad = Agent(
        tools=['RAG'],
        gui=True,
        config='config.json',
        interactive=False
    )
    minimal_enrichment_df = enrichment_df[['rank', 'path_name', 'BRAD Result']].copy()
    return brad.invoke(SYSTEM_ENRICHMENT_TYPE_PROMPT + minimal_enrichment_df.to_json(orient="records"))


# Perform enrichment
cell_type_df = gget.enrichr(genes, database="celltypes")
print(f"{cell_type_df.columns=}") if verbose else None
pathway_df = gget.enrichr(genes, database="pathway")
print(f"{pathway_df.columns=}") if verbose else None

enrichment_dfs = [cell_type_df, pathway_df]

# Preprocess the dataframes
for dfi, df in enumerate(enrichment_dfs):
    filtered_df = df[df['p_val'] < THRESHOLD_P_VALUE].copy()
    filtered_df = filtered_df.sort_values(by='p_val').head(min(filtered_df.shape[0], MAXIMUM_ENRICHMENT_TERMS))
    filtered_df = filtered_df.drop(["database"], axis=1)
    enrichment_dfs[dfi] = filtered_df.copy()

# Process Enrichment with RAG
for dfi, df in enumerate(enrichment_dfs):
    pathways = df['path_name'].tolist()
    
    # Use ThreadPoolExecutor for parallel execution
    with ThreadPoolExecutor() as executor:
        enrichment_summaries = list(executor.map(process_pathway, pathways))

#    print(f"{len(enrichment_summaries)=}")
#    print(f"{df.shape=}")
    df.loc[:, 'BRAD Result'] = enrichment_summaries  # Safe assignment
    enrichment_dfs[dfi] = df.copy()  # Update the list with the processed DataFrame


# Use ThreadPoolExecutor for parallel execution
with ThreadPoolExecutor() as executor:
    enrichment_summaries = list(executor.map(summarize_enrichment_type, enrichment_dfs))

highlevel_df = pd.DataFrame(
    {
        "Enrichment Type": ["Cell Type", "Pathway"],
        "Summaries": enrichment_summaries
    }
)

reference_df = pd.DataFrame(
    {
        'Topic': [
            'Gene Enrichment', 
            'Fisher\'s Exact Test', 
            'Gene Ontology (GO)', 
            'Panglaodb Database'
        ],
        'Explanation': [
            "Gene enrichment analysis is used to identify biological terms or pathways that are statistically overrepresented in a set of genes, compared to a reference set. This helps in understanding the biological processes or functions that may be associated with the gene set.",
            "Fisher's Exact Test is a statistical test used to determine if there are nonrandom associations between two categorical variables. In gene enrichment, it is often used to evaluate whether a particular gene is overrepresented in a predefined set of categories (e.g., pathways or GO terms) compared to the whole genome.",
            "Gene Ontology (GO) provides a standardized vocabulary of terms to describe gene functions, cellular components, and biological processes. GO terms help in annotating genes and interpreting gene function across different species.",
            "Panglaodb is a comprehensive database that provides gene expression data from multiple species, integrating data from various sources. It allows researchers to access transcriptomic and functional genomics data to support gene enrichment analyses."
        ]
    }
)

# Save to Excel with two sheets
with pd.ExcelWriter(output_file, engine="openpyxl") as writer:

    # Write DataFrames to sheets
    highlevel_df.to_excel(writer, sheet_name="Overview", index=False)
    enrichment_dfs[0].to_excel(writer, sheet_name="Celltypes", index=False)
    enrichment_dfs[1].to_excel(writer, sheet_name="Pathways", index=False)
    reference_df.to_excel(writer, sheet_name="BRAD-Enrichment", index=False)

    # Access the workbook
    workbook = writer.book

    # Apply formatting to each sheet
    for sheeti, sheet_name in enumerate(writer.sheets):
        worksheet = writer.sheets[sheet_name]
        worksheet.freeze_panes = "A2"

        for coli, col in enumerate(worksheet.iter_cols()):
            col_name = col[0].value  # First row contains column names
            
            if col_name == "BRAD Result":
                current_width = worksheet.column_dimensions[col[0].column_letter].width or 8.43
                worksheet.column_dimensions[col[0].column_letter].width = current_width * 6
            elif col_name == "rank":
                current_width = worksheet.column_dimensions[col[0].column_letter].width or 8.43
                worksheet.column_dimensions[col[0].column_letter].width = current_width * 0.5
            elif col_name == "path_name":
                current_width = worksheet.column_dimensions[col[0].column_letter].width or 8.43
                worksheet.column_dimensions[col[0].column_letter].width = current_width * 1.5
            
            if sheeti == 0:
                if coli == 0:
                    current_width = worksheet.column_dimensions[col[0].column_letter].width or 8.43
                    worksheet.column_dimensions[col[0].column_letter].width = current_width * 2
                elif coli == 1:
                    current_width = worksheet.column_dimensions[col[0].column_letter].width or 8.43
                    worksheet.column_dimensions[col[0].column_letter].width = current_width * 8
                for cell in col:
                    cell.alignment = Alignment(wrap_text=True, vertical="top")

            if col_name in ["BRAD Result", "path_name"]:
                for cell in col[1:]:
                    cell.alignment = Alignment(wrap_text=True, vertical="top")
            else:
                for cell in col[1:]:
                    cell.alignment = Alignment(vertical="top")

    workbook.save(output_file)
