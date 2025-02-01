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
from BRAD.agent import Agent # , AgentFactory

# For the Video RAG
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

VERBOSE = True

THRESHOLD_P_VALUE = 0.05
MAXIMUM_ENRICHMENT_TERMS = 10

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

def perform_enrichment(gene_string):
    print(f"{gene_string=}")
    genes = [gene.strip() for gene in gene_string.split(',')]
    print(f"{genes=}")
    output_file = "enrichment_results.xlsx"

    # Perform enrichment
    cell_type_df = gget.enrichr(genes, database="celltypes")
    print(f"{cell_type_df.columns=}") if VERBOSE else None
    pathway_df = gget.enrichr(genes, database="pathway")
    print(f"{pathway_df.columns=}") if VERBOSE else None

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
