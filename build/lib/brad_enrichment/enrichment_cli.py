import click
import logging
from brad_enrichment import perform_enrichment  # Assuming perform_enrichment is defined elsewhere
from brad_enrichment import DEFAULT_QUERY

logger = logging.getLogger(__name__)

@click.command()
@click.argument("gene_string", type=str)
@click.option(
    "--databases", "-d", multiple=True, 
    default=['KEGG_2021_Human', 'GO_Biological_Process_2021', 'PanglaoDB_Augmented_2021'], 
    help="List of databases to use for enrichment (default: KEGG, GO, PanglaoDB)."
)
@click.option(
    "--threshold-p-value", "-p", type=float, default=0.05, 
    help="P-value threshold for enrichment results (default: 0.05)."
)
@click.option(
    "--minimum-enrichment-terms", "-min", type=int, default=3, 
    help="Minimum number of enrichment terms to report (default: 3)."
)
@click.option(
    "--maximum-enrichment-terms", "-max", type=int, default=10, 
    help="Maximum number of enrichment terms to report (default: 10)."
)
@click.option(
    "--literature-database", "-l", type=str, default="../databases/enrichment_database",
    help="Path to the literature database (default: ../databases/enrichment_database)."
)
@click.option(
    "--output", "-o", type=str, default="enrichment_results.xlsx",
    help="Output file name for results (default: enrichment_results.xlsx)."
)
@click.option(
    "--query", "-q", type=str, default=DEFAULT_QUERY, 
    help="Custom query for enrichment analysis (default: standard query)."
)
@click.option(
    "--verbose", "-v", is_flag=True, 
    help="Enable verbose mode for debugging and logging."
)
def cli(gene_string, databases, threshold_p_value, minimum_enrichment_terms, maximum_enrichment_terms, 
        literature_database, output, query, verbose):
    """
    Perform gene enrichment analysis using BRAD.
    
    GENE_STRING: Comma-separated list of genes (e.g., MYOD, P53, CDK2).
    """
    
    if verbose:
        logger.info(f"Output file: {output}")
        logger.info(f"Literature database: {literature_database}")

    # Run the enrichment function
    results = perform_enrichment(
        gene_string=gene_string,
        databases=list(databases),  # Convert tuple to list
        threshold_p_value=threshold_p_value,
        minimum_enrichment_terms=minimum_enrichment_terms,
        maximum_enrichment_terms=maximum_enrichment_terms,
        literature_database=literature_database,
        query=query,
        output_file=output,
        verbose=verbose
    )

if __name__ == "__main__":
    cli()
