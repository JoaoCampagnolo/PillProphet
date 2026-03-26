"""Run the data ingestion pipeline."""

from pillprophet.utils.logging import setup_logging

logger = setup_logging()


def main():
    logger.info("Starting data ingestion pipeline")
    # TODO: Implement ingestion from ClinicalTrials.gov API
    logger.info("Ingestion pipeline not yet implemented")


if __name__ == "__main__":
    main()
