"""
A Python file that can be run by the cProfile library for performance profiling.
The main function contains code that simulates the pre-loading of the
keywordRanker model and the processing of a single recommendation request.
"""
from models.evaluation import sample_resources as sr
from models.recommend import get_recommended_resources
from models.resource_filter import ResourceFilter
from models.resource_rankers.keyword_ranker import KeywordRanker

if __name__ == "__main__":
    # Load the sample resources used for evaluation.
    sample_resources = sr.load_resources_from_json()[
        sr.ARXIV_SAMPLE_FILEPATH
    ]

    # Pre-load the KeywordRanker model.
    keyword_model = KeywordRanker.get_model()

    # Run the recommendation algorithm on a single target resource.
    get_recommended_resources(
        target_resources=[sample_resources[0]],
        existing_resources=[],
        resource_filter=ResourceFilter({}),
        resource_database_ids=[],
        keyword_model=keyword_model
    )
