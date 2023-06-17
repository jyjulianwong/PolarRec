"""
A Python file used to calculate the proportion of out-of-vocabulary words that
exist in the list of sample resources used for evaluation.
"""
from models.evaluation import sample_resources as sr
from models.resource import Resource
from models.resource_rankers.keyword_ranker import KeywordRanker

if __name__ == "__main__":
    # Run this .py file to calculate the system's corpus coverage.
    # Collect all sample resources into a list.
    sample_resources: list[Resource] = []
    sample_resources += sr.load_resources_from_json()[
        sr.ARXIV_SAMPLE_FILEPATH
    ]
    sample_resources += sr.load_resources_from_json()[
        sr.IEEE_XPLORE_SAMPLE_FILEPATH
    ]

    # Pre-load the KeywordRanker model.
    model = KeywordRanker.get_model()

    # The total number of words processed from the sample resources.
    word_count = 0
    # The number of out-of-vocabulary words processed.
    oovw_count = 0

    for i, resource in enumerate(sample_resources):
        # Retrieve text from the resource's title and abstract.
        summary = f"{resource.title} {resource.abstract}"
        # Remove any punctuation and special characters, but keep whitespace.
        summary = Resource.get_comparable_str(summary)

        for word in summary.split():
            word_count += 1
            if word not in model.key_to_index:
                # This word does not exist in the model's vocabulary.
                oovw_count += 1

        # Preview a rolling statistic of how many OOV words have been processed.
        print(f"Sample resource {i}")
        print(f"\tRolling total word count: {word_count}")
        print(f"\tRolling out-of-vocabulary word count: {oovw_count}")

    print(f"Out-of-vocabulary percentage: {(oovw_count / word_count) * 100}%")
