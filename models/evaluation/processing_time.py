"""
Objects and methods used to calculate the processing time of the recommendation
algorithm.
"""
import time
from models.evaluation import sample_resources as sr
from models.recommend import get_ranked_resources
from models.resource_filter import ResourceFilter
from models.resource_rankers.keyword_ranker import KeywordRanker

if __name__ == "__main__":
    sample_resources = sr.load_resources_from_json()[sr.ARXIV_SAMPLE_FILEPATH]
    keyword_model = KeywordRanker.get_model()

    proc_times: list[float] = []
    for i, resource in enumerate(sample_resources):
        t1 = time.time()
        ranked_resources = get_ranked_resources(
            target_resources=[resource],
            existing_resources=[],
            resource_filter=ResourceFilter({}),
            resource_database_ids=[],
            keyword_model=keyword_model
        )["ranked_database_resources"]
        t2 = time.time()

        proc_time = t2 - t1
        proc_times.append(proc_time)
        print(f"Processing time {i}: {proc_time}")

    mean_proc_time = sum(proc_times) / len(proc_times)
    print(f"Mean processing time: {mean_proc_time}")
