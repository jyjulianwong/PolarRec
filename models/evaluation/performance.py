"""
Objects and methods used to calculate the processing time and mean peak memory
usage of the recommendation algorithm.
"""
import time
import tracemalloc
from models.evaluation import sample_resources as sr
from models.recommend import get_recommended_resources
from models.resource_filter import ResourceFilter
from models.resource_rankers.keyword_ranker import KeywordRanker

if __name__ == "__main__":
    # Run this .py file to calculate the system's performance metrics.
    sample_resources = sr.load_resources_from_json()[sr.ARXIV_SAMPLE_FILEPATH]
    # Pre-load the KeywordRanker model.
    keyword_model = KeywordRanker.get_model()

    # Record the processing times and peak memory usages for each run.
    proc_times: list[float] = []
    peak_mem_sizes: list[float] = []
    for i, resource in enumerate(sample_resources):
        # New memory blocks allocated from this point on will count as "usage".
        tracemalloc.start()
        # Record the starting timestamp of the recommender system's process.
        t1 = time.time()
        _, _ = get_recommended_resources(
            target_resources=[resource],
            existing_resources=[],
            resource_filter=ResourceFilter({}),
            resource_database_ids=[],
            keyword_model=keyword_model
        )
        # Record the ending timestamp of the recommender system's process.
        t2 = time.time()
        # Record the total size of new memory blocks allocated.
        peak_mem_size = tracemalloc.get_traced_memory()[1]
        # Reset the tracing of memory block allocation before the next run.
        tracemalloc.stop()

        # Make appropriate calculations after all tracing tasks have stopped.
        proc_time = t2 - t1
        proc_times.append(proc_time)
        print(f"Processing time {i}: {proc_time}")
        peak_mem_size /= 1048576
        peak_mem_sizes.append(peak_mem_size)
        print(f"Peak memory usage {i}: {peak_mem_size}")

    mean_proc_time = sum(proc_times) / len(proc_times)
    print(f"Mean processing time: {mean_proc_time} seconds")
    mean_peak_mem_size = sum(peak_mem_sizes) / len(peak_mem_sizes)
    print(f"Mean peak memory usage: {mean_peak_mem_size} MB")
