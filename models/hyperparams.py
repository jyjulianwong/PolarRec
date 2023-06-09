"""
The list of hyperparameter values for the whole recommendation algorithm.
"""


class Hyperparams:
    """
    The list of hyperparameter values for the whole recommendation algorithm.
    """
    # Whether resource and citation data should be retrieved from cache or not.
    DISABLE_CACHED_DATA = False

    # Max. # of keywords to extract from targets to generate resource queries.
    MAX_RES_QUERY_KEYWORDS_USED = 10
    # Max. # of results to be returned by each resource query made.
    # Each resource returned becomes a candidate for being recommended,
    # and is each processed in sequence by Rankers when being assigned scores.
    MAX_RES_QUERY_RESULTS_RETD = 20
    # Max. # of results to be returned by each citation query made.
    # This is used when searching for citation data given a known resource.
    # Normally, changing this value has no effect on processing time,
    # because the first result returned by the query is usually the correct one.
    MAX_CIT_QUERY_RESULTS_RETD = 10

    # The collaborative filtering method used for both author- and citation-based ranking.
    UNIVERSAL_CF_METHOD = "userbased"  # "userbased", "contextbased"
    # The threshold value above which two resources have significant author co-occurrence.
    # See https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7279056.
    # Referencing Section IV (D).
    AUTHOR_COOC_PROB_TS = 0.5
    # The threshold value above which two resources have significant citation co-occurrence.
    # See https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7279056.
    # Referencing Section IV (D).
    CITATION_COOC_PROB_TS = 0.5

    # The keyword extraction and ranking method used.
    KEYWORD_RANK_METHOD = "tfidf"  # "tfidf", "textrank"
    # Max. # of keywords to extract from each resource during keyword similarity comparison.
    # This is used to rank candidate resources by their keywords.
    MAX_SIM_COMPAR_KEYWORDS_USED = 20
