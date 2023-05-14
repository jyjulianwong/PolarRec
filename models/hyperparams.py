"""
The list of hyperparameter values for the whole recommendation algorithm.
"""


class Hyperparams:
    """
    The list of hyperparameter values for the whole recommendation algorithm.
    """
    # Max. # of keywords to extract from targets to generate resource queries.
    MAX_RES_QUERY_KEYWORDS_USED = 10
    # Max. # of results to be returned by each resource query made.
    MAX_RES_QUERY_RESULTS_RETD = 10
    # Max. # of results to be returned by each citation query made.
    MAX_CIT_QUERY_RESULTS_RETD = 10

    # The threshold value above which two resources have significant author co-occurrence.
    # See https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7279056.
    AUTHOR_COOC_PROB_TS = 0.4
    # The threshold value above which two resources have significant citation co-occurrence.
    # See https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7279056.
    CITATION_COOC_PROB_TS = 0.4

    # Max. # of keywords to extract from each resource during keyword similarity comparison.
    # This is used to rank candidate resources by their keywords.
    MAX_SIM_COMPAR_KEYWORDS_USED = 20
