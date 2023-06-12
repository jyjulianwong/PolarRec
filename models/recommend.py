"""
Objects and methods for the /recommend service of the API.
"""
import math
import time
from models.citation_database_adapters.adapter import Adapter
from models.citation_database_adapters.s2ag_adapter import S2agAdapter
from models.hyperparams import Hyperparams as hp
from models.resource import RankableResource, Resource
from models.resource_database_adapters.query_builder import QueryBuilder
from models.resource_database_adapters.arxiv_query_builder import \
    ArxivQueryBuilder
from models.resource_database_adapters.ieee_xplore_query_builder import \
    IEEEXploreQueryBuilder
from models.resource_filter import ResourceFilter
from models.resource_rankers.author_ranker import AuthorRanker
from models.resource_rankers.citation_ranker import CitationRanker
from models.resource_rankers.citation_count_ranker import CitationCountRanker
from models.resource_rankers.keyword_ranker import KeywordRanker
from typing import List, Type


# Support for type hinting requires Python 3.5.
def _get_res_db_query_builders() -> List[Type[QueryBuilder]]:
    """
    :return: The list of resource databases the recommender system supports.
    :rtype: list[Type[QueryBuilder]]
    """
    return [ArxivQueryBuilder, IEEEXploreQueryBuilder]


# Support for type hinting requires Python 3.5.
def _get_cit_db_adapter() -> Type[Adapter]:
    """
    :return: The selected citation database the recommender system uses.
    :rtype: Type[Adapter]
    """
    # FIXME: Modify this to return a list of adapters instead of a single one.
    return S2agAdapter


def _get_candidate_resources(
    target_keywords,
    target_authors,
    # Support for type hinting requires Python 3.5.
    query_builder: Type[QueryBuilder]
):
    """
    :param target_keywords: The list of keywords from the target resource(s).
    :type target_keywords: list[str]
    :param target_authors: The list of authors from the target resource(s).
    :type target_authors: list[str]
    :param query_builder: The resource database adapter to query.
    :type query_builder: Type[QueryBuilder]
    :return: A list of candidate resources based on target data.
    :rtype: list[Resource]
    """
    candidates: list[Resource] = []

    # Generate the keyword-based search query.
    if len(target_keywords) > 0:
        keyword_based_query_builder = query_builder()
        # Only add the top keywords into the query.
        keyword_based_query_builder.add_keywords(target_keywords[:min(
            len(target_keywords), hp.MAX_RES_QUERY_KEYWORDS_USED
        )])
        # Include results that only contain some of the keywords, not all.
        candidates += keyword_based_query_builder.get_resources(
            hp.MAX_RES_QUERY_RESULTS_RETD,
            must_have_all_fields=False
        )

    # Generate the author-based search query.
    if len(target_authors) > 0:
        author_based_query_builder = query_builder()
        author_based_query_builder.set_authors(target_authors)
        # Include results that only contain some of the authors, not all.
        candidates += author_based_query_builder.get_resources(
            hp.MAX_RES_QUERY_RESULTS_RETD,
            must_have_all_fields=False
        )

    return candidates


def _set_resource_references(resources):
    """
    Searches for and assigns references to resources using the chosen citation
    database adapter(s).

    :param resources: The list of resources to collect references for.
    :type resources: list[Resource]
    """
    # Only process resources that do not have existing reference data.
    ress_with_no_refs: list[Resource] = []
    for resource in resources:
        if resource.references is None:
            ress_with_no_refs.append(resource)

    cit_db_adapter = _get_cit_db_adapter()
    reference_dict = cit_db_adapter.get_references(ress_with_no_refs)
    for resource, references in reference_dict.items():
        if len(references) > 0:
            # The citation database adapter has found data for this resource.
            resource.references = references


def _set_resource_citation_counts(resources):
    """
    Searches for and assigns citation counts to resources using the chosen
    citation database adapter(s).

    :param resources: The list of resources to collect citation counts for.
    :type resources: list[Resource]
    """
    # Only process resources that do not have existing citation count data.
    ress_with_no_cit_counts: list[Resource] = []
    for resource in resources:
        if resource.citation_count is None:
            ress_with_no_cit_counts.append(resource)

    cit_db_adapter = _get_cit_db_adapter()
    cit_count_dict = cit_db_adapter.get_citation_count(ress_with_no_cit_counts)
    for resource, count in cit_count_dict.items():
        if count >= 0:
            # The citation database adapter has found data for this resource.
            resource.citation_count = count


def _get_candidate_mean_rank_dict(
    candidate_resources,
    target_resources,
    target_keywords,
    keyword_model
):
    """
    Ranks candidates using multiple specialised algorithms (Rankers).
    Uses the mean ranking across all rankers as the "recommendation score".
    Therefore, a smaller value corresponds to a better recommendation, with the
    best possible score being 1.

    :param candidate_resources: The list of candidate resources.
    :type candidate_resources: list[Resource]
    :param target_resources: The list of target resources.
    :type target_resources: list[Resource]
    :param target_keywords: The list of keywords from the target resource.
    :type target_keywords: list[str]
    :param keyword_model: The word embedding model to be used for keywords.
    :type keyword_model: Word2Vec.KeyedVectors
    :return: The ranked candidate resources with their mean rankings.
    :rtype: dict[RankableResource, float]
    """
    # Convert Resource objects to RankableResource objects.
    rankable_cand_ress = [c.to_rankable_resource() for c in candidate_resources]
    # The collection of resource rankers used to rank the candidates.
    rankers = [AuthorRanker, CitationRanker, CitationCountRanker, KeywordRanker]
    for ranker in rankers:
        # Assign rankings for each type of Ranker.
        ranker.set_resource_rankings(
            rankable_resources=rankable_cand_ress,
            target_resources=target_resources,
            # Additional arguments for KeywordRanker.
            kw_model=keyword_model
        )

    # The mapping of candidates to their weighted mean rankings.
    cand_mean_rank_dict: dict[RankableResource, float] = {}
    for ranked_cand_res in rankable_cand_ress:
        # Retrieve rankings set by resource rankers.
        # Calculate the weighted mean ranking across all resource rankers.
        mean_rank = 0.25 * ranked_cand_res.author_based_ranking
        mean_rank += 0.25 * ranked_cand_res.citation_based_ranking
        mean_rank += 0.15 * ranked_cand_res.citation_count_ranking
        mean_rank += 0.35 * ranked_cand_res.keyword_based_ranking
        cand_mean_rank_dict[ranked_cand_res] = mean_rank
    return cand_mean_rank_dict


def get_recommended_resources(
    target_resources,
    existing_resources,
    resource_filter,
    resource_database_ids,
    keyword_model
):
    """
    The main recommendation algorithm of the recommender system.
    Queries research and citation databases and recommends lists of academic
    resources based on target resources, sorted according to their relevance.
    If ``resource_database_ids`` is empty, assumes that all available resource
    databases can be searched through to obtain recommendations.

    :param target_resources: The list of target resources.
    :type target_resources: list[Resource]
    :param existing_resources: The list of existing resources.
    :type existing_resources: list[Resource]
    :param resource_filter: The filter to be applied to the results.
    :type resource_filter: ResourceFilter
    :param resource_database_ids: The list of databases to use for searching.
    :type resource_database_ids: list[str]
    :param keyword_model: The word embedding model to be used for keywords.
    :type keyword_model: Word2Vec.KeyedVectors
    :return: Two lists of existing and database-extracted recommendations.
    :rtype: tuple[list[RankableResource], list[RankableResource]]
    """
    # Extract keywords from target resources.
    target_keywords = KeywordRanker.get_keywords(resources=target_resources)
    # Extract the list of authors from target resources.
    target_authors = []
    for resource in target_resources:
        if resource.authors is not None:
            target_authors += resource.authors

    # Generate keyword- and author-based search queries to resource databases,
    # and collect candidate resources.
    candidate_resources: list[Resource] = []
    # Generate the appropriate queries for each supported resource database.
    for query_builder in _get_res_db_query_builders():
        if len(resource_database_ids) > 0:
            # User has specified filters for what databases to use.
            if query_builder.get_id() not in resource_database_ids:
                # This database is not included in the user's "whitelist".
                continue

        candidate_resources += _get_candidate_resources(
            target_keywords=target_keywords,
            target_authors=target_authors,
            query_builder=query_builder
        )

    # Add existing resources into the pool of candidate resources for ranking.
    candidate_resources += existing_resources
    # Remove any duplicate resources from the multiple queries.
    candidate_resources = list(dict.fromkeys(candidate_resources))
    # Remove any occurrences of target resources themselves from the candidates.
    for target_resource in target_resources:
        if target_resource in candidate_resources:
            candidate_resources.remove(target_resource)

    # Remove any candidates that do not match the specified filters.
    candidate_resources = resource_filter.get_filtered(candidate_resources)

    # Collect references for all resources using citation database adapters.
    # This information is not available from resource database adapters.
    _set_resource_references(candidate_resources + target_resources)
    _set_resource_citation_counts(candidate_resources + target_resources)

    # Assign a weighted mean ranking (the "score") for every candidate resource.
    candidate_mean_rank_dict = _get_candidate_mean_rank_dict(
        candidate_resources=candidate_resources,
        target_resources=target_resources,
        target_keywords=target_keywords,
        keyword_model=keyword_model
    )
    # Make the weighted mean ranking the primary key used for sorting.
    candidate_mean_rank_dict = [
        (s, c) for c, s in candidate_mean_rank_dict.items()
    ]
    # Sort the list of candidates in order of their scores.
    # Only include existing resources if they appear in the top half of all...
    # ... the recommended resources, i.e. are genuinely related to the targets.
    existing_included_th = math.floor(0.5 * len(candidate_mean_rank_dict))
    reco_existing_resources = [
        c for s, c in sorted(candidate_mean_rank_dict)[:existing_included_th] if
        c in existing_resources
    ]
    reco_database_resources = [
        c for s, c in sorted(candidate_mean_rank_dict) if
        c not in existing_resources
    ]

    return reco_existing_resources, reco_database_resources


if __name__ == "__main__":
    # Pre-load the KeywordRanker model.
    keyword_model = KeywordRanker.get_model()

    # Define the example target data to be fed into the recommender system.
    target_data = {
        "authors": ["Vijay Badrinarayanan", "Alex Kendall", "Roberto Cipolla"],
        "title": "SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation",
        "year": 2015,
        "abstract": """We present a novel and practical deep fully convolutional 
neural network architecture for semantic pixel-wise segmentation termed SegNet. 
This core trainable segmentation engine consists of an encoder network, a 
corresponding decoder network followed by a pixel-wise classification layer. The
architecture of the encoder network is topologically identical to the 13 
convolutional layers in the VGG16 network [1] . The role of the decoder network 
is to map the low resolution encoder feature maps to full input resolution 
feature maps for pixel-wise classification. The novelty of SegNet lies is in the
manner in which the decoder upsamples its lower resolution input feature map(s).
Specifically, the decoder uses pooling indices computed in the max-pooling step 
of the corresponding encoder to perform non-linear upsampling. This eliminates 
the need for learning to upsample. The upsampled maps are sparse and are then 
convolved with trainable filters to produce dense feature maps. We compare our 
proposed architecture with the widely adopted FCN [2] and also with the well 
known DeepLab-LargeFOV [3] , DeconvNet [4] architectures. This comparison 
reveals the memory versus accuracy trade-off involved in achieving good 
segmentation performance. SegNet was primarily motivated by scene understanding 
applications. Hence, it is designed to be efficient both in terms of memory and 
computational time during inference. It is also significantly smaller in the 
number of trainable parameters than other competing architectures and can be 
trained end-to-end using stochastic gradient descent. We also performed a 
controlled benchmark of SegNet and other architectures on both road scenes and 
SUN RGB-D indoor scene segmentation tasks. These quantitative assessments show 
that SegNet provides good performance with competitive inference time and most 
efficient inference memory-wise as compared to other architectures. We also 
provide a Caffe implementation of SegNet and a web demo at 
http://mi.eng.cam.ac.uk/projects/segnet/.""",
        "doi": "10.1109/TPAMI.2016.2644615",
        "url": "https://ieeexplore.ieee.org/document/7803544"
    }
    target_resource = Resource(target_data)

    print("\nrecommend: Search and set the reference list for a resource")

    t1 = time.time()
    _set_resource_references([target_resource])
    t2 = time.time()
    print("recommend: target_resource.references:")
    if target_resource.references is None:
        # No references could be found for this target resource.
        print(None)
    else:
        for i, reference in enumerate(target_resource.references):
            # Preview the title of each reference used in this target resource.
            print(f"\t[{i}]: {reference.title}")
    print(f"recommend: Time taken to execute: {t2 - t1} seconds")

    print("\nrecommend: Generate database recommendations for a resource")

    t1 = time.time()
    _, reco_database_ress = get_recommended_resources(
        [target_resource],
        [],
        ResourceFilter({
            # "authors": ["Vijay Badrinarayanan"]
        }),
        [],
        keyword_model
    )
    t2 = time.time()
    print(f"recommend: ranked_resources: {len(reco_database_ress)}")
    for i, reco_res in enumerate(reco_database_ress):
        # Preview the information from each recommendation,
        # in a similar fashion to the Zotero plugin's results view.
        print(f"{i}: {reco_res.title}")
        print(f"\t{reco_res.authors}")
        print(f"\t{reco_res.year}")
        print(f"\t{reco_res.doi}")
        print(f"\t{reco_res.url}")
        print(f"\tAuthor-based ranking:   {reco_res.author_based_ranking}")
        print(f"\tCitation-based ranking: {reco_res.citation_based_ranking}")
        print(f"\tCitation count ranking: {reco_res.citation_count_ranking}")
        print(f"\tKeyword-based ranking:  {reco_res.keyword_based_ranking}")
    print(f"recommend: Time taken to execute: {t2 - t1} seconds")
