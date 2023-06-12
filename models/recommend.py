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
    :return: The list of resource databases the recommendation algorithm calls.
    :rtype: list[Type[QueryBuilder]]
    """
    return [ArxivQueryBuilder, IEEEXploreQueryBuilder]


# Support for type hinting requires Python 3.5.
def _get_cit_db_adapter() -> Type[Adapter]:
    """
    :return: The citation database the recommendation algorithm calls.
    :rtype: Type[Adapter]
    """
    # TODO: Return a list of adapters instead of a chosen one.
    return S2agAdapter


def _get_candidate_resources(
    target_keywords,
    target_authors,
    # Support for type hinting requires Python 3.5.
    query_builder: Type[QueryBuilder]
):
    """
    :param target_keywords: The list of keywords from the target resources.
    :type target_keywords: list[str]
    :param target_authors: The list of authors from the target resources.
    :type target_authors: list[str]
    :param query_builder: The resource database to call.
    :type query_builder: Type[QueryBuilder]
    :return: A list of candidate resources based on the target resource.
    :rtype: list[Resource]
    """
    candidates: list[Resource] = []

    # Query 1: The keyword-based query.
    if len(target_keywords) > 0:
        keyword_based_query_builder = query_builder()
        keyword_based_query_builder.add_keywords(target_keywords[:min(
            len(target_keywords), hp.MAX_RES_QUERY_KEYWORDS_USED
        )])
        candidates += keyword_based_query_builder.get_resources(
            hp.MAX_RES_QUERY_RESULTS_RETD,
            must_have_all_fields=False
        )

    # Query 2: The author-based query.
    if len(target_authors) > 0:
        author_based_query_builder = query_builder()
        author_based_query_builder.set_authors(target_authors)
        candidates += author_based_query_builder.get_resources(
            hp.MAX_RES_QUERY_RESULTS_RETD,
            must_have_all_fields=False
        )

    return candidates


def _set_resource_references(resources):
    """
    Searches for and assigns references to resources using the chosen citation
    database adapters.

    :param resources: The list of resources to collect references for.
    :type resources: list[Resource]
    """
    ress_with_no_refs: list[Resource] = []
    for resource in resources:
        if resource.references is None:
            ress_with_no_refs.append(resource)

    cit_db_adapter = _get_cit_db_adapter()
    reference_dict = cit_db_adapter.get_references(ress_with_no_refs)
    for resource, references in reference_dict.items():
        if len(references) > 0:
            resource.references = references


def _set_resource_citation_counts(resources):
    """
    Searches for and assigns citation counts to resources using the chosen
    citation database adapters.

    :param resources: The list of resources to collect citation counts for.
    :type resources: list[Resource]
    """
    ress_with_no_cit_counts: list[Resource] = []
    for resource in resources:
        if resource.citation_count is None:
            ress_with_no_cit_counts.append(resource)

    cit_db_adapter = _get_cit_db_adapter()
    cit_count_dict = cit_db_adapter.get_citation_count(ress_with_no_cit_counts)
    for resource, count in cit_count_dict.items():
        if count > 0:
            resource.citation_count = count


def _get_candidate_mean_rank_dict(
    candidate_resources,
    target_resources,
    target_keywords,
    keyword_model
):
    """
    Ranks candidates using multiple specialised algorithms (Rankers).
    Uses the mean ranking across all algorithms as the "recommendation score".
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
    rankers = [AuthorRanker, CitationRanker, CitationCountRanker, KeywordRanker]
    for ranker in rankers:
        # Assign rankings for each type of Ranker.
        ranker.set_resource_rankings(
            rankable_resources=rankable_cand_ress,
            target_resources=target_resources,
            # Additional arguments for KeywordRanker.
            kw_model=keyword_model
        )

    cand_mean_rank_dict: dict[RankableResource, float] = {}
    for ranked_cand_res in rankable_cand_ress:
        # Retrieve rankings set by rankers.
        # Calculate the mean ranking across all rankers.
        mean_rank = 0.25 * ranked_cand_res.author_based_ranking
        mean_rank += 0.25 * ranked_cand_res.citation_based_ranking
        mean_rank += 0.15 * ranked_cand_res.citation_count_ranking
        mean_rank += 0.35 * ranked_cand_res.keyword_based_ranking
        cand_mean_rank_dict[ranked_cand_res] = mean_rank
    return cand_mean_rank_dict


def get_ranked_resources(
    target_resources,
    existing_resources,
    resource_filter,
    resource_database_ids,
    keyword_model
):
    """
    Calls research database APIs and recommends lists of academic resources
    based on target resources, sorted according to their scores.
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
    # Extract the complete list of authors from target resources.
    target_authors = []
    for resource in target_resources:
        if resource.authors is not None:
            target_authors += resource.authors

    # Collect candidate resources from resource databases.
    candidate_resources: list[Resource] = []
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

    # Assign a recommendation score for every candidate resource.
    candidate_mean_rank_dict = _get_candidate_mean_rank_dict(
        candidate_resources=candidate_resources,
        target_resources=target_resources,
        target_keywords=target_keywords,
        keyword_model=keyword_model
    )
    # Make the mean rank the primary key used for sorting.
    candidate_mean_rank_dict = [
        (s, c) for c, s in candidate_mean_rank_dict.items()
    ]
    # Sort the list of candidates in order of their scores.
    # Only include existing resources if they appear in the top half of all...
    # ... the ranked resources, i.e. is genuinely related to the targets.
    existing_included_th = math.floor(0.5 * len(candidate_mean_rank_dict))
    ranked_existing_resources = [
        c for s, c in sorted(candidate_mean_rank_dict)[:existing_included_th] if
        c in existing_resources
    ]
    ranked_database_resources = [
        c for s, c in sorted(candidate_mean_rank_dict) if
        c not in existing_resources
    ]

    return ranked_existing_resources, ranked_database_resources


if __name__ == "__main__":
    keyword_model = KeywordRanker.get_model()
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
        print(None)
    else:
        for i, reference in enumerate(target_resource.references):
            print(f"\t[{i}]: {reference.title}")
    print(f"recommend: Time taken to execute: {t2 - t1} seconds")

    print("\nrecommend: Generate a recommendation list for a resource")

    t1 = time.time()
    # TODO: Test existing and citation resources.
    _, ranked_resources = get_ranked_resources(
        [target_resource],
        [],
        ResourceFilter({
            # "authors": ["Vijay Badrinarayanan"]
        }),
        [],
        keyword_model
    )
    t2 = time.time()
    print(f"recommend: ranked_resources: {len(ranked_resources)}")
    for i, ranked_res in enumerate(ranked_resources):
        print(f"{i}: {ranked_res.title}")
        print(f"\t{ranked_res.authors}")
        print(f"\t{ranked_res.year}")
        print(f"\t{ranked_res.doi}")
        print(f"\t{ranked_res.url}")
        print(f"\tAuthor-based ranking:   {ranked_res.author_based_ranking}")
        print(f"\tCitation-based ranking: {ranked_res.citation_based_ranking}")
        print(f"\tCitation count ranking: {ranked_res.citation_count_ranking}")
        print(f"\tKeyword-based ranking:  {ranked_res.keyword_based_ranking}")
    print(f"recommend: Time taken to execute: {t2 - t1} seconds")
