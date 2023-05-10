"""
Objects and methods for the /recommend service of the API.
"""
import time
from models import keywords
from models.resource import Resource
from models.resource_database_adapters.adapter import QueryBuilder
from models.resource_database_adapters.arxiv_adapter import ArxivQueryBuilder
from models.resource_database_adapters.ieee_xplore_adapter import \
    IEEEXploreQueryBuilder
from models.citation_database_adapters.adapter import Adapter
from models.citation_database_adapters.crossref_adapter import CrossrefAdapter
from models.citation_database_adapters.s2ag_adapter import S2agAdapter
from models.resource_filter import ResourceFilter

# Recommendation ranking-related global hyperparameters.
# Max. number of keywords to extract from targets to generate queries.
MAX_TARGET_KEYWORDS_COUNT = 10
# Max. number of keywords to extract from candidates for keyword comparison.
MAX_COMPARISON_KEYWORDS_COUNT = 20
# Max. number of candidates to be returned by each query made.
MAX_CANDIDATES_RETURNED = 100


def get_res_db_query_builders():
    """
    :return: The list of resource databases the recommendation algorithm calls.
    :rtype: list[type[QueryBuilder]]
    """
    return [ArxivQueryBuilder, IEEEXploreQueryBuilder]


def get_cit_db_adapters():
    """
    :return: The list of citation databases the recommendation algorithm calls.
    :rtype: list[type[Adapter]]
    """
    return [CrossrefAdapter, S2agAdapter]


def get_candidate_resources(target_keywords, target_authors, query_builder):
    """
    :param target_keywords: The list of keywords from the target resources.
    :type target_keywords: list[str]
    :param target_authors: The list of authors from the target resources.
    :type target_authors: list[str]
    :param query_builder: The resource database to call.
    :type query_builder: type[QueryBuilder]
    :return: A list of candidate resources based on the target resource.
    :rtype: list[Resource]
    """
    candidates: list[Resource] = []

    # Query 1: The keyword-based query.
    if len(target_keywords) > 0:
        base_query_builder = query_builder()
        base_query_builder.add_keywords(target_keywords[:min(
            len(target_keywords), MAX_TARGET_KEYWORDS_COUNT
        )])
        candidates += base_query_builder.get_resources(
            MAX_CANDIDATES_RETURNED,
            must_have_all_fields=False
        )

    # Query 2: The author-biased query.
    if len(target_authors) > 0:
        author_biased_query_builder = query_builder()
        author_biased_query_builder.set_authors(target_authors)
        candidates += author_biased_query_builder.get_resources(
            MAX_CANDIDATES_RETURNED,
            must_have_all_fields=False
        )

    return candidates


def get_candidate_score(candidate_resource, target_keywords):
    """
    :param candidate_resource: The candidate resource.
    :type candidate_resource: Resource
    :param target_keywords: The list of keywords from the target resource.
    :type target_keywords: list[str]
    :return: The recommendation score for the candidate resource.
    :rtype: float
    """
    candidate_keywords = keywords.get_keywords_from_resources(
        [candidate_resource]
    )
    similarity = keywords.keywords_similarity(
        keywords_model,
        target_keywords[:min(
            len(target_keywords), MAX_COMPARISON_KEYWORDS_COUNT
        )],
        candidate_keywords[:min(
            len(candidate_keywords), MAX_COMPARISON_KEYWORDS_COUNT
        )]
    )
    return similarity


def get_related_resources(
    target_resources,
    existing_related_resources,
    resource_filter,
    keywords_model
):
    """
    Calls research database APIs and recommends a list of academic resources
    based on target resources.

    :param target_resources: The list of target resources.
    :type target_resources: list[Resource]
    :param existing_related_resources: The list of existing related resources.
    :type existing_related_resources: list[Resource]
    :param resource_filter: The filter to be applied to the results.
    :type resource_filter: ResourceFilter
    :param keywords_model: The word embedding model to be used for keywords.
    :type keywords_model: Word2Vec.KeyedVectors
    :return: A list of recommended academic resources.
    :rtype: list[Resource]
    """
    # TODO: Implement scoring system based on keywords, authors and citations?
    # Extract keywords from target resources.
    target_keywords = keywords.get_keywords_from_resources(target_resources)
    # Extract the complete list of authors from target resources.
    target_authors = []
    for resource in target_resources:
        if resource.authors is not None:
            target_authors += resource.authors

    # Collect candidate resources from resource databases.
    candidates: list[Resource] = []
    for query_builder in get_res_db_query_builders():
        candidates += get_candidate_resources(
            target_keywords=target_keywords,
            target_authors=target_authors,
            query_builder=query_builder
        )

    # Remove any duplicate resources from the multiple queries.
    candidates = list(dict.fromkeys(candidates))

    # Assign a recommendation score for every candidate resource.
    candidate_scores: list[float] = []
    for candidate in candidates:
        candidate_score = get_candidate_score(
            candidate_resource=candidate,
            target_keywords=target_keywords
        )
        candidate_scores.append(candidate_score)

    # Sort the list of candidates in order of their scores.
    candidates = [c for s, c in sorted(zip(candidate_scores, candidates))]
    return candidates


if __name__ == "__main__":
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
    keywords_model = keywords.get_model()

    t1 = time.time()
    related_resources = get_related_resources(
        [target_resource],
        [],
        ResourceFilter({}),
        keywords_model
    )
    t2 = time.time()
    print(f"recommend: Time taken to execute: {t2 - t1} seconds")
    print(f"recommend: related_resources: {len(related_resources)}")
    for i, related_resource in enumerate(related_resources):
        print(f"{i}: \t{related_resource.title}")
        print(f"\t{related_resource.authors}")
        print(f"\t{related_resource.year}")
        print(f"\t{related_resource.doi}")
        print(f"\t{related_resource.url}")
