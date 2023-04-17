"""
Objects and methods for the /recommend service of the API.
"""
import time
from models import keywords
from models.research_database_adapters.arxiv_adapter import ArxivQueryBuilder
from models.resource import Resource


def get_related_resources(resources, keywords_model):
    """
    Calls research database APIs and recommends a list of academic resources
    based on target resources.

    :param resources: The list of target resources.
    :type resources: list[Resource]
    :param keywords_model: The word embedding model to be used for keywords.
    :type keywords_model: Word2Vec.KeyedVectors
    :return: A list of recommended academic resources.
    :rtype: list[Resource]
    """
    # Max. number of keywords to extract from target to generate queries.
    max_compulsory_target_keywords_count = 3
    max_optional_target_keywords_count = 10
    # Max. number of keywords to extract from candidates for keyword comparison.
    max_comparison_keywords_count = 20
    # Max. number of candidates to be returned by each query made.
    max_candidates_returned = 20

    # TODO: Support multiple resources.
    target = resources[0]
    t_title = target.title if target.title is not None else ""
    t_abstract = target.abstract if target.abstract is not None else ""
    t_summary = f"{t_title} {t_abstract}"
    t_keywords = keywords.get_keywords(t_summary)

    candidates: list[Resource] = []

    # TODO: Support multiple databases.
    # Query 1: The default and compulsory keyword-based query.
    base_query_builder = ArxivQueryBuilder()
    base_query_builder.add_keywords(t_keywords[:min(
        len(t_keywords), max_compulsory_target_keywords_count
    )])
    candidates += base_query_builder.get_resources(max_candidates_returned)

    # Query 2: The more-relaxed keyword-based query.
    base_query_builder = ArxivQueryBuilder()
    base_query_builder.add_keywords(t_keywords[:min(
        len(t_keywords), max_optional_target_keywords_count
    )])
    candidates += base_query_builder.get_resources(
        max_candidates_returned,
        must_have_all_fields=False
    )

    # Query 3: The author-biased query (if applicable).
    if target.authors:
        author_biased_query_builder = ArxivQueryBuilder()
        author_biased_query_builder.set_authors(target.authors)
        candidates += author_biased_query_builder.get_resources(
            max_candidates_returned,
            must_have_all_fields=False
        )

    # Remove any duplicate resources from the multiple queries.
    candidates = list(set(candidates))

    c_scores: list[float] = []
    for candidate in candidates:
        c_title = candidate.title if candidate.title is not None else ""
        c_abstract = candidate.abstract if candidate.abstract is not None else ""
        c_summary = f"{c_title} {c_abstract}"
        c_keywords = keywords.get_keywords(c_summary)

        similarity = keywords.keywords_similarity(
            keywords_model,
            t_keywords[:min(
                len(t_keywords), max_comparison_keywords_count
            )],
            c_keywords[:min(
                len(t_keywords), max_comparison_keywords_count
            )]
        )
        c_scores.append(similarity)

    sorted_candidates = [c for s, c in sorted(zip(c_scores, candidates))]
    return sorted_candidates


if __name__ == "__main__":
    target_data = {
        "authors": ["Vijay Badrinarayanan", "Alex Kendall", "Roberto Cipolla"],
        "title": "SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation",
        "year": 2017,
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
    related_resources = get_related_resources([target_resource], keywords_model)
    t2 = time.time()
    print(f"recommend: Time taken to execute: {t2 - t1} seconds")
    print(f"recommend: related_resources: {len(related_resources)}")
    for i, related_resource in enumerate(related_resources):
        print(f"{i}: \t{related_resource.title}")
        print(f"\t{related_resource.authors}")
        print(f"\t{related_resource.year}")
        print(f"\t{related_resource.doi}")
        print(f"\t{related_resource.url}")
