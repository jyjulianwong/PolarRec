"""
Objects and methods used to calculate the classification accuracy of the
recommendation algorithm.
"""
import unicodedata
import urllib
import xmltodict
from models.custom_logger import log
from models.evaluation import sample_resources as sr
from models.recommend import get_recommended_resources
from models.resource import Resource
from models.resource_filter import ResourceFilter
from models.resource_rankers.keyword_ranker import KeywordRanker


def _get_resource_category_dict(resources):
    """
    :param resources: The resources to collect subject categories for.
    :type resources: list[Resource]
    :return: The mapping between resources and their subject categories.
    :rtype: dict[Resource, None | list[str]]
    """
    # The mapping between resources and their subject categories.
    res_cat_dict: dict[Resource, list[str]] = {}

    for resource in resources:
        # A simplified version of the code used in ArxivQueryBuilder,
        # and the QueryBuilder interface.
        query_args = {
            "search_query": f"ti:{resource.title.replace(' ', '+')}",
            "start": 0,
            "max_results": 2
        }

        url = "http://export.arxiv.org/api/query?" + "&".join(
            [f"{key}={val}" for key, val in query_args.items()]
        )
        url = unicodedata.normalize("NFKD", url)
        url = url.encode("ascii", "ignore").decode("ascii")
        try:
            res = urllib.request.urlopen(url)
            res = res.read().decode("utf-8")
            res = xmltodict.parse(res)
        except Exception as err:
            log(str(err), "evaluation.classification", "error")
            res_cat_dict[resource] = None
            continue

        log(f"Successful response from {url}", "evaluation.classification")

        if res["feed"]["opensearch:totalResults"]["#text"] == "0":
            res_cat_dict[resource] = None
            continue

        if isinstance(res["feed"]["entry"], list):
            resource_data = res["feed"]["entry"][0]
        else:
            resource_data = res["feed"]["entry"]

        if not isinstance(resource_data, dict):
            log(
                f"Expected type 'dict' from result returned, got '{type(resource_data)}' instead: {resource_data}",
                "evaluation.classification",
                "error"
            )
            res_cat_dict[resource] = None
            continue

        # Extract the subject category from the API response data.
        if "category" in resource_data:
            if isinstance(resource_data["category"], list):
                # The resource has been assigned multiple subject categories.
                res_cat_dict[resource] = []
                for category_data in resource_data["category"]:
                    res_cat_dict[resource].append(category_data["@term"])
            else:
                # There is a single subject category for this resource.
                res_cat_dict[resource] = [resource_data["category"]["@term"]]
            continue

    return res_cat_dict


def get_classif_accuracy(target_resource, keyword_model):
    """
    Macro-accuracy considers candidate resources that match the target's wider
    subject area as being a correct classification.
    Micro-accuracy only considers candidate resources that match the target's
    exact specific subject area as being a correct classification.
    For example, the difference between ``cs`` (computer science in general) and
    ``cs.AI`` (artificial intelligence-specific resources).

    :param target_resource: The target sample resource.
    :type target_resource: Resource
    :param keyword_model: The word embedding model to be used for keywords.
    :type keyword_model: Word2Vec.KeyedVectors
    :return: The classification accuracy of the recommendation algorithm.
    :rtype: tuple[float, float]
    """
    macro_hit_count = 0
    micro_hit_count = 0
    macro_miss_count = 0
    micro_miss_count = 0

    _, reco_database_ress = get_recommended_resources(
        target_resources=[target_resource],
        existing_resources=[],
        resource_filter=ResourceFilter({}),
        resource_database_ids=[],
        keyword_model=keyword_model
    )
    # Only consider the top 10 resources returned by the algorithm.
    reco_database_ress = reco_database_ress[:min(len(reco_database_ress), 10)]

    # Collect micro-categories for resources.
    res_micro_cat_dict = _get_resource_category_dict(
        reco_database_ress + [target_resource]
    )

    # Derive macro-categories from micro-categories.
    res_macro_cat_dict: dict[Resource, list[str]] = {}
    for resource, categories in res_micro_cat_dict.items():
        if categories is None:
            res_macro_cat_dict[resource] = None
            continue

        # For example, extract "cs" from the "cs.AI" category string.
        macro_categories = [c.split(".")[0] for c in categories]
        res_macro_cat_dict[resource] = macro_categories

    for reco_resource in reco_database_ress:
        # Calculate micro-accuracy.
        pred_cats = res_micro_cat_dict[reco_resource]
        true_cats = res_micro_cat_dict[target_resource]
        if pred_cats is None or true_cats is None:
            # A NoneType indicates no category could be found for this resource.
            # This should not be included in our calculation.
            continue

        common_cats = list(set(pred_cats).intersection(true_cats))
        if len(common_cats) > 0:
            # The recommended resource belongs in the same subject category.
            micro_hit_count += 1
        else:
            # The recommended resource belongs in a different subject category.
            micro_miss_count += 1

        # Calculate macro-accuracy.
        pred_cats = res_macro_cat_dict[reco_resource]
        true_cats = res_macro_cat_dict[target_resource]
        if pred_cats is None or true_cats is None:
            # A NoneType indicates no category could be found for this resource.
            # This should not be included in our calculation.
            continue

        common_cats = list(set(pred_cats).intersection(true_cats))
        if len(common_cats) > 0:
            # The recommended resource belongs in the same subject category.
            macro_hit_count += 1
        else:
            # The recommended resource belongs in a different subject category.
            macro_miss_count += 1

    macro_accuracy = macro_hit_count / (macro_hit_count + macro_miss_count)
    micro_accuracy = micro_hit_count / (micro_hit_count + micro_miss_count)
    return macro_accuracy, micro_accuracy


if __name__ == "__main__":
    # Run this .py file to calculate the system's classification accuracy.
    # Use the ArXiv samples because we need subject area classification data.
    sample_resources = sr.load_resources_from_json()[sr.ARXIV_SAMPLE_FILEPATH]
    keyword_model = KeywordRanker.get_model()

    macro_cas: list[float] = []
    micro_cas: list[float] = []
    for i, resource in enumerate(sample_resources):
        macro_ca, micro_ca = get_classif_accuracy(resource, keyword_model)
        macro_cas.append(macro_ca)
        micro_cas.append(micro_ca)
        print(f"Macro-classification accuracy {i}: {macro_ca}")
        print(f"Micro-classification accuracy {i}: {micro_ca}")

    mean_macro_ca = sum(macro_cas) / len(macro_cas)
    mean_micro_ca = sum(micro_cas) / len(micro_cas)
    print(f"Mean macro-classification accuracy: {mean_macro_ca}")
    print(f"Mean micro-classification accuracy: {mean_micro_ca}")
