"""
Objects and methods for ranking academic resources based on keywords.
"""
import nltk
import time
from gensim import downloader
from models.custom_logger import log, log_extended_line
from models.hyperparams import Hyperparams as hp
from models.resource import RankableResource, Resource
from models.resource_rankers.ranker import Ranker
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer


class KeywordRanker(Ranker):
    @classmethod
    def _get_keywords(cls, text):
        """
        Extracts a list of keywords from text using the TD-IDF algorithm.

        :param text: The text from which keywords are extracted.
        :type text: str
        :return: A list of keywords, sorted in order of importance.
        :rtype: list[str]
        """
        # Custom tokenizer used to whitelist certain characters such as hyphens.
        vectorizer = TfidfVectorizer(
            stop_words=stopwords.words("english"),
            token_pattern=r"(?u)\b\w[\w-]*\w\b",
            smooth_idf=False
        )
        tfidf_mat = vectorizer.fit_transform([text])
        words = vectorizer.get_feature_names_out()
        tdidf_coo = tfidf_mat.tocoo()
        tfidf_zip = zip(tdidf_coo.col, tdidf_coo.data)
        tfidf_zip = sorted(tfidf_zip, key=lambda x: (x[1], x[0]), reverse=True)
        keywords = [words[i] for i, score in tfidf_zip]
        return keywords

    @classmethod
    def _get_keyword_list_similarity(cls, model, l1, l2, show_oovs=False):
        """
        Returns the average cosine-based similarity of two lists of keywords.
        Used as a metric for how similar two lists of keywords are.

        :param model: The keywords word embedding model.
        :type model: gensim.models.KeyedVectors
        :param l1: A list of keywords.
        :type l1: list[str]
        :param l2: A list of keywords.
        :type l2: list[str]
        :param show_oovs: Prints a list of the keywords that are out-of-vocabulary.
        :type show_oovs: bool
        :return: The average cosine-based similarity of two lists of keywords.
        :rtype: float
        """
        if len(l1) == 0 or len(l2) == 0:
            return 0.0

        similarities = []
        oovs = []
        for w1 in l1:
            if w1 in model.key_to_index:
                for w2 in l2:
                    if w2 in model.key_to_index:
                        similarities.append(model.similarity(w1, w2))
                    else:
                        oovs.append(w2)
            else:
                oovs.append(w1)

        if show_oovs:
            log(
                "_get_keyword_list_similarity: Listing out-of-vocabulary words…",
                "KeywordRanker"
            )
            for i, oov in enumerate(list(set(oovs))):
                log_extended_line(f"[{i}]: {oov}")

        return sum(similarities) / max(len(similarities), 1)

    @classmethod
    def get_model(cls):
        """
        Loads the word embedding model remotely via the Gensim API.
        This must be run before the Flask application starts. Otherwise, NLTK
        does not know where to find the stopwords corpus.

        :return: The keyword model
        :rtype: gensim.models.KeyedVectors
        """
        log("NLTK Stopwords corpus download started", "KeywordRanker")
        nltk.download("stopwords")
        log("NLTK Stopwords corpus download completed", "KeywordRanker")

        log("GloVe model download via Gensim started", "KeywordRanker")
        model = downloader.load("glove-wiki-gigaword-50")
        log("GloVe model download via Gensim completed", "KeywordRanker")

        return model

    @classmethod
    def get_keywords(cls, resources):
        """
        :param resources: The list of resources from which keywords are extracted.
        :type resources: list[Resource]
        :return: A list of keywords, sorted in order of importance.
        :rtype: list[str]
        """
        summary = ""
        for res in resources:
            title = res.title if res.title is not None else ""
            abstract = res.abstract if res.abstract is not None else ""
            summary += f"{title} {abstract} "
        keywords = cls._get_keywords(summary)
        return keywords

    @classmethod
    def set_ranking_for_resources(
        cls,
        rankable_resources,
        target_resources,
        **kwargs
    ):
        """
        Ranks candidate resources from best to worst according to how well their
        keyword lists match the targets' keyword lists.
        This function requires 2 additional keyword arguments:
            ``model: gensim.models.KeyedVectors``,
            ``target_keywords: list[str]``.

        :param rankable_resources: The list of resources to rank.
        :type rankable_resources: list[RankableResource]
        :param target_resources: The list of target resources to base ranking on.
        :type target_resources: list[Resource]
        """
        # Extract additional required keyword arguments.
        if "model" in kwargs:
            model = kwargs["model"]
        else:
            model = cls.get_model()
        if "target_keywords" in kwargs:
            target_keywords = kwargs["target_keywords"]
        else:
            target_keywords = cls.get_keywords(target_resources)

        sim_dict: dict[RankableResource, float] = {}
        for candidate_resource in rankable_resources:
            candidate_keywords = cls.get_keywords(
                [candidate_resource]
            )
            similarity = cls._get_keyword_list_similarity(
                model,
                target_keywords[:min(
                    len(target_keywords), hp.MAX_SIM_COMPAR_KEYWORDS_USED
                )],
                candidate_keywords[:min(
                    len(candidate_keywords), hp.MAX_SIM_COMPAR_KEYWORDS_USED
                )]
            )
            sim_dict[candidate_resource] = similarity

        sorted_ress = [(s, c) for c, s in sim_dict.items()]
        sorted_ress = sorted(sorted_ress, reverse=True)
        sorted_ress = [c for s, c in sorted_ress]

        # Assign the ranking position for each Resource object.
        for i, res in enumerate(sorted_ress):
            res.keyword_based_ranking = i + 1


if __name__ == '__main__':
    abstract1 = """The 'computable' numbers may be described briefly as the real
numbers whose expressions as a decimal are calculable by finite means. Although 
the subject of this paper is ostensibly the computable numbers. It is almost 
equally easy to define and investigate computable functions of an integral 
variable or a real or computable variable, computable predicates, and so forth. 
The fundamental problems involved are, however, the same in each case, and I 
have chosen the computable numbers for explicit treatment as involving the least
cumbrous technique. I hope shortly to give an account of the relations of the 
computable numbers, functions, and so forth to one another. This will include a 
development of the theory of functions of a real variable expressed in terms of 
computable numbers. According to my definition, a number is computable if its 
decimal can be written down by a machine."""
    abstract2 = """We present a novel and practical deep fully convolutional
neural network architecture for semantic pixel-wise segmentation termed SegNet. 
This core trainable segmentation engine consists of an encoder network, a 
corresponding decoder network followed by a pixel-wise classification layer. The
architecture of the encoder network is topologically identical to the 13 
convolutional layers in the VGG16 network. The role of the decoder network is to
map the low resolution encoder feature maps to full input resolution feature 
maps for pixel-wise classification. The novelty of SegNet lies is in the manner 
in which the decoder upsamples its lower resolution input feature map(s). 
Specifically, the decoder uses pooling indices computed in the max-pooling step 
of the corresponding encoder to perform non-linear upsampling. This eliminates 
the need for learning to upsample. The upsampled maps are sparse and are then 
convolved with trainable filters to produce dense feature maps. We compare our 
proposed architecture with the widely adopted FCN and also with the well known 
DeepLab-LargeFOV, DeconvNet architectures. This comparison reveals the memory 
versus accuracy trade-off involved in achieving good segmentation performance.
SegNet was primarily motivated by scene understanding applications. Hence, it is
designed to be efficient both in terms of memory and computational time during 
inference. It is also significantly smaller in the number of trainable 
parameters than other competing architectures. We also performed a controlled 
benchmark of SegNet and other architectures on both road scenes and SUN RGB-D 
indoor scene segmentation tasks. We show that SegNet provides good performance 
with competitive inference time and more efficient inference memory-wise as 
compared to other architectures. We also provide a Caffe implementation of 
SegNet and a web demo at this http URL."""

    t1 = time.time()
    model = KeywordRanker.get_model()
    t2 = time.time()
    print(f"keyword_ranker: model: {model}")
    print(f"keyword_ranker: Time taken to execute: {t2 - t1} seconds")

    t1 = time.time()
    keywords1 = KeywordRanker._get_keywords(abstract1)
    keywords2 = KeywordRanker._get_keywords(abstract2)
    t2 = time.time()
    print(f"keyword_ranker: keywords1: {keywords1[:10]}")
    print(f"keyword_ranker: keywords2: {keywords2[:10]}")
    print(f"keyword_ranker: Time taken to execute: {t2 - t1} seconds")

    resource1 = Resource({"title": "r1", "abstract": abstract1})
    resource2 = Resource({"title": "r2", "abstract": abstract2})
    t1 = time.time()
    combined_keywords = KeywordRanker.get_keywords(
        [resource1, resource2]
    )
    t2 = time.time()
    print(f"keyword_ranker: combined_keywords: {combined_keywords[:10]}")
    print(f"keyword_ranker: Time taken to execute: {t2 - t1} seconds")

    t1 = time.time()
    similarity = KeywordRanker._get_keyword_list_similarity(
        model,
        keywords1[:40],
        keywords2[:40],
        show_oovs=True
    )
    t2 = time.time()
    print(f"keyword_ranker: similarity: 40: {similarity}")
    print(f"keyword_ranker: Time taken to execute: {t2 - t1} seconds")

    t1 = time.time()
    similarity = KeywordRanker._get_keyword_list_similarity(
        model,
        keywords1[:20],
        keywords2[:20],
        show_oovs=True
    )
    t2 = time.time()
    print(f"keyword_ranker: similarity: 20: {similarity}")
    print(f"keyword_ranker: Time taken to execute: {t2 - t1} seconds")

    t1 = time.time()
    similarity = KeywordRanker._get_keyword_list_similarity(
        model,
        keywords1[:10],
        keywords2[:10],
        show_oovs=True
    )
    t2 = time.time()
    print(f"keyword_ranker: similarity: 10: {similarity}")
    print(f"keyword_ranker: Time taken to execute: {t2 - t1} seconds")
