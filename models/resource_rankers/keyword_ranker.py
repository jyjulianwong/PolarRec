"""
Objects and methods for ranking academic resources based on keywords.
"""
import math
import nltk
import numpy as np
import pandas as pd
import pytextrank
import spacy
import subprocess
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
    def _get_keywords_from_text(
        cls,
        text,
        kw_rank_method,
        min_phrase_length=1,
        max_phrase_length=3
    ):
        """
        Extracts a list of keywords from text using either the TF-IDF algorithm
        (``"tdidf"``) or the TextRank algorithm (``"textrank"``).

        :param text: The text from which keywords are extracted.
        :type text: str
        :param kw_rank_method: The keyword extraction algorithm to use.
        :type kw_rank_method: str
        :param min_phrase_length: The min. number of words in each key phrase.
        :type min_phrase_length: int
        :param max_phrase_length: The max. number of words in each key phrase.
        :type min_phrase_length: int
        :return: A list of keywords, sorted in order of importance.
        :rtype: list[str]
        """
        # Remove any newline characters and quotation marks in the input text.
        text = text.replace("\n", " ")
        text = text.replace("\"", " ")
        text = text.replace("\'", " ")
        # Replace occurrences of multiple whitespaces with a single whitespace.
        text = " ".join(text.split())

        if kw_rank_method == "tfidf":
            # Custom tokenizer used to whitelist some characters like hyphens.
            vectorizer = TfidfVectorizer(
                stop_words=stopwords.words("english"),
                token_pattern=r"(?u)\b\w[\w-]*\w\b",
                ngram_range=(min_phrase_length, max_phrase_length),
                smooth_idf=False
            )
            tfidf_mat = vectorizer.fit_transform([text])
            words = vectorizer.get_feature_names_out()
            tdidf_coo = tfidf_mat.tocoo()
            tfidf_zip = zip(tdidf_coo.col, tdidf_coo.data)
            tfidf_zip = sorted(
                tfidf_zip,
                key=lambda x: (x[1], x[0]),
                reverse=True
            )
            keywords = [words[i] for i, score in tfidf_zip]
            return keywords

        if kw_rank_method == "textrank":
            # This code is based on the official PyTextRank documentation:
            # https://derwen.ai/docs/ptr/sample/
            @spacy.registry.misc("articles_scrubber")
            def articles_scrubber():
                """
                :rtype: function
                """

                def scrubber_func(span):
                    """
                    :type span: spacy.util.Span
                    :rtype: str
                    """
                    for token in span:
                        if token.pos_ not in ["DET", "PRON"]:
                            break
                        span = span[1:]
                    return span.text

                return scrubber_func

            # Load the pre-trained spaCy pipeline that contains a PoS tagger,
            # a lemmatiser, a parser, and an entity recogniser.
            nlp = spacy.load("en_core_web_md")
            # Add the additional TextRank pipe to the end of the pipeline.
            # TextRank uses PoS and dependency data to rank key phrases.
            nlp.add_pipe(
                "textrank",
                config={"scrubber": {"@misc": "articles_scrubber"}}
            )
            # Process the text with the full pipeline and save the results.
            doc = nlp(text)
            # Sort the key phrases by their "rank", or TextRank score.
            kw_rank_pairs = [
                (phrase.rank, phrase.text.lower()) for phrase in doc._.phrases
            ]
            kw_rank_pairs = sorted(kw_rank_pairs, reverse=True)
            keywords = [kw for rank, kw in kw_rank_pairs]
            # Only show key phrases that are within the length limit.
            keywords = [
                w for w in keywords if len(w.split(" ")) <= max_phrase_length
            ]
            return keywords

    @classmethod
    def _get_phrase_embedding_vector(cls, model, phrase):
        """
        :param model: The keywords word embedding model.
        :type model: gensim.models.KeyedVectors
        :param phrase: The phrase or n-gram.
        :type phrase: str
        :param show_oovs: Print list of the keywords that are out-of-vocabulary.
        :type show_oovs: bool
        :return: The embedding vector for a phrase, calculated with vector mean.
        :rtype: tuple[(None | np.ndarray), list[str]]
        """
        word_vecs = []
        oovs = []
        for word in phrase.split(" "):
            if word in model.key_to_index:
                word_vecs.append(model.get_vector(word))
            else:
                oovs.append(word)

        if len(oovs) == 0:
            word_vecs = np.array(word_vecs)
            phrase_vector = np.mean(word_vecs, axis=0)
            return phrase_vector, list(set(oovs))
        else:
            # It is not a representative embedding if some words not present.
            return None, list(set(oovs))

    @classmethod
    def _get_phrase_similarity(cls, model, p1, p2):
        """
        :param model: The keywords word embedding model.
        :type model: gensim.models.KeyedVectors
        :param p1: The phrase or n-gram.
        :type p1: str
        :param p2: The phrase or n-gram.
        :type p2: str
        :param show_oovs: Print list of the keywords that are out-of-vocabulary.
        :type show_oovs: bool
        :return: The cosine-based similarity of two phrases or n-grams.
        :rtype: tuple[(None | float), list[str]]
        """
        vec1, oovs1 = cls._get_phrase_embedding_vector(model, p1)
        vec2, oovs2 = cls._get_phrase_embedding_vector(model, p2)
        if vec1 is None or vec2 is None:
            return None, list(set(oovs1 + oovs2))

        num = np.dot(vec1, vec2)
        den = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        # Set minimum denominator to avoid division-by-zero.
        return num / max(den, 1e-9), list(set(oovs1 + oovs2))

    @classmethod
    def _get_keyword_list_similarity(
        cls,
        model,
        l1,
        l2,
        show_oovs=False,
        show_kw_sim_mat=False
    ):
        """
        Returns the average pair-wise cosine-based similarity of two lists of
        keywords.
        Used as a metric for how similar two lists of keywords are.
        Assumes that l1 and l2 are ranked lists of keywords, with keywords that
        are more important appearing first in the list.
        This comparison algorithm is mono-directional. It calculates how well
        the keywords in l2 match those in l1.

        :param model: The keywords word embedding model.
        :type model: gensim.models.KeyedVectors
        :param l1: The target list of keywords.
        :type l1: list[str]
        :param l2: The candidate list of keywords.
        :type l2: list[str]
        :param show_oovs: Print list of the keywords that are out-of-vocabulary.
        :type show_oovs: bool
        :param show_kw_sim_mat: Print the pair-wise keyword similarity matrix.
        :type show_kw_sim_mat: bool
        :return: The average cosine-based similarity of two lists of keywords.
        :rtype: float
        """
        if len(l1) == 0 or len(l2) == 0:
            return 0.0

        p1s = []
        p2s = []
        weights = []
        psims = []
        wsims = []
        oovs = []
        for i1, p1 in enumerate(l1):
            if p1 in l2:
                # Both lists contain this exact phrase. The similarity is 1.0.
                i2 = l2.index(p1)
                p1s.append(p1)
                p2s.append(p1)
                # TODO: Remove hard-coded lambda parameter.
                # Calculate weight using the negative exponential distribution.
                w1 = 0.2 * math.e ** (-0.2 * i1)
                w2 = 0.2 * math.e ** (-0.2 * i2)
                weights.append(w1 * w2)
                psims.append(1.0)
                continue

            p2_i2_dict: dict[str, int] = {}
            p2_psim_dict: dict[str, float] = {}
            for i2, p2 in enumerate(l2):
                # Collect similarity scores for every p1-p2 pair.
                psim, some_oovs = cls._get_phrase_similarity(model, p1, p2)
                if psim is not None:
                    p2_i2_dict[p2] = i2
                    p2_psim_dict[p2] = psim
                oovs += some_oovs

            if len(p2_psim_dict) == 0:
                # This should not happen in normal circumstances.
                continue

            # Find the phrase from l2 that is most similar to p1.
            p2_psim_list = [(s, p) for p, s in p2_psim_dict.items()]
            p2_psim_list = sorted(p2_psim_list, reverse=True)
            p2 = p2_psim_list[0][1]
            i2 = p2_i2_dict[p2]
            psim = p2_psim_list[0][0]

            # TODO: Remove hard-coded lambda parameter.
            # Calculate weight using the negative exponential distribution.
            w1 = 0.2 * math.e ** (-0.2 * i1)
            w2 = 0.2 * math.e ** (-0.2 * i2)
            p1s.append(p1)
            p2s.append(p2)
            weights.append(w1 * w2)
            psims.append(psim)

        # The sum of all the weights should be 1.
        weights = [w / max(sum(weights), 1e-9) for w in weights]
        # Add weighting to each similarity score to indicate their "importance".
        wsims = [weights[i] * psims[i] for i in range(len(psims))]

        if show_oovs and len(oovs) > 0:
            log(
                "_get_keyword_list_similarity: Listing out-of-vocabulary words…",
                "KeywordRanker"
            )
            for i, oov in enumerate(list(set(oovs))):
                log_extended_line(f"[{i}]: {oov}")

        if show_kw_sim_mat:
            log(
                "_get_keyword_list_similarity: Showing pair-wise keyword similarity matrix…",
                "KeywordRanker"
            )
            pd.set_option("display.min_rows", 20)
            pd.set_option("display.max_columns", None)
            pd.set_option("expand_frame_repr", False)
            kw_sim_df = pd.DataFrame()
            kw_sim_df["p1"] = p1s
            kw_sim_df["p2"] = p2s
            kw_sim_df["weight"] = weights
            kw_sim_df["psim"] = psims
            kw_sim_df["wsim"] = wsims
            print(kw_sim_df)

        return sum(wsims)

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

        log("spaCy pre-trained pipeline download started", "KeywordRanker")
        try:
            subprocess.run(
                ["spacy download en_core_web_sm"],
                shell=True,
                check=True
            )
            log(
                "spaCy pre-trained pipeline download completed",
                "KeywordRanker"
            )
        except subprocess.CalledProcessError as err:
            log(
                f"spaCy pre-trained pipeline download failed: {err}",
                "KeywordRanker",
                "error"
            )

        log("GloVe model download via Gensim started", "KeywordRanker")
        model = downloader.load("glove-wiki-gigaword-50")
        log("GloVe model download via Gensim completed", "KeywordRanker")

        return model

    @classmethod
    def get_keywords(cls, resources, kw_rank_method=hp.KEYWORD_RANK_METHOD):
        """
        :param resources: The list of resources from which keywords are extracted.
        :type resources: list[Resource]
        :param kw_rank_method: The keyword extraction algorithm to use.
        :type kw_rank_method: str
        :return: A list of keywords, sorted in order of importance.
        :rtype: list[str]
        """
        # Summarise all the target resources in a singular piece of text.
        summary = ""
        for res in resources:
            title = res.title if res.title is not None else ""
            abstract = res.abstract if res.abstract is not None else ""
            summary += f"{title} {abstract} "

        # Extract a list of ranked keywords from the summary text.
        keywords = cls._get_keywords_from_text(summary, kw_rank_method)
        return keywords

    @classmethod
    def set_resource_rankings(
        cls,
        rankable_resources,
        target_resources,
        **kwargs
    ):
        """
        Ranks candidate resources from best to worst according to how well their
        keyword lists match the targets' keyword lists.
        This function requires 1 additional keyword argument:
            ``kw_model: gensim.models.KeyedVectors``.
        This function optionally accepts 1 additional keyword argument:
            ``kw_rank_method: str``.

        :param rankable_resources: The list of resources to rank.
        :type rankable_resources: list[RankableResource]
        :param target_resources: The list of target resources to base ranking on.
        :type target_resources: list[Resource]
        """
        # Extract additional required keyword arguments.
        if "kw_model" in kwargs:
            model = kwargs["kw_model"]
        else:
            model = cls.get_model()
        if "kw_rank_method" in kwargs:
            kw_rank_method = kwargs["kw_rank_method"]
        else:
            kw_rank_method = hp.KEYWORD_RANK_METHOD

        target_keywords = cls.get_keywords(target_resources, kw_rank_method)

        sim_dict: dict[RankableResource, float] = {}
        for candidate_resource in rankable_resources:
            candidate_keywords = cls.get_keywords(
                [candidate_resource],
                kw_rank_method
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

    print("\nkeyword_ranker: Load the word embedding model")
    t1 = time.time()
    model = KeywordRanker.get_model()
    t2 = time.time()
    print(f"keyword_ranker: model: {model}")
    print(f"keyword_ranker: Time taken to execute: {t2 - t1} seconds")

    print("\nkeyword_ranker: Extract keywords from an abstract")
    t1 = time.time()
    keywords1 = KeywordRanker._get_keywords_from_text(
        abstract1,
        hp.KEYWORD_RANK_METHOD
    )
    keywords2 = KeywordRanker._get_keywords_from_text(
        abstract2,
        hp.KEYWORD_RANK_METHOD
    )
    t2 = time.time()
    print(f"keyword_ranker: keywords1:")
    for i, keyword in enumerate(keywords1[:20]):
        print(f"\t[{i}]: {keyword}")
    print(f"keyword_ranker: keywords2:")
    for i, keyword in enumerate(keywords2[:20]):
        print(f"\t[{i}]: {keyword}")
    print(f"keyword_ranker: Time taken to execute: {t2 - t1} seconds")

    print("\nkeyword_ranker: Extract keywords from two abstracts together")
    resource1 = Resource({"title": "r1", "abstract": abstract1})
    resource2 = Resource({"title": "r2", "abstract": abstract2})
    t1 = time.time()
    combined_keywords = KeywordRanker.get_keywords([resource1, resource2])
    t2 = time.time()
    print(f"keyword_ranker: combined_keywords:")
    for i, keyword in enumerate(combined_keywords[:20]):
        print(f"\t[{i}]: {keyword}")
    print(f"keyword_ranker: Time taken to execute: {t2 - t1} seconds")

    print("\nkeyword_ranker: Compare keyword lists of length 40")
    t1 = time.time()
    similarity = KeywordRanker._get_keyword_list_similarity(
        model,
        keywords1[:40],
        keywords2[:40],
        show_oovs=True,
        show_kw_sim_mat=True
    )
    t2 = time.time()
    print(f"keyword_ranker: similarity: 40: {similarity}")
    print(f"keyword_ranker: Time taken to execute: {t2 - t1} seconds")

    print("\nkeyword_ranker: Compare keyword lists of length 20")
    t1 = time.time()
    similarity = KeywordRanker._get_keyword_list_similarity(
        model,
        keywords1[:20],
        keywords2[:20],
        show_oovs=True,
        show_kw_sim_mat=True
    )
    t2 = time.time()
    print(f"keyword_ranker: similarity: 20: {similarity}")
    print(f"keyword_ranker: Time taken to execute: {t2 - t1} seconds")

    print("\nkeyword_ranker: Compare keyword lists of length 10")
    t1 = time.time()
    similarity = KeywordRanker._get_keyword_list_similarity(
        model,
        keywords1[:10],
        keywords2[:10],
        show_oovs=True,
        show_kw_sim_mat=True
    )
    t2 = time.time()
    print(f"keyword_ranker: similarity: 10: {similarity}")
    print(f"keyword_ranker: Time taken to execute: {t2 - t1} seconds")

    print("\nkeyword_ranker: Compare keyword lists that are identical")
    similarity = KeywordRanker._get_keyword_list_similarity(
        model,
        keywords1[:20],
        keywords1[:20],
        show_oovs=True,
        show_kw_sim_mat=True
    )
    print(f"keyword_ranker: similarity: Identical: {similarity}")

    print("\nkeyword_ranker: Compare keyword lists that are near-identical")
    similarity = KeywordRanker._get_keyword_list_similarity(
        model,
        ["python", "jumble", "easy", "difficult", "answer", "xylophone"],
        ["python", "jumble", "easy", "difficult", "answer", "piano"],
        show_oovs=True,
        show_kw_sim_mat=True
    )
    print(f"keyword_ranker: similarity: Near-identical: {similarity}")

    print("\nkeyword_ranker: Compare keyword lists that are similar")
    similarity = KeywordRanker._get_keyword_list_similarity(
        model,
        ["python", "jumble", "easy", "difficult", "answer", "xylophone"],
        ["java", "scramble", "simple", "hard", "response", "piano"],
        show_oovs=True,
        show_kw_sim_mat=True
    )
    print(f"keyword_ranker: similarity: Similar: {similarity}")
