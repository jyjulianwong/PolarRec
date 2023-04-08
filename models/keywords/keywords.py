"""
Objects and methods for ranking academic resources based on keywords.
"""
import os
import time
from gensim.models.keyedvectors import load_word2vec_format
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer


def get_model():
    return load_word2vec_format(
        os.path.join(
            os.path.dirname(__file__),
            "GoogleNews-vectors-negative300.bin"
        ),
        binary=True
    )


def get_keywords(text):
    """
    Extracts a list of keywords from text using the TextRank algorithm.

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


def keywords_similarity(model, l1, l2):
    """
    Returns the average cosine-based similarity of two lists of keywords.
    Used as a metric for how similar two lists of keywords are.

    :param model:
    :type model: Word2Vec.model
    :param l1:
    :type l1: list[str]
    :param l2:
    :type l2: list[str]
    :return:
    :rtype: float
    """
    if len(l1) == 0 or len(l2) == 0:
        return 0.0

    similarities = []
    for w1 in l1:
        for w2 in l2:
            if w1 in model.key_to_index and w2 in model.key_to_index:
                similarities.append(model.similarity(w1, w2))
    return sum(similarities) / max(len(similarities), 1)


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
    keywords1 = get_keywords(abstract1)
    keywords2 = get_keywords(abstract2)
    t2 = time.time()
    print(f"keywords: Time taken to execute: {t2 - t1} seconds")
    print(f"keywords: keywords1: {keywords1[:10]}")
    print(f"keywords: keywords2: {keywords2[:10]}")

    t1 = time.time()
    model = get_model()
    t2 = time.time()
    print(f"keywords: Time taken to execute: {t2 - t1} seconds")
    print(f"keywords: model: {model}")

    t1 = time.time()
    similarity = keywords_similarity(model, keywords1[:20], keywords2[:20])
    t2 = time.time()
    print(f"keywords: Time taken to execute: {t2 - t1} seconds")
    print(f"keywords: similarity: 20: {similarity}")

    t1 = time.time()
    similarity = keywords_similarity(model, keywords1[:10], keywords2[:10])
    t2 = time.time()
    print(f"keywords: Time taken to execute: {t2 - t1} seconds")
    print(f"keywords: similarity: 10: {similarity}")

    t1 = time.time()
    similarity = keywords_similarity(model, keywords1[:5], keywords2[:5])
    t2 = time.time()
    print(f"keywords: Time taken to execute: {t2 - t1} seconds")
    print(f"keywords: similarity: 5: {similarity}")
