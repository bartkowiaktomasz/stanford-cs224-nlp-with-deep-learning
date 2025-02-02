#!/usr/bin/env python

import argparse
from typing import Optional, Union
import numpy as np
import random

from utils.gradcheck import gradcheck_naive, grad_tests_softmax, grad_tests_negsamp
from utils.utils import normalizeRows, softmax


def sigmoid(x: Union[float, np.array]):
    """
    Compute the sigmoid function for the input here.
    Arguments:
    x -- A scalar or numpy array.
    Return:
    s -- sigmoid(x)
    """
    s = 1 / (1 + np.e**-x)
    return s


def naiveSoftmaxLossAndGradient(
    centerWordVec: np.array,
    outsideWordIdx: int,
    outsideVectors: np.array,
    dataset: Optional[np.array] = None,
):
    """Naive Softmax loss & gradient function for word2vec models

    Implement the naive softmax loss and gradients between a center word's
    embedding and an outside word's embedding. This will be the building block
    for our word2vec models. For those unfamiliar with numpy notation, note
    that a numpy ndarray with a shape of (x, ) is a one-dimensional array, which
    you can effectively treat as a vector with length x.

    Arguments:
    centerWordVec -- numpy ndarray, center word's embedding
                    in shape (word vector length, )
                    (v_c in the pdf handout)
    outsideWordIdx -- integer, the index of the outside word
                    (o of u_o in the pdf handout)
    outsideVectors -- outside vectors is
                    in shape (num words in vocab, word vector length)
                    for all words in vocab (tranpose of U in the pdf handout)
    dataset -- needed for negative sampling, unused here.

    Return:
    loss -- naive softmax loss
    gradCenterVec -- the gradient with respect to the center word vector
                     in shape (word vector length, )
                     (dJ / dv_c in the pdf handout)
    gradOutsideVecs -- the gradient with respect to all the outside word vectors
                    in shape (num words in vocab, word vector length)
                    (dJ / dU)
    """
    ### Please use the provided softmax function (imported earlier in this file)
    ### This numerically stable implementation helps you avoid issues pertaining
    ### to integer overflow.

    y = np.zeros((outsideVectors.shape[0],))
    y[outsideWordIdx] = 1
    # p(O = o | C = c))
    y_pred = softmax(outsideVectors @ centerWordVec)  # (N, E) @ (E, ) => (N, )

    loss = -np.log(y_pred[outsideWordIdx])
    # Gradient need to have the shape that allows to update the underlying parameter
    # gradCenterVec needs to be (E, ) beucause it is used to update centerWordVec
    gradCenterVec = outsideVectors.T @ (y_pred - y)  # (E, N) @ (N, ) => (E, )
    # gradOutsideVecs has to have the shape (N, E) because it is used to update outsideVectors
    gradOutsideVecs = np.outer(y_pred - y, centerWordVec)
    return loss, gradCenterVec, gradOutsideVecs


def getNegativeSamples(outsideWordIdx, dataset, K):
    """Samples K indexes which are not the outsideWordIdx"""

    negSampleWordIndices = [None] * K
    for k in range(K):
        newidx = dataset.sampleTokenIdx()
        while newidx == outsideWordIdx:
            newidx = dataset.sampleTokenIdx()
        negSampleWordIndices[k] = newidx
    return negSampleWordIndices


def negSamplingLossAndGradient(
    centerWordVec, outsideWordIdx, outsideVectors, dataset, K=10  # (N, E)
):
    """Negative sampling loss function for word2vec models

    Implement the negative sampling loss and gradients for a centerWordVec
    and a outsideWordIdx word vector as a building block for word2vec
    models. K is the number of negative samples to take.

    Note: The same word may be negatively sampled multiple times. For
    example if an outside word is sampled twice, you shall have to
    double count the gradient with respect to this word. Thrice if
    it was sampled three times, and so forth.

    Arguments/Return Specifications: same as naiveSoftmaxLossAndGradient
    """

    # Negative sampling of words is done for you. Do not modify this if you
    # wish to match the autograder and receive points!
    negSampleWordIndices = getNegativeSamples(outsideWordIdx, dataset, K)
    indices = [outsideWordIdx] + negSampleWordIndices

    # initialise returned values
    gradOutsideVecs = np.zeros_like(outsideVectors)
    y_pred = sigmoid(outsideVectors[outsideWordIdx] @ centerWordVec)
    loss = -np.log(y_pred)
    gradCenterVec = np.dot((y_pred - 1), outsideVectors[outsideWordIdx])
    gradOutsideVecs[outsideWordIdx] = np.dot(y_pred - 1, centerWordVec)

    # TODO: try to remove "for" loop
    for s in range(K):
        negSampleWordIdx = negSampleWordIndices[s]
        y_pred_neg = sigmoid(-outsideVectors[negSampleWordIdx] @ centerWordVec)
        loss += -np.log(y_pred_neg)
        gradCenterVec += np.dot(1 - y_pred_neg, outsideVectors[negSampleWordIdx])
        gradOutsideVecs[negSampleWordIdx] += np.dot(1 - y_pred_neg, centerWordVec)
    return loss, gradCenterVec, gradOutsideVecs


def skipgram(
    currentCenterWord,
    windowSize,
    outsideWords,
    word2Ind,
    centerWordVectors,
    outsideVectors,
    dataset,
    word2vecLossAndGradient=naiveSoftmaxLossAndGradient,
):
    """Skip-gram model in word2vec

    Implement the skip-gram model in this function.

    Arguments:
    currentCenterWord -- a string of the current center word
    windowSize -- integer, context window size
    outsideWords -- list of no more than 2*windowSize strings, the outside words
    word2Ind -- a dictionary that maps words to their indices in
              the word vector list
    centerWordVectors -- center word vectors (as rows) is in shape
                        (num words in vocab, word vector length)
                        for all words in vocab (V in pdf handout)
    outsideVectors -- outside vectors is in shape
                        (num words in vocab, word vector length)
                        for all words in vocab (transpose of U in the pdf handout)
    word2vecLossAndGradient -- the loss and gradient function for
                               a prediction vector given the outsideWordIdx
                               word vectors, could be one of the two
                               loss functions you implemented above.

    Return:
    loss -- the loss function value for the skip-gram model
            (J in the pdf handout)
    gradCenterVecs -- the gradient with respect to the center word vector
                     in shape (num words in vocab, word vector length)
                     (dJ / dv_c in the pdf handout)
    gradOutsideVecs -- the gradient with respect to all the outside word vectors
                    in shape (num words in vocab, word vector length)
                    (dJ / dU)
    """

    total_loss = 0.0
    total_gradCenterVecs = np.zeros(centerWordVectors.shape)
    total_gradOutsideVectors = np.zeros(outsideVectors.shape)

    ### YOUR CODE HERE (~8 Lines)
    center_word_idx = word2Ind[currentCenterWord]
    center_word_emb = centerWordVectors[center_word_idx]
    for outside_word in outsideWords:
        outside_word_idx = word2Ind[outside_word]
        loss, gradCenterVecs, gradOutsideVectors = word2vecLossAndGradient(
            centerWordVec=center_word_emb,
            outsideWordIdx=outside_word_idx,
            outsideVectors=outsideVectors,
            dataset=dataset,
        )
        total_loss += loss
        total_gradCenterVecs[center_word_idx] += gradCenterVecs
        total_gradOutsideVectors += gradOutsideVectors

    ### END YOUR CODE
    return total_loss, total_gradCenterVecs, total_gradOutsideVectors


#############################################
# Testing functions below. DO NOT MODIFY!   #
#############################################


def word2vec_sgd_wrapper(
    word2vecModel,
    word2Ind,
    wordVectors,
    dataset,
    windowSize,
    word2vecLossAndGradient=naiveSoftmaxLossAndGradient,
):
    batchsize = 50
    loss = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    centerWordVectors = wordVectors[: int(N / 2), :]
    outsideVectors = wordVectors[int(N / 2) :, :]
    for i in range(batchsize):
        windowSize1 = random.randint(1, windowSize)
        centerWord, context = dataset.getRandomContext(windowSize1)

        c, gin, gout = word2vecModel(
            centerWord,
            windowSize1,
            context,
            word2Ind,
            centerWordVectors,
            outsideVectors,
            dataset,
            word2vecLossAndGradient,
        )
        loss += c / batchsize
        grad[: int(N / 2), :] += gin / batchsize
        grad[int(N / 2) :, :] += gout / batchsize

    return loss, grad


def test_sigmoid():
    """Test sigmoid function"""
    print("=== Sanity check for sigmoid ===")
    assert sigmoid(0) == 0.5
    assert np.allclose(sigmoid(np.array([0])), np.array([0.5]))
    assert np.allclose(
        sigmoid(np.array([1, 2, 3])), np.array([0.73105858, 0.88079708, 0.95257413])
    )
    print("Tests for sigmoid passed!")


def getDummyObjects():
    """Helper method for naiveSoftmaxLossAndGradient and negSamplingLossAndGradient tests"""

    def dummySampleTokenIdx():
        return random.randint(0, 4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0, 4)], [
            tokens[random.randint(0, 4)] for i in range(2 * C)
        ]

    dataset = type("dummy", (), {})()
    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10, 3))
    dummy_tokens = dict([("a", 0), ("b", 1), ("c", 2), ("d", 3), ("e", 4)])

    return dataset, dummy_vectors, dummy_tokens


def test_naiveSoftmaxLossAndGradient():
    """Test naiveSoftmaxLossAndGradient"""
    dataset, dummy_vectors, dummy_tokens = getDummyObjects()

    print("==== Gradient check for naiveSoftmaxLossAndGradient ====")

    def temp(vec):
        loss, gradCenterVec, gradOutsideVecs = naiveSoftmaxLossAndGradient(
            vec, 1, dummy_vectors, dataset
        )
        return loss, gradCenterVec

    gradcheck_naive(
        temp, np.random.randn(3), "naiveSoftmaxLossAndGradient gradCenterVec"
    )

    centerVec = np.random.randn(3)

    def temp(vec):
        loss, gradCenterVec, gradOutsideVecs = naiveSoftmaxLossAndGradient(
            centerVec, 1, vec, dataset
        )
        return loss, gradOutsideVecs

    gradcheck_naive(temp, dummy_vectors, "naiveSoftmaxLossAndGradient gradOutsideVecs")


def test_negSamplingLossAndGradient():
    """Test negSamplingLossAndGradient"""
    dataset, dummy_vectors, dummy_tokens = getDummyObjects()

    print("==== Gradient check for negSamplingLossAndGradient ====")

    def temp(vec):
        loss, gradCenterVec, gradOutsideVecs = negSamplingLossAndGradient(
            vec, 1, dummy_vectors, dataset
        )
        return loss, gradCenterVec

    gradcheck_naive(
        temp, np.random.randn(3), "negSamplingLossAndGradient gradCenterVec"
    )

    centerVec = np.random.randn(3)

    def temp(vec):
        loss, gradCenterVec, gradOutsideVecs = negSamplingLossAndGradient(
            centerVec, 1, vec, dataset
        )
        return loss, gradOutsideVecs

    gradcheck_naive(temp, dummy_vectors, "negSamplingLossAndGradient gradOutsideVecs")


def test_skipgram():
    """Test skip-gram with naiveSoftmaxLossAndGradient"""
    dataset, dummy_vectors, dummy_tokens = getDummyObjects()

    print("==== Gradient check for skip-gram with naiveSoftmaxLossAndGradient ====")
    gradcheck_naive(
        lambda vec: word2vec_sgd_wrapper(
            skipgram, dummy_tokens, vec, dataset, 5, naiveSoftmaxLossAndGradient
        ),
        dummy_vectors,
        "naiveSoftmaxLossAndGradient Gradient",
    )
    grad_tests_softmax(skipgram, dummy_tokens, dummy_vectors, dataset)

    print("==== Gradient check for skip-gram with negSamplingLossAndGradient ====")
    gradcheck_naive(
        lambda vec: word2vec_sgd_wrapper(
            skipgram, dummy_tokens, vec, dataset, 5, negSamplingLossAndGradient
        ),
        dummy_vectors,
        "negSamplingLossAndGradient Gradient",
    )
    grad_tests_negsamp(
        skipgram, dummy_tokens, dummy_vectors, dataset, negSamplingLossAndGradient
    )


def test_word2vec():
    """Test the two word2vec implementations, before running on Stanford Sentiment Treebank"""
    test_sigmoid()
    test_naiveSoftmaxLossAndGradient()
    test_negSamplingLossAndGradient()
    test_skipgram()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test your implementations.")
    parser.add_argument(
        "function",
        nargs="?",
        type=str,
        default="all",
        help="Name of the function you would like to test.",
    )

    args = parser.parse_args()
    if args.function == "sigmoid":
        test_sigmoid()
    elif args.function == "naiveSoftmaxLossAndGradient":
        test_naiveSoftmaxLossAndGradient()
    elif args.function == "negSamplingLossAndGradient":
        test_negSamplingLossAndGradient()
    elif args.function == "skipgram":
        test_skipgram()
    elif args.function == "all":
        test_word2vec()
