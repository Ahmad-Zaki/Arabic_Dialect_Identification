import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from typing import Any


class ArabicTextNormalizer(BaseEstimator, TransformerMixin):
    """Normalize arabic text by a using a set of rules

    Parameters
    ----------
    alef: bool, default = False
        Replace all alef-with-hamza(أإآ) with alef(أ).

    taa: bool, default = False
        Replace taa-marbota(ة) with haa(ه).

    yaa: bool, default = False
        Replace alef-maqsura(ى) with yaa(ي).

    jeem: bool, default = False
        Replace jeem-with-3-dot(چ) with regular jeem(ج).

    faa: bool, default = False
        Replace faa-with-3-dot(ڤ) with regular faa(ف).

    kaf: bool, default = False
        Replace fancy-kaf(گ) with regular kaf(ك).

    urls: bool, default = False
        Replace any URL with the word 'URL'.

    mentions: bool, default = False
        Replace any mention with the word '@USER'.

    repetition: bool, default = False
        Limit any repeating character to a length 2.

    spaces: bool, default = False
        Replace any whitespace character('\n','\t', ' ') with a single space (' ').

    non_arabic: bool, default = False
        Remove any non-arabic letters (Digits, Special characters, URLS, Punctuation, Mentions, Emojis, diacritics).
    """

    def __init__(self, alef: bool = True, taa: bool = True, yaa: bool = True, jeem: bool = True, faa: bool = True, kaf: bool = True, urls: bool = True, mentions: bool = True, repetition: bool = True, spaces: bool = True, non_arabic: bool = True) -> None:
        self.alef = alef
        self.taa = taa
        self.yaa = yaa
        self.jeem = jeem
        self.faa = faa
        self.kaf = kaf
        self.urls = urls
        self.mentions = mentions
        self.repetition = repetition
        self.spaces = spaces
        self.non_arabic = non_arabic

    def fit(self, X: Any, y: Any = None) -> "ArabicTextNormalizer":
        return self

    def transform(self, X: Any, y: Any = None) -> Any:
        if not isinstance(X, pd.Series): X = pd.Series(X)
        X_ = X.copy()

        if self.alef: X_ = X_.str.replace(r"[أإآ]", "ا")
        if self.taa: X_ = X_.str.replace(r"[ة]", "ه")
        if self.yaa: X_ = X_.str.replace(r"[ى]", "ي")
        if self.jeem: X_ = X_.str.replace(r"[چ]", "ج")
        if self.faa: X_ = X_.str.replace(r"[ڤ]", "ف")
        if self.kaf: X_ = X_.str.replace(r"[گ]", "ك")
        if self.urls: X_ = X_.str.replace(r"https?\://\S+", "")
        if self.mentions: X_ = X_.str.replace(r"\@\S+", "")
        if self.repetition: X_ = X_.str.replace(r"(.)\1{3,}", r"\1\1")
        if self.spaces: X_ = X_.str.replace(r"\s+", " ")
        if self.non_arabic: X_ = X_.str.replace(r"[^ابتثجحخدذرزسشصضطظعغفقكلمنهوي ]+", "")

        return X_


def preprocessing_pipeline(steps: list[str], normalizer_kwarg: dict = {} ,victorizer_kwarg: dict = {}) -> Pipeline:
    """Create a preprocessing pipeline.

    Parameters
    ----------
    steps: list[str]
        A list of the desired preprocessing steps. the available steps are 'normalization' using ArabicTextNormalizer, 'bag of words' using CountVectorizer, and 'tfidf' using TfidfVectorizer.
    
    normalizer_kwarg: dict, default = {}
        Arguments that are passed to ArabicTextNormalizer, if unspecified, default arguments are used.

    victorizer_kwarg: dict, default = {}
        Arguments that are passed to CountVectorizer and TfidfVectorizer, if unspecified, default arguments are used.

    Returns
    -------
    pipeline: Pipeline
        A pipeline object that contains the specified preprocessing steps.
    """

    preprocessing_tools = {"normalization": ArabicTextNormalizer(),
                           "bag of words": CountVectorizer(**victorizer_kwarg),
                           "tfidf": TfidfVectorizer(**victorizer_kwarg)}

    pipeline_steps = [preprocessing_tools[step] for step in steps]
    pipeline = make_pipeline(*pipeline_steps)
    return pipeline


def train_val_test_split(X: Any, y: Any, stratified: bool = True, seed: int = 42) -> tuple:
    """Split the dataset into training, validation, and testing splits with (8:1:1) ratio.
    
    Parameters
    ----------
    X: array-like
        An array-like onject that contains features values.
        
    y: array-like
        An array-like object that contains the target values.

    stratify: bool
        Specify whether to do a stratified split or not.
        default = True

    seed: int
        specife the random state of spliting.
        default = 42
        
    Returns
    -------
    (X_train, X_val, X_test, y_train, y_val, y_test): tuple
        a tuple containing training, validation, and testing splits features and targets. 
    """
    
    stratify = y if stratified else None
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size = 0.1, 
                                                        stratify = stratify, 
                                                        random_state = seed)

    stratify = y_train if stratified else None
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, 
                                                      test_size = 1/9, 
                                                      stratify = stratify, 
                                                      random_state = seed) # validation ratio = 0.9 × 1/9 = 0.1

    return X_train, X_val, X_test, y_train, y_val, y_test