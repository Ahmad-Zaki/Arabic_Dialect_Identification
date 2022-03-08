from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from typing import Any


class ArabicTextNormalizer(BaseEstimator, TransformerMixin):
    """Normalize arabic text by a using a set of rules:
    - Replace taa-marbota(ة) with haa(ه).
    - Replace all alef-with-hamza(أإآ) with alef(أ).
    - Replace alef-maqsura(ى) with yaa(ي).
    - Replace jeem-with-3-dot(چ) with regular jeem(ج).
    - Replace faa-with-3-dot(ڤ) with regular faa(ف).
    - Replace fancy-kaf(گ) with regular kaf(ك).
    - Limit any repeating character to a length 2.
    - Remove mentions, URLs, and hashtags.
    - Limit consecutive whitespace characters to 1.
    - Remove any non-arabic letters (Digits, Special characters, URLS,Punctuation, Mentions, Emojis, diacritics).
    """

    def fit(self, X: Any, y: Any = None) -> "ArabicTextNormalizer":
        return self

    def transform(self, X: Any, y: Any = None) -> Any:
        X_ = X.copy()
        X_ = (X_.str.replace(r"[ة]", "ه")
                .str.replace(r"[أإآ]", "ا")
                .str.replace(r"[ى]", "ي")
                .str.replace(r"[چ]", "ج")
                .str.replace(r"[ڤ]", "ف")
                .str.replace(r"[گ]", "ك")
                .str.replace(r"(.)\1{3,}", r"\1\1")
                .str.replace(r"(?:\@|https?\://|#)\S+", "")
                .str.replace(r"\s+", " ")
                .str.replace(r"[^ابتثجحخدذرزسشصضطظعغفقكلمنهوي ]+", ""))
        return X_


def preprocessing_pipeline(steps: list[str], **victorizer_kwarg) -> Pipeline:
    """Create a preprocessing pipeline.

    Parameters
    ----------
    steps: list[str]
        A list of the desired preprocessing steps. the available steps are 'normalization' using ArabicTextNormalizer, 'bag of words' using CountVectorizer, and 'tfidf' using TfidfVectorizer.
    
    victorizer_kwarg: **kwarg
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