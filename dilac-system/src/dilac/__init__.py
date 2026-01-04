"""
DiLAC: Dictionary of Contemporary Arabic Lexical Resource
==========================================================

A comprehensive Arabic lexical resource for semantic processing,
based on the Dictionary of Contemporary Arabic (معجم اللغة العربية المعاصرة)
by Ahmed Mukhtar Omar.

Features:
- LMF-compliant lexical database
- Lesk-ar semantic similarity measure
- Arabic Word Sense Disambiguation
- Benchmark evaluation framework
"""

__version__ = "1.0.0"
__author__ = "DiLAC Project"

from .lmf_schema import (
    LexicalResource,
    LexicalEntry,
    Sense,
    Example,
    MorphologicalFeature,
    PartOfSpeech,
    DILAC_DOMAINS
)

from .parser import (
    DictionaryParser,
    DiLACLeskPreprocessor
)

from .similarity import (
    LeskAr,
    ExtendedLesk,
    ArabicPreprocessor,
    SimilarityBenchmark
)

from .wsd import (
    ArabicWSD,
    SimplifiedLesk,
    ContextBasedWSD,
    DomainAwareWSD,
    DisambiguatedWord
)

from .evaluation import (
    SimilarityEvaluator,
    EvaluationMetrics,
    AWSS_BENCHMARK_40,
    REFERENCE_RESULTS
)


__all__ = [
    # Schema
    'LexicalResource',
    'LexicalEntry',
    'Sense',
    'Example',
    'MorphologicalFeature',
    'PartOfSpeech',
    'DILAC_DOMAINS',
    
    # Parser
    'DictionaryParser',
    'DiLACLeskPreprocessor',
    
    # Similarity
    'LeskAr',
    'ExtendedLesk',
    'ArabicPreprocessor',
    'SimilarityBenchmark',
    
    # WSD
    'ArabicWSD',
    'SimplifiedLesk',
    'ContextBasedWSD',
    'DomainAwareWSD',
    'DisambiguatedWord',
    
    # Evaluation
    'SimilarityEvaluator',
    'EvaluationMetrics',
    'AWSS_BENCHMARK_40',
    'REFERENCE_RESULTS',
]
