# DiLAC API Documentation

## Overview

DiLAC provides a comprehensive API for Arabic semantic processing, including:

- Lexical database access
- Semantic similarity measurement (Lesk-ar)
- Word Sense Disambiguation (WSD)
- Evaluation framework

## Installation

```python
pip install dilac
```

## Quick Start

```python
from dilac import LeskAr, ArabicWSD, DiLACDatabase

# Load database
db = DiLACDatabase("data/processed/dilac.json")

# Search for a word
entries = db.search("كتاب")
for entry in entries:
    print(f"{entry.lemma}: {entry.senses[0].definition}")
```

## Core Classes

### DiLACDatabase

Main database handler for accessing the lexical resource.

```python
from dilac import DiLACDatabase

# Initialize and load
db = DiLACDatabase("data/processed/dilac.json")

# Search methods
entries = db.search("كتاب")           # By lemma
entries = db.search_by_root("ك ت ب")  # By root
entries = db.search_by_domain("طب")   # By domain

# Get definitions and examples
definitions = db.get_definitions("كتاب")
examples = db.get_examples("كتاب")

# Statistics
print(db.stats)
# {'total_entries': 32300, 'total_senses': 63019, ...}
```

### LeskAr

Implementation of the Lesk-ar semantic similarity measure.

```python
from dilac import LeskAr

# Initialize
lesk = LeskAr("data/processed/dilac_lesk.json")

# Calculate similarity
score = lesk.similarity("ساحل", "شاطئ")
print(f"Similarity: {score}")  # ~0.60

# Options
score = lesk.similarity(
    "ساحل", "شاطئ",
    use_weighting=True,    # Use IDF weighting
    use_domain=True,       # Boost for same domain
    normalize=True         # Normalize by gloss size
)

# Get best sense pair
sense1_idx, sense2_idx, score = lesk.best_sense_pair("بنك", "مال")
```

### ArabicWSD

Word Sense Disambiguation for Arabic.

```python
from dilac import ArabicWSD

# Initialize
wsd = ArabicWSD("data/processed/dilac_lesk.json")

# Disambiguate a single word
result = wsd.disambiguate(
    target_word="بنك",
    context="ذهبت إلى البنك لسحب المال",
    method='simplified_lesk'  # or 'context_based', 'domain_aware'
)

print(f"Best sense: {result.selected_sense_definition}")
print(f"Confidence: {result.confidence_score}")

# Disambiguate all words in text
results = wsd.disambiguate_text(
    "ذهبت إلى البنك لسحب المال من حسابي"
)

for r in results:
    print(f"{r.word}: {r.selected_sense_definition}")
```

### SimilarityEvaluator

Evaluate similarity measures against benchmarks.

```python
from dilac import SimilarityEvaluator, LeskAr, AWSS_BENCHMARK_40

# Create measure and evaluator
lesk = LeskAr("data/processed/dilac_lesk.json")
evaluator = SimilarityEvaluator(lesk)

# Run evaluation
metrics = evaluator.evaluate_benchmark(AWSS_BENCHMARK_40)

print(f"MSE: {metrics.mse}")
print(f"Correlation: {metrics.correlation}")

# Compare with reference methods
comparison = evaluator.compare_with_references(metrics)
for method, comp in comparison.items():
    print(f"vs {method}: MSE diff = {comp['mse_diff']}")

# Generate report
report = evaluator.generate_markdown_report(metrics)
```

## Data Classes

### LexicalEntry

Represents a dictionary entry.

```python
@dataclass
class LexicalEntry:
    id: str                 # Unique identifier
    lemma: str              # Word form (المدخل)
    pos: PartOfSpeech       # Part of speech
    morphology: MorphologicalFeature
    pronunciation: Optional[Pronunciation]
    senses: List[Sense]     # Word senses
```

### Sense

Represents a word sense.

```python
@dataclass
class Sense:
    id: str
    definition: str         # التعريف
    domain: Optional[str]   # المجال
    examples: List[Example]
    semantic_relations: List[SemanticRelation]
    contextual_expressions: List[str]  # تعبيرات سياقية
```

### DisambiguatedWord

Result of WSD.

```python
@dataclass
class DisambiguatedWord:
    word: str
    lemma: str
    selected_sense_id: str
    selected_sense_definition: str
    confidence_score: float
    all_sense_scores: List[Tuple[str, float]]
    context_window: List[str]
```

## LMF Schema

DiLAC follows the ISO 24613 (LMF) standard. See [LMF_SCHEMA.md](LMF_SCHEMA.md) for details.

### XML Structure

```xml
<LexicalResource>
  <GlobalInformation>
    <feat att="label" val="DiLAC"/>
  </GlobalInformation>
  <Lexicon>
    <feat att="language" val="ar"/>
    <LexicalEntry id="entry_001">
      <Lemma>
        <feat att="writtenForm" val="كتاب"/>
      </Lemma>
      <feat att="partOfSpeech" val="noun"/>
      <MorphologicalPattern>
        <feat att="root" val="ك ت ب"/>
      </MorphologicalPattern>
      <Sense id="sense_001_1">
        <Definition>
          <feat att="text" val="مجموعة من الصفحات..."/>
        </Definition>
        <feat att="domain" val="ثقافة"/>
        <Context id="ex_001">
          <feat att="text" val="قرأت كتابًا مفيدًا"/>
        </Context>
      </Sense>
    </LexicalEntry>
  </Lexicon>
</LexicalResource>
```

## Preprocessing

### Arabic Text Normalization

```python
from dilac import ArabicPreprocessor

# Normalize Arabic text
text = ArabicPreprocessor.normalize("أحمد إبراهيم")
# Result: "احمد ابراهيم"

# Tokenize
tokens = ArabicPreprocessor.tokenize(
    "ذهبت إلى المدرسة",
    remove_stopwords=True
)
# Result: ['ذهبت', 'المدرسه']
```

## Algorithms

### Simplified Lesk (Eq. 6.4-6.5)

```
overlap = |D(Sj) ∩ C(w)|
normalized_overlap = overlap / log₂(|D(Sj)|)
```

Where:
- D(Sj): Definition/gloss of sense j
- C(w): Context window words

### Weighted Overlap

Uses IDF weighting:

```
weight(word) = log(N / freq(word))
```

Where N is total number of definitions.

## Benchmark Data

### AWSS Benchmark

40 Arabic word pairs with human similarity judgments (scale 0-1).

```python
from dilac import AWSS_BENCHMARK_40

for word1_en, word2_en, word1_ar, word2_ar, score in AWSS_BENCHMARK_40:
    print(f"{word1_ar} - {word2_ar}: {score}")
```

### Reference Results (Table 6.2)

```python
from dilac import REFERENCE_RESULTS

print(REFERENCE_RESULTS['Lesk_ar_DiLAC'])
# {'mse': 0.020308627, 'correlation': 0.916607931, ...}
```

## Error Handling

```python
from dilac import LeskAr

lesk = LeskAr()

try:
    lesk.load_database("nonexistent.json")
except FileNotFoundError:
    print("Database not found")

# Unknown words return 0 similarity
score = lesk.similarity("موجود", "غير_موجود")
assert score == 0.0
```

## Performance Tips

1. **Load database once**: Initialize `LeskAr` or `ArabicWSD` once and reuse
2. **Batch processing**: Use `disambiguate_text()` for multiple words
3. **Preprocessing**: Pre-normalize text before similarity calculations
4. **Cache results**: Cache similarity scores for repeated word pairs

## Examples

See the `examples/` directory for complete usage examples:

- `similarity_demo.py`: Basic similarity calculation
- `wsd_demo.py`: Word sense disambiguation
- `benchmark_comparison.py`: Benchmark evaluation (Table 6.2)
