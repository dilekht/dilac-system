# DiLAC: A New Arabic Lexical Resource for Semantic Processing

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Overview

**DiLAC** (Dictionnaire de la Langue Arabe Contemporaine) is a comprehensive Arabic lexical resource based on the "Dictionary of Contemporary Arabic" (معجم اللغة العربية المعاصرة) by Ahmed Mukhtar Omar. This project provides:

- A machine-readable lexical database compliant with **LMF** (Lexical Markup Framework)
- Rich linguistic information: definitions, examples, domains, grammatical categories, and morphology
- Implementation of the **Lesk-ar** semantic similarity measure
- A complete **Word Sense Disambiguation (WSD)** algorithm for Arabic
- Comprehensive evaluation framework with benchmark datasets

## Features

### Lexical Database
- **5,778 roots** (جذور)
- **32,300 lexical entries** (10,475 verbs, 21,457 nouns, 368 particles)
- **63,019 meanings** (معاني)
- **43,384 additional examples** (أمثلة إضافية)
- **17,883 contextual expressions** (تعبيرات سياقية)

### Semantic Processing
- Lesk-ar similarity measure adapted for Arabic
- Context-based and sense-based WSD algorithms
- Domain information integration
- Morphological analysis support

## Installation

```bash
# Clone the repository
git clone https://github.com/dilekht/dilac-system.git
cd dilac-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download Arabic NLP resources
python scripts/download_resources.py
```

## Quick Start

### Loading the Dictionary

```python
from dilac import DiLACDatabase

# Load the LMF-compliant database
db = DiLACDatabase("data/dilac_lmf.xml")

# Search for a word
entries = db.search("كتاب")
for entry in entries:
    print(f"Word: {entry.lemma}")
    print(f"POS: {entry.pos}")
    print(f"Definitions: {entry.definitions}")
```

### Semantic Similarity

```python
from dilac.similarity import LeskAr

lesk = LeskAr(db)

# Calculate similarity between two words
score = lesk.similarity("ساحل", "شاطئ")
print(f"Similarity: {score}")  # Output: ~0.60
```

### Word Sense Disambiguation

```python
from dilac.wsd import ArabicWSD

wsd = ArabicWSD(db)

# Disambiguate a word in context
sentence = "ذهبت إلى البنك لسحب المال"
target_word = "البنك"
best_sense = wsd.disambiguate(target_word, sentence)
print(f"Best sense: {best_sense}")
```

## Project Structure

```
dilac-system/
├── README.md
├── LICENSE
├── requirements.txt
├── setup.py
├── src/
│   └── dilac/
│       ├── __init__.py
│       ├── database.py        # LMF database handler
│       ├── parser.py          # Dictionary parser
│       ├── lmf_schema.py      # LMF XML schema
│       ├── similarity.py      # Lesk-ar implementation
│       ├── wsd.py             # WSD algorithms
│       ├── morphology.py      # Arabic morphological analyzer
│       ├── preprocessing.py   # Text preprocessing
│       └── evaluation.py      # Evaluation metrics
├── data/
│   ├── raw/                   # Raw dictionary text
│   ├── processed/             # Processed data
│   └── benchmarks/            # AWSS benchmark dataset
├── tests/
│   ├── test_similarity.py
│   ├── test_wsd.py
│   └── test_database.py
├── scripts/
│   ├── parse_dictionary.py
│   ├── generate_lmf.py
│   ├── run_evaluation.py
│   └── download_resources.py
├── docs/
│   ├── API.md
│   ├── LMF_SCHEMA.md
│   └── EVALUATION.md
└── examples/
    ├── similarity_demo.py
    ├── wsd_demo.py
    └── benchmark_comparison.py
```

## Evaluation Results

### Comparison with Existing Methods (Table 6.2)

| Measure | MSE | Correlation | MSE (Low) | MSE (Med) | MSE (High) |
|---------|-----|-------------|-----------|-----------|------------|
| **Wup (AWN)** | 0.0165 | 0.941 | 0.0028 | 0.0281 | 0.0174 |
| **AWSS** | 0.0270 | 0.890 | 0.0076 | 0.0453 | 0.0266 |
| **Lesk-ar (DiLAC)** | 0.0203 | 0.917 | 0.0083 | 0.0239 | 0.0268 |

### Key Findings
- Lesk-ar achieves competitive performance with MSE of 0.0203
- Strong correlation (0.917) with human judgments
- Superior performance on medium similarity pairs
- DiLAC's rich definitions enable effective gloss-based disambiguation

## API Documentation

See [docs/API.md](docs/API.md) for complete API documentation.

## LMF Schema

The database follows the ISO 24613 (LMF) standard. See [docs/LMF_SCHEMA.md](docs/LMF_SCHEMA.md) for schema details.

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Citation

If you use DiLAC in your research, please cite:

```bibtex
@article{dilac2024,
  title={DiLAC: A New Arabic Lexical Resource for Semantic Processing},
  author={tahar.dilekh@univ-batna2.dz},
  journal={Journal Name},
  year={2024}
}
```

## References

- Omar, A. M. (2008). معجم اللغة العربية المعاصرة. عالم الكتب.
- Lesk, M. (1986). Automatic sense disambiguation using machine readable dictionaries.
- Almarsoomi et al. (2013). AWSS: An Algorithm for Measuring Arabic Word Semantic Similarity.

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file.

## Acknowledgments

- Ahmed Mukhtar Omar for the original dictionary
- Arabic WordNet project
- AWSS benchmark dataset creators
