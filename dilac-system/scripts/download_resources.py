#!/usr/bin/env python3
"""
Download Resources Script
==========================

Downloads and sets up required resources for DiLAC.
This includes creating necessary directories and 
preparing sample data for testing.

Usage:
    python scripts/download_resources.py
"""

import os
import sys
import json
from pathlib import Path

def main():
    print("=" * 60)
    print("DiLAC Resource Setup")
    print("=" * 60)
    print()
    
    # Get project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # Create directories
    directories = [
        project_root / "data" / "raw",
        project_root / "data" / "processed", 
        project_root / "data" / "benchmarks",
    ]
    
    print("Creating directories...")
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {directory}")
    
    print()
    
    # Check if benchmark data exists
    benchmark_file = project_root / "data" / "benchmarks" / "awss_40.json"
    if benchmark_file.exists():
        print(f"✓ Benchmark data already exists: {benchmark_file}")
    else:
        print("Creating benchmark data...")
        create_benchmark_data(benchmark_file)
        print(f"  ✓ Created {benchmark_file}")
    
    print()
    
    # Create sample dictionary entry for testing
    sample_file = project_root / "data" / "processed" / "sample_entries.json"
    if not sample_file.exists():
        print("Creating sample dictionary entries for testing...")
        create_sample_entries(sample_file)
        print(f"  ✓ Created {sample_file}")
    
    print()
    print("=" * 60)
    print("Setup Complete!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. Place your dictionary file in: data/raw/")
    print("  2. Run: python scripts/parse_dictionary.py data/raw/dictionary_full.txt")
    print("  3. Test: python examples/benchmark_comparison.py")
    print()
    print("Optional Arabic NLP libraries (install manually if needed):")
    print("  pip install camel-tools    # For advanced Arabic processing")
    print("  pip install pyarabic       # For Arabic text utilities")
    print()


def create_benchmark_data(filepath: Path):
    """Create AWSS benchmark dataset"""
    data = {
        "name": "AWSS Arabic Word Semantic Similarity Benchmark",
        "description": "Human similarity judgments for 40 Arabic word pairs from Table 6.2",
        "source": "Fazza et al. (2012), Mohammed & Deri (2017)",
        "scale": "0.0 to 1.0 (normalized from 0-4)",
        "pairs": [
            {"id": 1, "word1_en": "Coast", "word2_en": "Endorsement", "word1": "ساحل", "word2": "تصديق", "human_score": 0.01, "category": "low"},
            {"id": 2, "word1_en": "Noon", "word2_en": "String", "word1": "ظهر", "word2": "خيط", "human_score": 0.01, "category": "low"},
            {"id": 3, "word1_en": "Stove", "word2_en": "Walk", "word1": "موقد", "word2": "مشي", "human_score": 0.02, "category": "low"},
            {"id": 4, "word1_en": "Cord", "word2_en": "Midday", "word1": "حبل", "word2": "ظهيرة", "human_score": 0.02, "category": "low"},
            {"id": 5, "word1_en": "Signature", "word2_en": "String", "word1": "توقيع", "word2": "خيط", "human_score": 0.02, "category": "low"},
            {"id": 6, "word1_en": "Boy", "word2_en": "Endorsement", "word1": "صبي", "word2": "تصديق", "human_score": 0.03, "category": "low"},
            {"id": 7, "word1_en": "Boy", "word2_en": "Midday", "word1": "صبي", "word2": "ظهيرة", "human_score": 0.04, "category": "low"},
            {"id": 8, "word1_en": "Smile", "word2_en": "Village", "word1": "ابتسامة", "word2": "قرية", "human_score": 0.05, "category": "low"},
            {"id": 9, "word1_en": "Noon", "word2_en": "Fasting", "word1": "ظهر", "word2": "صيام", "human_score": 0.07, "category": "low"},
            {"id": 10, "word1_en": "Glass", "word2_en": "Diamond", "word1": "كأس", "word2": "الماس", "human_score": 0.09, "category": "low"},
            {"id": 11, "word1_en": "Sepulcher", "word2_en": "Sheikh", "word1": "ضريح", "word2": "شيخ", "human_score": 0.22, "category": "low"},
            {"id": 12, "word1_en": "Countryside", "word2_en": "Vegetable", "word1": "ريف", "word2": "خضار", "human_score": 0.31, "category": "low"},
            {"id": 13, "word1_en": "Tumbler", "word2_en": "Tool", "word1": "قدح", "word2": "أداة", "human_score": 0.33, "category": "medium"},
            {"id": 14, "word1_en": "Laugh", "word2_en": "Feast", "word1": "ضحك", "word2": "عيد", "human_score": 0.34, "category": "medium"},
            {"id": 15, "word1_en": "Girl", "word2_en": "Odalisque", "word1": "فتاة", "word2": "جارية", "human_score": 0.49, "category": "medium"},
            {"id": 16, "word1_en": "Feast", "word2_en": "Fasting", "word1": "عيد", "word2": "صيام", "human_score": 0.49, "category": "medium"},
            {"id": 17, "word1_en": "Coach", "word2_en": "Means", "word1": "حافلة", "word2": "وسيلة", "human_score": 0.52, "category": "medium"},
            {"id": 18, "word1_en": "Sage", "word2_en": "Sheikh", "word1": "حكيم", "word2": "شيخ", "human_score": 0.57, "category": "medium"},
            {"id": 19, "word1_en": "Girl", "word2_en": "Sister", "word1": "فتاة", "word2": "اخت", "human_score": 0.60, "category": "medium"},
            {"id": 20, "word1_en": "Hen", "word2_en": "Pigeon", "word1": "دجاجة", "word2": "حمامة", "human_score": 0.65, "category": "medium"},
            {"id": 21, "word1_en": "Hill", "word2_en": "Mountain", "word1": "تل", "word2": "جبل", "human_score": 0.65, "category": "medium"},
            {"id": 22, "word1_en": "Master", "word2_en": "Sheikh", "word1": "سيد", "word2": "شيخ", "human_score": 0.67, "category": "high"},
            {"id": 23, "word1_en": "Food", "word2_en": "Vegetable", "word1": "طعام", "word2": "خضار", "human_score": 0.69, "category": "high"},
            {"id": 24, "word1_en": "Slave", "word2_en": "Odalisque", "word1": "عبد", "word2": "جارية", "human_score": 0.71, "category": "high"},
            {"id": 25, "word1_en": "Run", "word2_en": "Walk", "word1": "جري", "word2": "مشي", "human_score": 0.75, "category": "high"},
            {"id": 26, "word1_en": "Cord", "word2_en": "String", "word1": "حبل", "word2": "خيط", "human_score": 0.77, "category": "high"},
            {"id": 27, "word1_en": "Forest", "word2_en": "Woodland", "word1": "غابة", "word2": "أحراش", "human_score": 0.79, "category": "high"},
            {"id": 28, "word1_en": "Sage", "word2_en": "Thinker", "word1": "حكيم", "word2": "مفكر", "human_score": 0.83, "category": "high"},
            {"id": 29, "word1_en": "Journey", "word2_en": "Travel", "word1": "رحلة", "word2": "سفر", "human_score": 0.85, "category": "high"},
            {"id": 30, "word1_en": "Gem", "word2_en": "Diamond", "word1": "جوهرة", "word2": "الماس", "human_score": 0.85, "category": "high"},
            {"id": 31, "word1_en": "Countryside", "word2_en": "Village", "word1": "ريف", "word2": "قرية", "human_score": 0.85, "category": "high"},
            {"id": 32, "word1_en": "Cushion", "word2_en": "Pillow", "word1": "مسند", "word2": "مخدة", "human_score": 0.85, "category": "high"},
            {"id": 33, "word1_en": "Smile", "word2_en": "Laugh", "word1": "ابتسامة", "word2": "ضحك", "human_score": 0.87, "category": "high"},
            {"id": 34, "word1_en": "Signature", "word2_en": "Endorsement", "word1": "توقيع", "word2": "تصديق", "human_score": 0.90, "category": "high"},
            {"id": 35, "word1_en": "Tool", "word2_en": "Means", "word1": "أداة", "word2": "وسيلة", "human_score": 0.92, "category": "high"},
            {"id": 36, "word1_en": "Sepulcher", "word2_en": "Grave", "word1": "ضريح", "word2": "قبر", "human_score": 0.94, "category": "high"},
            {"id": 37, "word1_en": "Boy", "word2_en": "Lad", "word1": "صبي", "word2": "فتى", "human_score": 0.93, "category": "high"},
            {"id": 38, "word1_en": "Wizard", "word2_en": "Magician", "word1": "ساحر", "word2": "مشعوذ", "human_score": 0.94, "category": "high"},
            {"id": 39, "word1_en": "Coach", "word2_en": "Bus", "word1": "حافلة", "word2": "باص", "human_score": 0.95, "category": "high"},
            {"id": 40, "word1_en": "Glass", "word2_en": "Tumbler", "word1": "كأس", "word2": "قدح", "human_score": 0.95, "category": "high"}
        ],
        "reference_results": {
            "Wup_AWN": {"mse": 0.016540541, "correlation": 0.941168501},
            "AWSS": {"mse": 0.026978378, "correlation": 0.889771136},
            "Lesk_ar_DiLAC": {"mse": 0.020308627, "correlation": 0.916607931}
        }
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def create_sample_entries(filepath: Path):
    """Create sample dictionary entries for testing"""
    data = {
        "name": "DiLAC Sample Entries",
        "description": "Sample entries for testing the system",
        "word_to_id": {
            "ساحل": 1, "شاطئ": 2, "بحر": 3, "ماء": 4, "موج": 5,
            "كتاب": 6, "مجلد": 7, "ورق": 8, "قراءة": 9, "كتابة": 10,
            "حبل": 11, "خيط": 12, "ربط": 13, "وثاق": 14,
            "جبل": 15, "تل": 16, "هضبة": 17, "ارتفاع": 18,
            "ضحك": 19, "ابتسامة": 20, "فرح": 21, "سرور": 22
        },
        "id_to_word": {
            "1": "ساحل", "2": "شاطئ", "3": "بحر", "4": "ماء", "5": "موج",
            "6": "كتاب", "7": "مجلد", "8": "ورق", "9": "قراءة", "10": "كتابة",
            "11": "حبل", "12": "خيط", "13": "ربط", "14": "وثاق",
            "15": "جبل", "16": "تل", "17": "هضبة", "18": "ارتفاع",
            "19": "ضحك", "20": "ابتسامة", "21": "فرح", "22": "سرور"
        },
        "word_frequencies": {
            "ساحل": 50, "شاطئ": 45, "بحر": 120, "ماء": 200, "موج": 30,
            "كتاب": 300, "مجلد": 25, "ورق": 80, "قراءة": 90, "كتابة": 85,
            "حبل": 40, "خيط": 35, "ربط": 60, "وثاق": 15,
            "جبل": 100, "تل": 55, "هضبة": 20, "ارتفاع": 70,
            "ضحك": 75, "ابتسامة": 65, "فرح": 95, "سرور": 45
        },
        "entries": [
            {
                "id": "entry_001",
                "lemma": "ساحل",
                "senses": [
                    {
                        "id": "sense_001_1",
                        "definition": "شاطئ البحر أو النهر",
                        "encoded_gloss": [2, 3, 4, 5],
                        "domain": "جغرافيا"
                    }
                ]
            },
            {
                "id": "entry_002",
                "lemma": "شاطئ",
                "senses": [
                    {
                        "id": "sense_002_1",
                        "definition": "ساحل البحر حيث يلتقي الماء باليابسة",
                        "encoded_gloss": [1, 3, 4, 5],
                        "domain": "جغرافيا"
                    }
                ]
            },
            {
                "id": "entry_003",
                "lemma": "كتاب",
                "senses": [
                    {
                        "id": "sense_003_1",
                        "definition": "مجلد يضم صفحات من الورق للقراءة والكتابة",
                        "encoded_gloss": [7, 8, 9, 10],
                        "domain": "ثقافة"
                    }
                ]
            },
            {
                "id": "entry_004",
                "lemma": "حبل",
                "senses": [
                    {
                        "id": "sense_004_1",
                        "definition": "خيط غليظ للربط والوثاق",
                        "encoded_gloss": [12, 13, 14],
                        "domain": None
                    }
                ]
            },
            {
                "id": "entry_005",
                "lemma": "خيط",
                "senses": [
                    {
                        "id": "sense_005_1",
                        "definition": "حبل رفيع للربط والخياطة",
                        "encoded_gloss": [11, 13],
                        "domain": None
                    }
                ]
            },
            {
                "id": "entry_006",
                "lemma": "جبل",
                "senses": [
                    {
                        "id": "sense_006_1",
                        "definition": "ارتفاع كبير في الأرض أعلى من التل والهضبة",
                        "encoded_gloss": [16, 17, 18],
                        "domain": "جغرافيا"
                    }
                ]
            },
            {
                "id": "entry_007",
                "lemma": "تل",
                "senses": [
                    {
                        "id": "sense_007_1",
                        "definition": "ارتفاع صغير في الأرض أقل من الجبل",
                        "encoded_gloss": [15, 17, 18],
                        "domain": "جغرافيا"
                    }
                ]
            },
            {
                "id": "entry_008",
                "lemma": "ضحك",
                "senses": [
                    {
                        "id": "sense_008_1",
                        "definition": "إظهار الفرح والسرور بصوت أو ابتسامة",
                        "encoded_gloss": [20, 21, 22],
                        "domain": None
                    }
                ]
            },
            {
                "id": "entry_009",
                "lemma": "ابتسامة",
                "senses": [
                    {
                        "id": "sense_009_1",
                        "definition": "ضحك خفيف يظهر الفرح والسرور",
                        "encoded_gloss": [19, 21, 22],
                        "domain": None
                    }
                ]
            }
        ]
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
