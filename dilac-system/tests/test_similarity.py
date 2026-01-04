"""
Tests for DiLAC Similarity Measures
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from dilac.similarity import LeskAr, ArabicPreprocessor


class TestArabicPreprocessor:
    """Test Arabic text preprocessing"""
    
    def test_normalize_alef(self):
        """Test alef normalization"""
        text = "أحمد إبراهيم آدم"
        result = ArabicPreprocessor.normalize(text)
        assert "ا" in result
        assert "أ" not in result
        assert "إ" not in result
        assert "آ" not in result
    
    def test_normalize_teh_marbuta(self):
        """Test teh marbuta normalization"""
        text = "مدرسة جامعة"
        result = ArabicPreprocessor.normalize(text)
        assert "ه" in result
        assert "ة" not in result
    
    def test_tokenize_basic(self):
        """Test basic tokenization"""
        text = "هذا كتاب جميل"
        tokens = ArabicPreprocessor.tokenize(text, remove_stopwords=False)
        assert len(tokens) >= 2
    
    def test_tokenize_remove_stopwords(self):
        """Test stopword removal"""
        text = "من هذا إلى ذلك الكتاب"
        tokens_with = ArabicPreprocessor.tokenize(text, remove_stopwords=False)
        tokens_without = ArabicPreprocessor.tokenize(text, remove_stopwords=True)
        assert len(tokens_without) < len(tokens_with)
    
    def test_stopwords_list(self):
        """Test that common stopwords are in the list"""
        assert "من" in ArabicPreprocessor.STOPWORDS
        assert "إلى" in ArabicPreprocessor.STOPWORDS
        assert "في" in ArabicPreprocessor.STOPWORDS


class TestLeskAr:
    """Test Lesk-ar similarity measure"""
    
    @pytest.fixture
    def mock_database(self, tmp_path):
        """Create mock database for testing"""
        import json
        
        db_data = {
            'word_to_id': {
                'ساحل': 1,
                'شاطئ': 2,
                'بحر': 3,
                'ماء': 4,
                'كتاب': 5,
                'مجلد': 6,
                'ورق': 7,
            },
            'id_to_word': {
                '1': 'ساحل',
                '2': 'شاطئ',
                '3': 'بحر',
                '4': 'ماء',
                '5': 'كتاب',
                '6': 'مجلد',
                '7': 'ورق',
            },
            'word_frequencies': {
                'ساحل': 10,
                'شاطئ': 15,
                'بحر': 20,
                'ماء': 25,
                'كتاب': 30,
                'مجلد': 5,
                'ورق': 8,
            },
            'entries': [
                {
                    'id': 'entry_1',
                    'lemma': 'ساحل',
                    'senses': [
                        {
                            'id': 'sense_1_1',
                            'encoded_gloss': [2, 3, 4],  # شاطئ، بحر، ماء
                            'domain': 'جغرافيا'
                        }
                    ]
                },
                {
                    'id': 'entry_2',
                    'lemma': 'شاطئ',
                    'senses': [
                        {
                            'id': 'sense_2_1',
                            'encoded_gloss': [1, 3, 4],  # ساحل، بحر، ماء
                            'domain': 'جغرافيا'
                        }
                    ]
                },
                {
                    'id': 'entry_3',
                    'lemma': 'كتاب',
                    'senses': [
                        {
                            'id': 'sense_3_1',
                            'encoded_gloss': [6, 7],  # مجلد، ورق
                            'domain': 'ثقافة'
                        }
                    ]
                }
            ]
        }
        
        db_path = tmp_path / "test_db.json"
        with open(db_path, 'w', encoding='utf-8') as f:
            json.dump(db_data, f, ensure_ascii=False)
        
        return str(db_path)
    
    def test_load_database(self, mock_database):
        """Test database loading"""
        lesk = LeskAr(mock_database)
        assert len(lesk.entries) == 3
        assert 'ساحل' in lesk.entries
    
    def test_similarity_same_domain(self, mock_database):
        """Test similarity between words in same domain"""
        lesk = LeskAr(mock_database)
        score = lesk.similarity('ساحل', 'شاطئ')
        assert score > 0
    
    def test_similarity_different_domain(self, mock_database):
        """Test similarity between words in different domains"""
        lesk = LeskAr(mock_database)
        score_same = lesk.similarity('ساحل', 'شاطئ')
        score_diff = lesk.similarity('ساحل', 'كتاب')
        assert score_same > score_diff
    
    def test_similarity_unknown_word(self, mock_database):
        """Test similarity with unknown word"""
        lesk = LeskAr(mock_database)
        score = lesk.similarity('ساحل', 'غير_موجود')
        assert score == 0.0
    
    def test_get_gloss_words(self, mock_database):
        """Test getting gloss words"""
        lesk = LeskAr(mock_database)
        gloss = lesk.get_gloss_words('ساحل')
        assert len(gloss) > 0
        assert 2 in gloss  # شاطئ


class TestSimilarityBenchmark:
    """Test benchmark evaluation"""
    
    def test_pearson_correlation(self):
        """Test correlation calculation"""
        from dilac.similarity import SimilarityBenchmark
        
        # Create mock measure
        class MockMeasure:
            def similarity(self, w1, w2):
                return 0.5
        
        benchmark = SimilarityBenchmark(MockMeasure())
        
        # Test perfect correlation
        x = [0.1, 0.5, 0.9]
        y = [0.2, 0.6, 1.0]
        corr = benchmark._pearson_correlation(x, y)
        assert abs(corr - 1.0) < 0.01
        
        # Test no correlation (constant values)
        x = [0.5, 0.5, 0.5]
        y = [0.1, 0.5, 0.9]
        corr = benchmark._pearson_correlation(x, y)
        assert corr == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
