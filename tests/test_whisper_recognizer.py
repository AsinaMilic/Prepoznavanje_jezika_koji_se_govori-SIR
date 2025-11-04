"""
Unit tests for WhisperLanguageRecognizer.

Tests cover:
- Model loading with different model sizes
- recognize_from_file with sample audio files
- recognize_from_audio with numpy arrays
- Language code mapping
- Error handling when model is not available
"""

import unittest
from unittest.mock import Mock, MagicMock, patch, MagicMock as MockModule
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Mock whisper module before importing whisper_recognizer
sys.modules['whisper'] = MagicMock()

from src.whisper_recognizer import (
    WhisperLanguageRecognizer,
    WhisperRecognizerError,
    ModelLoadError
)


class TestWhisperLanguageRecognizer(unittest.TestCase):
    """Test suite for WhisperLanguageRecognizer class"""
    
    @patch('src.whisper_recognizer.whisper.load_model')
    def test_initialization_with_default_model(self, mock_load_model):
        """Test initialization with default 'base' model size"""
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model
        
        recognizer = WhisperLanguageRecognizer()
        
        self.assertEqual(recognizer.model_size, 'base')
        self.assertEqual(recognizer.model, mock_model)
        mock_load_model.assert_called_once_with('base')
    
    @patch('src.whisper_recognizer.whisper.load_model')
    def test_initialization_with_tiny_model(self, mock_load_model):
        """Test initialization with 'tiny' model size"""
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model
        
        recognizer = WhisperLanguageRecognizer(model_size='tiny')
        
        self.assertEqual(recognizer.model_size, 'tiny')
        self.assertEqual(recognizer.model, mock_model)
        mock_load_model.assert_called_once_with('tiny')
    
    @patch('src.whisper_recognizer.whisper.load_model')
    def test_initialization_with_small_model(self, mock_load_model):
        """Test initialization with 'small' model size"""
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model
        
        recognizer = WhisperLanguageRecognizer(model_size='small')
        
        self.assertEqual(recognizer.model_size, 'small')
        mock_load_model.assert_called_once_with('small')
    
    @patch('src.whisper_recognizer.whisper.load_model')
    def test_initialization_with_medium_model(self, mock_load_model):
        """Test initialization with 'medium' model size"""
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model
        
        recognizer = WhisperLanguageRecognizer(model_size='medium')
        
        self.assertEqual(recognizer.model_size, 'medium')
        mock_load_model.assert_called_once_with('medium')
    
    @patch('src.whisper_recognizer.whisper.load_model')
    def test_initialization_with_large_model(self, mock_load_model):
        """Test initialization with 'large' model size"""
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model
        
        recognizer = WhisperLanguageRecognizer(model_size='large')
        
        self.assertEqual(recognizer.model_size, 'large')
        mock_load_model.assert_called_once_with('large')

    @patch('src.whisper_recognizer.whisper.load_model')
    def test_initialization_model_load_failure(self, mock_load_model):
        """Test that ModelLoadError is raised when model fails to load"""
        mock_load_model.side_effect = Exception("Model download failed")
        
        with self.assertRaises(ModelLoadError) as context:
            WhisperLanguageRecognizer(model_size='base')
        
        self.assertIn("Failed to load Whisper model", str(context.exception))
        self.assertIn("base", str(context.exception))
        self.assertIn("Model download failed", str(context.exception))
    
    @patch('src.whisper_recognizer.whisper.load_model')
    def test_initialization_model_not_available(self, mock_load_model):
        """Test error handling when Whisper is not installed"""
        mock_load_model.side_effect = ImportError("No module named 'whisper'")
        
        with self.assertRaises(ModelLoadError) as context:
            WhisperLanguageRecognizer()
        
        self.assertIn("Failed to load Whisper model", str(context.exception))
        self.assertIn("openai-whisper", str(context.exception))
    
    @patch('src.whisper_recognizer.whisper.load_model')
    def test_get_supported_languages(self, mock_load_model):
        """Test get_supported_languages returns sorted list of language names"""
        mock_load_model.return_value = MagicMock()
        
        recognizer = WhisperLanguageRecognizer()
        languages = recognizer.get_supported_languages()
        
        self.assertIsInstance(languages, list)
        self.assertGreater(len(languages), 0)
        
        # Check that common languages are included
        self.assertIn('english', languages)
        self.assertIn('serbian', languages)
        self.assertIn('german', languages)
        self.assertIn('spanish', languages)
        self.assertIn('french', languages)
        
        # Check that list is sorted
        self.assertEqual(languages, sorted(languages))
    
    @patch('src.whisper_recognizer.whisper.load_model')
    def test_language_code_mapping(self, mock_load_model):
        """Test that language codes are correctly mapped to readable names"""
        mock_load_model.return_value = MagicMock()
        
        recognizer = WhisperLanguageRecognizer()
        
        # Test specific mappings
        self.assertEqual(recognizer.LANGUAGE_MAP['en'], 'english')
        self.assertEqual(recognizer.LANGUAGE_MAP['sr'], 'serbian')
        self.assertEqual(recognizer.LANGUAGE_MAP['de'], 'german')
        self.assertEqual(recognizer.LANGUAGE_MAP['es'], 'spanish')
        self.assertEqual(recognizer.LANGUAGE_MAP['fr'], 'french')
        self.assertEqual(recognizer.LANGUAGE_MAP['it'], 'italian')
        self.assertEqual(recognizer.LANGUAGE_MAP['pt'], 'portuguese')
        self.assertEqual(recognizer.LANGUAGE_MAP['ru'], 'russian')
        self.assertEqual(recognizer.LANGUAGE_MAP['zh'], 'chinese')
        self.assertEqual(recognizer.LANGUAGE_MAP['ja'], 'japanese')
    
    @patch('src.whisper_recognizer.whisper.load_model')
    @patch('src.whisper_recognizer.whisper.load_audio')
    @patch('src.whisper_recognizer.whisper.pad_or_trim')
    @patch('src.whisper_recognizer.whisper.log_mel_spectrogram')
    def test_recognize_from_file_success(self, mock_log_mel, mock_pad, mock_load_audio, mock_load_model):
        """Test successful language recognition from audio file"""
        # Setup mocks
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model
        
        mock_audio = np.random.randn(16000 * 3).astype(np.float32)
        mock_load_audio.return_value = mock_audio
        mock_pad.return_value = mock_audio
        
        mock_mel = MagicMock()
        mock_mel.to.return_value = mock_mel
        mock_log_mel.return_value = mock_mel
        
        # Mock language detection results
        mock_probs = {
            'en': 0.85,
            'es': 0.10,
            'fr': 0.05
        }
        mock_model.detect_language.return_value = (None, mock_probs)
        mock_model.device = 'cpu'
        
        recognizer = WhisperLanguageRecognizer()
        results = recognizer.recognize_from_file('test_audio.wav')
        
        # Verify results
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 3)
        
        # Check top result
        self.assertEqual(results[0][0], 'english')
        self.assertAlmostEqual(results[0][1], 0.85, places=2)
        
        # Check second result
        self.assertEqual(results[1][0], 'spanish')
        self.assertAlmostEqual(results[1][1], 0.10, places=2)
        
        # Check third result
        self.assertEqual(results[2][0], 'french')
        self.assertAlmostEqual(results[2][1], 0.05, places=2)
        
        # Verify that results are sorted by probability (descending)
        self.assertGreaterEqual(results[0][1], results[1][1])
        self.assertGreaterEqual(results[1][1], results[2][1])
        
        # Verify mock calls
        mock_load_audio.assert_called_once_with('test_audio.wav')
        mock_pad.assert_called_once()
        mock_log_mel.assert_called_once()
        mock_model.detect_language.assert_called_once()

    @patch('src.whisper_recognizer.whisper.load_model')
    @patch('src.whisper_recognizer.whisper.load_audio')
    def test_recognize_from_file_with_unmapped_language(self, mock_load_audio, mock_load_model):
        """Test recognition with language code not in mapping"""
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model
        
        mock_audio = np.random.randn(16000 * 3).astype(np.float32)
        mock_load_audio.return_value = mock_audio
        
        mock_mel = MagicMock()
        mock_mel.to.return_value = mock_mel
        
        # Mock with unmapped language code
        mock_probs = {
            'xx': 0.90,  # Unmapped code
            'en': 0.05,
            'es': 0.05
        }
        mock_model.detect_language.return_value = (None, mock_probs)
        mock_model.device = 'cpu'
        
        with patch('src.whisper_recognizer.whisper.pad_or_trim') as mock_pad, \
             patch('src.whisper_recognizer.whisper.log_mel_spectrogram') as mock_log_mel:
            mock_pad.return_value = mock_audio
            mock_log_mel.return_value = mock_mel
            
            recognizer = WhisperLanguageRecognizer()
            results = recognizer.recognize_from_file('test_audio.wav')
            
            # Unmapped code should be returned as-is
            self.assertEqual(results[0][0], 'xx')
            self.assertAlmostEqual(results[0][1], 0.90, places=2)
    
    @patch('src.whisper_recognizer.whisper.load_model')
    @patch('src.whisper_recognizer.whisper.load_audio')
    def test_recognize_from_file_error_handling(self, mock_load_audio, mock_load_model):
        """Test error handling when file cannot be loaded"""
        mock_load_model.return_value = MagicMock()
        mock_load_audio.side_effect = FileNotFoundError("File not found")
        
        recognizer = WhisperLanguageRecognizer()
        
        with self.assertRaises(WhisperRecognizerError) as context:
            recognizer.recognize_from_file('nonexistent.wav')
        
        self.assertIn("Failed to recognize language", str(context.exception))
        self.assertIn("nonexistent.wav", str(context.exception))
    
    @patch('src.whisper_recognizer.whisper.load_model')
    @patch('src.whisper_recognizer.whisper.pad_or_trim')
    @patch('src.whisper_recognizer.whisper.log_mel_spectrogram')
    def test_recognize_from_audio_success(self, mock_log_mel, mock_pad, mock_load_model):
        """Test successful language recognition from numpy audio data"""
        # Setup mocks
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model
        
        mock_mel = MagicMock()
        mock_mel.to.return_value = mock_mel
        mock_log_mel.return_value = mock_mel
        mock_pad.return_value = np.random.randn(16000 * 3).astype(np.float32)
        
        # Mock language detection results
        mock_probs = {
            'sr': 0.92,
            'en': 0.05,
            'de': 0.03
        }
        mock_model.detect_language.return_value = (None, mock_probs)
        mock_model.device = 'cpu'
        
        recognizer = WhisperLanguageRecognizer()
        
        # Create test audio data (3 seconds at 16kHz)
        audio_data = np.random.randn(16000 * 3).astype(np.float32)
        results = recognizer.recognize_from_audio(audio_data, sample_rate=16000)
        
        # Verify results
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 3)
        
        # Check top result
        self.assertEqual(results[0][0], 'serbian')
        self.assertAlmostEqual(results[0][1], 0.92, places=2)
        
        # Verify mock calls
        mock_pad.assert_called_once()
        mock_log_mel.assert_called_once()
        mock_model.detect_language.assert_called_once()
    
    @patch('src.whisper_recognizer.whisper.load_model')
    @patch('src.whisper_recognizer.librosa.resample')
    @patch('src.whisper_recognizer.whisper.pad_or_trim')
    @patch('src.whisper_recognizer.whisper.log_mel_spectrogram')
    def test_recognize_from_audio_with_resampling(self, mock_log_mel, mock_pad, mock_resample, mock_load_model):
        """Test recognition with audio that needs resampling"""
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model
        
        mock_mel = MagicMock()
        mock_mel.to.return_value = mock_mel
        mock_log_mel.return_value = mock_mel
        
        # Mock resampled audio
        resampled_audio = np.random.randn(16000 * 3).astype(np.float32)
        mock_resample.return_value = resampled_audio
        mock_pad.return_value = resampled_audio
        
        mock_probs = {'en': 0.85, 'es': 0.10, 'fr': 0.05}
        mock_model.detect_language.return_value = (None, mock_probs)
        mock_model.device = 'cpu'
        
        recognizer = WhisperLanguageRecognizer()
        
        # Create audio at 44.1kHz (needs resampling to 16kHz)
        audio_data = np.random.randn(44100 * 3).astype(np.float32)
        results = recognizer.recognize_from_audio(audio_data, sample_rate=44100)
        
        # Verify resampling was called
        mock_resample.assert_called_once()
        call_args = mock_resample.call_args
        self.assertEqual(call_args[1]['orig_sr'], 44100)
        self.assertEqual(call_args[1]['target_sr'], 16000)
        
        # Verify results
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0][0], 'english')

    @patch('src.whisper_recognizer.whisper.load_model')
    @patch('src.whisper_recognizer.whisper.pad_or_trim')
    @patch('src.whisper_recognizer.whisper.log_mel_spectrogram')
    def test_recognize_from_audio_with_int16_data(self, mock_log_mel, mock_pad, mock_load_model):
        """Test recognition with int16 audio data (needs conversion to float32)"""
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model
        
        mock_mel = MagicMock()
        mock_mel.to.return_value = mock_mel
        mock_log_mel.return_value = mock_mel
        mock_pad.return_value = np.random.randn(16000 * 3).astype(np.float32)
        
        mock_probs = {'en': 0.85, 'es': 0.10, 'fr': 0.05}
        mock_model.detect_language.return_value = (None, mock_probs)
        mock_model.device = 'cpu'
        
        recognizer = WhisperLanguageRecognizer()
        
        # Create int16 audio data
        audio_data = np.random.randint(-32768, 32767, 16000 * 3, dtype=np.int16)
        results = recognizer.recognize_from_audio(audio_data, sample_rate=16000)
        
        # Verify results
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0][0], 'english')
    
    @patch('src.whisper_recognizer.whisper.load_model')
    @patch('src.whisper_recognizer.whisper.pad_or_trim')
    @patch('src.whisper_recognizer.whisper.log_mel_spectrogram')
    def test_recognize_from_audio_with_unnormalized_data(self, mock_log_mel, mock_pad, mock_load_model):
        """Test recognition with audio data that needs normalization"""
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model
        
        mock_mel = MagicMock()
        mock_mel.to.return_value = mock_mel
        mock_log_mel.return_value = mock_mel
        mock_pad.return_value = np.random.randn(16000 * 3).astype(np.float32)
        
        mock_probs = {'en': 0.85, 'es': 0.10, 'fr': 0.05}
        mock_model.detect_language.return_value = (None, mock_probs)
        mock_model.device = 'cpu'
        
        recognizer = WhisperLanguageRecognizer()
        
        # Create unnormalized audio data (values > 1.0)
        audio_data = np.random.randn(16000 * 3).astype(np.float32) * 100
        results = recognizer.recognize_from_audio(audio_data, sample_rate=16000)
        
        # Verify results
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0][0], 'english')
    
    @patch('src.whisper_recognizer.whisper.load_model')
    def test_recognize_from_audio_error_handling(self, mock_load_model):
        """Test error handling when audio data cannot be processed"""
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model
        
        recognizer = WhisperLanguageRecognizer()
        
        # Pass invalid audio data
        with self.assertRaises(WhisperRecognizerError) as context:
            recognizer.recognize_from_audio(None, sample_rate=16000)
        
        self.assertIn("Failed to recognize language", str(context.exception))
    
    @patch('src.whisper_recognizer.whisper.load_model')
    @patch('src.whisper_recognizer.whisper.pad_or_trim')
    @patch('src.whisper_recognizer.whisper.log_mel_spectrogram')
    def test_recognize_from_audio_returns_top_3_only(self, mock_log_mel, mock_pad, mock_load_model):
        """Test that recognize_from_audio returns only top 3 languages"""
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model
        
        mock_mel = MagicMock()
        mock_mel.to.return_value = mock_mel
        mock_log_mel.return_value = mock_mel
        mock_pad.return_value = np.random.randn(16000 * 3).astype(np.float32)
        
        # Mock with more than 3 languages
        mock_probs = {
            'en': 0.50,
            'es': 0.20,
            'fr': 0.15,
            'de': 0.10,
            'sr': 0.05
        }
        mock_model.detect_language.return_value = (None, mock_probs)
        mock_model.device = 'cpu'
        
        recognizer = WhisperLanguageRecognizer()
        audio_data = np.random.randn(16000 * 3).astype(np.float32)
        results = recognizer.recognize_from_audio(audio_data, sample_rate=16000)
        
        # Should return only top 3
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0][0], 'english')
        self.assertEqual(results[1][0], 'spanish')
        self.assertEqual(results[2][0], 'french')
    
    @patch('src.whisper_recognizer.whisper.load_model')
    @patch('src.whisper_recognizer.whisper.load_audio')
    @patch('src.whisper_recognizer.whisper.pad_or_trim')
    @patch('src.whisper_recognizer.whisper.log_mel_spectrogram')
    def test_recognize_from_file_returns_top_3_only(self, mock_log_mel, mock_pad, mock_load_audio, mock_load_model):
        """Test that recognize_from_file returns only top 3 languages"""
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model
        
        mock_audio = np.random.randn(16000 * 3).astype(np.float32)
        mock_load_audio.return_value = mock_audio
        mock_pad.return_value = mock_audio
        
        mock_mel = MagicMock()
        mock_mel.to.return_value = mock_mel
        mock_log_mel.return_value = mock_mel
        
        # Mock with more than 3 languages
        mock_probs = {
            'sr': 0.60,
            'en': 0.15,
            'de': 0.12,
            'es': 0.08,
            'fr': 0.05
        }
        mock_model.detect_language.return_value = (None, mock_probs)
        mock_model.device = 'cpu'
        
        recognizer = WhisperLanguageRecognizer()
        results = recognizer.recognize_from_file('test_audio.wav')
        
        # Should return only top 3
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0][0], 'serbian')
        self.assertEqual(results[1][0], 'english')
        self.assertEqual(results[2][0], 'german')
    
    @patch('src.whisper_recognizer.whisper.load_model')
    def test_probabilities_are_floats(self, mock_load_model):
        """Test that returned probabilities are Python floats, not numpy types"""
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model
        
        mock_mel = MagicMock()
        mock_mel.to.return_value = mock_mel
        
        mock_probs = {'en': 0.85, 'es': 0.10, 'fr': 0.05}
        mock_model.detect_language.return_value = (None, mock_probs)
        mock_model.device = 'cpu'
        
        with patch('src.whisper_recognizer.whisper.pad_or_trim') as mock_pad, \
             patch('src.whisper_recognizer.whisper.log_mel_spectrogram') as mock_log_mel:
            mock_pad.return_value = np.random.randn(16000 * 3).astype(np.float32)
            mock_log_mel.return_value = mock_mel
            
            recognizer = WhisperLanguageRecognizer()
            audio_data = np.random.randn(16000 * 3).astype(np.float32)
            results = recognizer.recognize_from_audio(audio_data, sample_rate=16000)
            
            # Check that probabilities are Python floats
            for lang, prob in results:
                self.assertIsInstance(prob, float)
                self.assertNotIsInstance(prob, np.floating)


if __name__ == '__main__':
    unittest.main()
