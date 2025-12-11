from typing import List
from deepmultilingualpunctuation import PunctuationModel
from src.tags import VideoTag

class Prettifier:
    """Handles text correction, punctuation, and capitalization"""
    
    def __init__(self):
        self.punctuation_model = PunctuationModel()
    
    def prettify(self, tags: List[VideoTag], max_gap: int) -> List[VideoTag]:
        """
        Apply punctuation and capitalization to word-level tags
        
        Args:
            tags: List of word-level VideoTags
            max_gap: Maximum gap in ms to consider words part of same sentence
        
        Returns:
            List of VideoTags with corrected text
        """
        if len(tags) == 0:
            return []
        
        # Group into sentences based on time gaps
        sentences = []
        current_sentence = [tags[0].text]
        last_start = tags[0].start_time
        
        for tag in tags[1:]:
            if tag.start_time - last_start > max_gap:
                sentences.append(' '.join(current_sentence))
                current_sentence = [tag.text]
            else:
                current_sentence.append(tag.text)
            last_start = tag.start_time
        
        sentences.append(' '.join(current_sentence))
        
        # Apply corrections to each sentence
        corrected_sentences = [self._correct_text(s) for s in sentences]
        corrected_text = ' '.join(corrected_sentences)
        corrected_words = corrected_text.split()
        
        # Update tags with corrected words
        output_tags: List[VideoTag] = []
        for i, tag in enumerate(tags):
            if i < len(corrected_words):
                output_tags.append(VideoTag(
                    start_time=tag.start_time,
                    end_time=tag.end_time,
                    text=corrected_words[i]
                ))
        
        return output_tags
    
    def _correct_text(self, text: str) -> str:
        """Apply punctuation and capitalization to a single piece of text"""
        if text == "":
            return text
        
        res = self._capitalize_proper_nouns(text)
        res = self.punctuation_model.restore_punctuation(res)
        
        if not res.endswith("."):
            res += "."
        
        # Capitalize first letter of sentences
        sentence_delimiters = ['.', '?', '!']
        capitalized = []
        for i, c in enumerate(res):
            if i == 0 or (i > 1 and res[i-2] in sentence_delimiters):
                capitalized.append(c.upper())
            else:
                capitalized.append(c)
        
        return ''.join(capitalized)
    
    def _capitalize_proper_nouns(self, sentence: str) -> str:
        # TODO: add back spacy
        return sentence