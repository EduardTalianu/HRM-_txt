import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from collections import Counter
import math
import re
import string

class LanguageModelAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("Language Model Output Analyzer")
        self.root.geometry("1200x800")
        
        # Default GPT-1 reference text
        self.gpt1_text = """Alice was the only one who would be the first woman to be admitted to the U.S. Supreme Court. In the summer of 1868, the U.S. Supreme Court ruled that the federal government could not prohibit women from serving in the military. The ruling was not without precedent in American history. In 1869, a woman named Mary Todd Lincoln was elected to Congress as the first female member of the House of Representatives. She was the first female U.S. Senator from Illinois"""
        
        self.create_widgets()
        
    def create_widgets(self):
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="Language Model Output Analyzer", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Left panel - Input
        input_frame = ttk.LabelFrame(main_frame, text="Text Input", padding="10")
        input_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        input_frame.columnconfigure(0, weight=1)
        input_frame.rowconfigure(1, weight=1)
        input_frame.rowconfigure(3, weight=1)
        
        # Small model input
        ttk.Label(input_frame, text="Small Model Output:").grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        self.small_model_text = scrolledtext.ScrolledText(input_frame, height=10, width=50)
        self.small_model_text.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        # Default small model text
        default_small = """wcoming off the things went to make firstencess the She shooking eaking--' he for
one up had her for of in her tone a little more after the new, 'Oh, which a at last.' As for her
had movent fund their heads: 'not you!' the worsted as long at lass so darkning his moment
byon't her invention suddenly: 'because three were this times at one and fronted, 'Wet's things
UThere and hands to know the feaer of the danch, and its gravely. 'You a
more to eat.

'What there you win not pet withouts--and the Hattle was eyes are
ideating the book, and she was withing the
one so throwing, this was so than held farmed and before she thoughts person."""
        
        self.small_model_text.insert("1.0", default_small)
        
        # GPT-1 reference input
        ttk.Label(input_frame, text="GPT-1 Reference Text:").grid(row=2, column=0, sticky=tk.W, pady=(0, 5))
        self.gpt1_text_widget = scrolledtext.ScrolledText(input_frame, height=10, width=50)
        self.gpt1_text_widget.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        self.gpt1_text_widget.insert("1.0", self.gpt1_text)
        
        # Analyze button
        analyze_btn = ttk.Button(input_frame, text="Analyze Output", command=self.analyze_text)
        analyze_btn.grid(row=4, column=0, pady=10)
        
        # Right panel - Results
        results_frame = ttk.LabelFrame(main_frame, text="Analysis Results", padding="10")
        results_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(1, weight=1)
        
        # Results display
        self.results_text = scrolledtext.ScrolledText(results_frame, height=35, width=60)
        self.results_text.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
    def char_ngrams(self, text, n=3):
        """Extract character n-grams from text"""
        text = text.replace("\n", " ")
        return [text[i:i+n] for i in range(len(text)-n+1)]
    
    def cosine_similarity(self, counter1, counter2):
        """Calculate cosine similarity between two frequency distributions"""
        all_ngrams = set(counter1.keys()) | set(counter2.keys())
        vec1 = [counter1.get(ng, 0) for ng in all_ngrams]
        vec2 = [counter2.get(ng, 0) for ng in all_ngrams]
        dot_product = sum(v1 * v2 for v1, v2 in zip(vec1, vec2))
        norm1 = math.sqrt(sum(v1**2 for v1 in vec1))
        norm2 = math.sqrt(sum(v2**2 for v2 in vec2))
        return dot_product / (norm1 * norm2) if norm1 and norm2 else 0.0
    
    def analyze_character_coherence(self, text):
        """Analyze character distribution and coherence"""
        total_chars = len(text)
        if total_chars == 0:
            return 0
            
        letters = sum(1 for c in text if c.isalpha())
        spaces = sum(1 for c in text if c.isspace())
        punctuation = sum(1 for c in text if c in string.punctuation)
        
        letter_ratio = letters / total_chars
        space_ratio = spaces / total_chars
        punct_ratio = punctuation / total_chars
        
        # Expected ratios for English text
        expected_letter = 0.75
        expected_space = 0.15
        expected_punct = 0.05
        
        # Calculate deviation from expected ratios
        letter_score = max(0, 100 - abs(letter_ratio - expected_letter) * 200)
        space_score = max(0, 100 - abs(space_ratio - expected_space) * 500)
        punct_score = max(0, 100 - abs(punct_ratio - expected_punct) * 1000)
        
        return round((letter_score + space_score + punct_score) / 3)
    
    def analyze_word_formation(self, text):
        """Analyze word formation quality"""
        words = re.findall(r'\b\w+\b', text.lower())
        if not words:
            return 0
            
        # Common English words for validation
        common_words = set([
            'the', 'and', 'or', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after',
            'above', 'below', 'was', 'were', 'been', 'be', 'have', 'had', 'do', 'does',
            'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can',
            'he', 'she', 'it', 'they', 'we', 'you', 'i', 'me', 'him', 'her', 'them',
            'us', 'this', 'that', 'these', 'those', 'what', 'which', 'who', 'when',
            'where', 'why', 'how', 'more', 'most', 'some', 'any', 'all', 'both',
            'each', 'every', 'one', 'two', 'three', 'first', 'last', 'time', 'way',
            'make', 'get', 'go', 'come', 'take', 'give', 'know', 'think', 'see',
            'look', 'want', 'use', 'find', 'tell', 'ask', 'work', 'seem', 'feel',
            'try', 'leave', 'call', 'hands', 'heads', 'eyes', 'book', 'things', 'eat'
        ])
        
        valid_words = sum(1 for word in words if word in common_words or len(word) > 3)
        malformed_penalty = sum(1 for word in words if len(word) > 8 and not word.isalpha())
        
        score = (valid_words / len(words)) * 100
        score = max(0, score - malformed_penalty * 5)
        
        return round(score)
    
    def analyze_sentence_structure(self, text):
        """Analyze sentence structure and grammar hints"""
        sentences = re.split(r'[.!?]+', text)
        if not sentences:
            return 0
            
        structure_score = 0
        total_sentences = len([s for s in sentences if s.strip()])
        
        if total_sentences == 0:
            return 0
            
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            words = re.findall(r'\b\w+\b', sentence.lower())
            if len(words) < 2:
                continue
                
            # Check for basic sentence patterns
            has_article = any(word in ['a', 'an', 'the'] for word in words)
            has_verb_like = any(word in ['was', 'were', 'is', 'are', 'had', 'have', 'do', 'did'] for word in words)
            has_noun_like = len(words) > 2
            
            if has_article:
                structure_score += 20
            if has_verb_like:
                structure_score += 30
            if has_noun_like:
                structure_score += 10
                
        return min(100, round(structure_score / total_sentences))
    
    def analyze_semantic_meaning(self, text):
        """Analyze semantic coherence and meaning"""
        meaningful_phrases = [
            'more to eat', 'at last', 'after the', 'in her', 'for her',
            'the book', 'she was', 'he was', 'they were', 'to know',
            'to make', 'to be', 'of the', 'and the', 'with the'
        ]
        
        text_lower = text.lower()
        found_phrases = sum(1 for phrase in meaningful_phrases if phrase in text_lower)
        
        # Check for coherent word sequences
        words = re.findall(r'\b\w+\b', text_lower)
        coherent_sequences = 0
        
        for i in range(len(words) - 2):
            sequence = ' '.join(words[i:i+3])
            if any(phrase in sequence for phrase in meaningful_phrases):
                coherent_sequences += 1
        
        score = (found_phrases * 15) + (coherent_sequences * 5)
        return min(100, score)
    
    def analyze_token_diversity(self, text):
        """Analyze vocabulary diversity and token variation"""
        words = re.findall(r'\b\w+\b', text.lower())
        if not words:
            return 0
            
        unique_words = len(set(words))
        total_words = len(words)
        
        # Type-token ratio
        ttr = unique_words / total_words
        
        # Length diversity
        lengths = [len(word) for word in words]
        avg_length = sum(lengths) / len(lengths)
        length_variance = sum((l - avg_length) ** 2 for l in lengths) / len(lengths)
        
        # Score based on diversity metrics
        ttr_score = min(100, ttr * 150)
        length_score = min(100, length_variance * 10)
        
        return round((ttr_score + length_score) / 2)
    
    def analyze_text(self):
        """Perform comprehensive text analysis"""
        small_text = self.small_model_text.get("1.0", tk.END).strip()
        gpt1_text = self.gpt1_text_widget.get("1.0", tk.END).strip()
        
        if not small_text:
            messagebox.showerror("Error", "Please enter text to analyze")
            return
            
        # Clear previous results
        self.results_text.delete("1.0", tk.END)
        
        # Calculate original cosine similarity
        n = 3
        small_ngrams = Counter(self.char_ngrams(small_text, n))
        gpt1_ngrams = Counter(self.char_ngrams(gpt1_text, n))
        similarity_score = self.cosine_similarity(small_ngrams, gpt1_ngrams)
        scaled_similarity = round(similarity_score * 100, 2)
        
        # Calculate all metrics
        char_coherence = self.analyze_character_coherence(small_text)
        word_formation = self.analyze_word_formation(small_text)
        sentence_structure = self.analyze_sentence_structure(small_text)
        semantic_meaning = self.analyze_semantic_meaning(small_text)
        token_diversity = self.analyze_token_diversity(small_text)
        
        # Calculate overall quality score
        overall_score = round((char_coherence + word_formation + sentence_structure + 
                              semantic_meaning + token_diversity) / 5)
        
        # Display results
        results = f"""LANGUAGE MODEL OUTPUT ANALYSIS REPORT
{'='*50}

CHARACTER TRIGRAM SIMILARITY TO GPT-1: {scaled_similarity}%

METRIC-BY-METRIC SCORING (Scale: 1-100)
{'='*50}

Metric                   Score    Explanation
{'-'*50}
Character Coherence      {char_coherence:3d}      Letter/space/punctuation distributions
Word Formation          {word_formation:3d}      Complete vs malformed word ratio  
Sentence Structure      {sentence_structure:3d}      Grammar hints and clause formation
Semantic Meaning        {semantic_meaning:3d}      Coherent phrases and meaning
Token Diversity         {token_diversity:3d}      Vocabulary variety and word lengths

OVERALL QUALITY SCORE:   {overall_score:3d}      Average of all metrics

DETAILED ANALYSIS
{'='*50}

Character Distribution:
• Text contains {len(small_text)} total characters
• {sum(1 for c in small_text if c.isalpha())} letters ({sum(1 for c in small_text if c.isalpha())/len(small_text)*100:.1f}%)
• {sum(1 for c in small_text if c.isspace())} spaces ({sum(1 for c in small_text if c.isspace())/len(small_text)*100:.1f}%)
• {sum(1 for c in small_text if c in string.punctuation)} punctuation marks

Word Analysis:
• {len(re.findall(r'\\b\\w+\\b', small_text))} total words found
• {len(set(re.findall(r'\\b\\w+\\b', small_text.lower())))} unique words
• Type-token ratio: {len(set(re.findall(r'\\b\\w+\\b', small_text.lower())))/max(1,len(re.findall(r'\\b\\w+\\b', small_text))):.3f}

Comparison with GPT-1:
• Character trigram overlap: {scaled_similarity}%
• Similar linguistic patterns indicate basic language understanding
• Higher scores suggest more human-like text generation

RECOMMENDATIONS
{'='*50}
"""

        if overall_score < 30:
            results += "• Focus on basic character and word formation\n"
            results += "• Improve vocabulary diversity\n"
        elif overall_score < 60:
            results += "• Work on sentence structure and grammar\n"
            results += "• Enhance semantic coherence\n"
        else:
            results += "• Good progress! Focus on semantic meaning\n"
            results += "• Consider more complex sentence structures\n"
        
        self.results_text.insert("1.0", results)

def main():
    root = tk.Tk()
    app = LanguageModelAnalyzer(root)
    root.mainloop()

if __name__ == "__main__":
    main()