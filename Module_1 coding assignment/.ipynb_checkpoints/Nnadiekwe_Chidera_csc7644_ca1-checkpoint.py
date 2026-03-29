"""
CSC 7644 - LLM Application Development
Module 1 Coding Assignment: Bigram Language Model

This module implements a bigram probabilistic language model that predicts
the next word based on the previous word. It also includes general Python
exercises to demonstrate fundamental programming concepts.

INSTRUCTIONS:
- Complete all functions and methods marked with # STUDENT_COMPLETE
- Do not modify function signatures or class structure
- Do not use any 3rd party libraries (only built-in modules)
- Follow PEP-8 style guidelines for comments and documentation
- Test your code using the command line interface

Author: [Chidera Nnadiekwe]
Date: [21th March 2026]
"""

import re
import pickle
import argparse
from typing import List, Dict, Tuple


# General Mode Functions =============================================================================
def reverse_string(s: str) -> str:
    """
    Reverse a string using slicing.
    
    Args:
        s: The input string to reverse.
        
    Returns:
        The reversed string.
        
    Example:
        >>> reverse_string("hello")
        'olleh'
    """
    # STUDENT_COMPLETE: Implement string reversal using Python slicing
    # Hint: Python strings support negative indexing and step values in slices



    # Use Python's slicing syntax  to reverse the string with a step of -1
    return s[::-1]


def count_vowels(s: str) -> int:
    """
    Count the number of vowels (a, e, i, o, u) in a string.
    Case-insensitive.
    
    Args:
        s: The input string to count vowels in.
        
    Returns:
        The count of vowels in the string.
        
    Example:
        >>> count_vowels("Hello World")
        3
    """
    # STUDENT_COMPLETE: Count vowels in the string
    # Hint: Define a string containing all vowels and iterate through the input
    # Remember to handle both uppercase and lowercase vowels



    # Convert the string to lowercase and count characters that are in 'aeiou'
    return sum(1 for char in s if char.lower() in 'aeiou')


def find_max(numbers: List[int]) -> int:
    """
    Find the maximum value in a list of integers without using built-in max().
    
    Args:
        numbers: A list of integers.
        
    Returns:
        The maximum integer in the list.
        
    Raises:
        ValueError: If the list is empty.
        
    Example:
        >>> find_max([3, 1, 4, 1, 5, 9, 2, 6])
        9
    """
    # STUDENT_COMPLETE: Find maximum without using max()
    # Hint: Initialize with first element, then compare each subsequent element
    # Don't forget to handle the empty list case



    # Check if the list is empty and raise ValueError if so
    if not numbers:
        raise ValueError("Empty list provided")
    max_val = numbers[0]
    for num in numbers[1:]:
        if num > max_val:
            max_val = num
    return max_val


def is_palindrome(s: str) -> bool:
    """
    Check if a string is a palindrome (reads the same forwards and backwards).
    Ignores case and non-alphanumeric characters.
    
    Args:
        s: The input string to check.
        
    Returns:
        True if the string is a palindrome, False otherwise.
        
    Example:
        >>> is_palindrome("A man a plan a canal Panama")
        True
    """
    # STUDENT_COMPLETE: Check if string is a palindrome
    # Step 1: Remove non-alphanumeric characters and convert to lowercase
    # Step 2: Compare the cleaned string with its reverse
    # Hint: Use str.isalnum() to check if a character is alphanumeric



    # Step 1: Keep only alphanumeric characters and lowercase everything.
    cleaned = "".join(c.lower() for c in s if c.isalnum())
    # Step 2: A palindrome reads the same forwards and backwards.
    return cleaned == cleaned[::-1]


def word_frequency(text: str) -> Dict[str, int]:
    """
    Count the frequency of each word in a text string.
    Words are converted to lowercase for counting.
    
    Args:
        text: The input text string.
        
    Returns:
        A dictionary mapping each word to its frequency count.
        
    Example:
        >>> word_frequency("the cat and the dog")
        {'the': 2, 'cat': 1, 'and': 1, 'dog': 1}
    """
    # STUDENT_COMPLETE: Count word frequencies
    # Step 1: Split text into words and convert to lowercase
    # Step 2: Build a dictionary counting occurrences of each word



    # Use regular expression to find words, convert to lowercase, and count frequencies
    word = re.findall(r"\b\w+\b", text.lower())
    freq = {}
    for w in word:
        freq[w] = freq.get(w, 0) + 1
    return freq


def fibonacci(n: int) -> List[int]:
    """
    Generate the first n Fibonacci numbers.
    
    Args:
        n: The number of Fibonacci numbers to generate.
        
    Returns:
        A list containing the first n Fibonacci numbers.
        
    Example:
        >>> fibonacci(7)
        [0, 1, 1, 2, 3, 5, 8]
    """
    # STUDENT_COMPLETE: Generate Fibonacci sequence
    # The Fibonacci sequence starts: 0, 1, 1, 2, 3, 5, 8, ...
    # Each number is the sum of the two preceding ones
    # Handle edge cases: n <= 0, n == 1, n == 2



    # Handle edge cases for n before generating the sequence
    if n <= 0:
        return []
    if n == 1:
        return [0]
    
    f_seq = [0, 1]
    for _ in range(2, n):
        # Append the sum of the last two numbers to the sequence
        f_seq.append(f_seq[-1] + f_seq[-2])
    return f_seq


def flatten_list(nested: List) -> List:
    """
    Flatten a nested list into a single-level list.
    
    Args:
        nested: A potentially nested list.
        
    Returns:
        A flattened list containing all elements.
        
    Example:
        >>> flatten_list([[1, 2], [3, [4, 5]], 6])
        [1, 2, 3, 4, 5, 6]
    """
    # STUDENT_COMPLETE: Flatten nested list
    # Hint: Use recursion - if an element is a list, flatten it recursively
    # Use isinstance(item, list) to check if an item is a list



    # Initialize an empty list to hold flattened elements
    flat = []
    for item in nested:
        if isinstance(item, list):
            # If the item is a list, recursively flatten it and extend the flat list
            flat.extend(flatten_list(item))
        else:
            flat.append(item)
    return flat


# Bigram Model Class =============================================================================
class BigramModel:
    """
    A bigram probabilistic language model that predicts the next word
    based on the previous word using conditional probabilities learned
    from a training corpus.
    
    Attributes:
        file_path: Path where the model will be saved/loaded.
        vocabulary: Set of unique words in the training corpus.
        bigram_counts: Dictionary mapping each word to counts of following words.
        bigram_probs: Dictionary mapping each word to probability distribution
                      of following words.
    """
    
    def __init__(self, file_path: str, load: bool = False):
        """
        Initialize the BigramModel.
        
        Args:
            file_path: Path to where the model will be saved or loaded from.
                       Should have .p extension for pickle files.
            load: If True, load an existing model from file_path.
                  If False, initialize an empty model for training.
        """
        # STUDENT_COMPLETE: Initialize the BigramModel
        # Store file_path as an instance variable
        # If load is True, call self._load() to load existing model
        # If load is False, initialize empty data structures:
        #   - self.vocabulary (set)
        #   - self.bigram_counts (dict)
        #   - self.bigram_probs (dict)



        # Store the file path for saving/loading the model
        self.file_path = file_path

        if load:
            # Restore previously trained model from disk
            self._load()
        else:
            # Initialize empty model data structures for training
            self.vocabulary: set = set()
            self.bigram_counts: Dict[str, Dict[str, int]] = {}
            self.bigram_probs: Dict[str, Dict[str, float]] = {}

    def _tokenize(self, corpus: str) -> List[str]:
        """
        Tokenize a corpus string into a list of word tokens.
        Treats punctuation as separate tokens.
        
        Args:
            corpus: The input text string to tokenize.
            
        Returns:
            A list of word/punctuation tokens.
        """
        # STUDENT_COMPLETE: Tokenize the corpus
        # Step 1: Add spaces around punctuation marks so they become separate tokens
        #         Use re.sub() to add spaces around: . ! ? , ; : ' " ( ) [ ] { }
        # Step 2: Split on whitespace using .split()
        # Step 3: Filter out any empty strings from the result
        # Return the list of tokens



        # Step 1: Add spaces around punctuation marks so they become separate tokens
        spaced = re.sub(r"([.!?,;:'\"()\[\]{}])", r" \1 ", corpus)

        # Step 2 and 3: Split on whitespace and filter out empty tokens
        tokens = [tk for tk in spaced.split() if tk]
        return tokens

    def train(self, corpus: str) -> None:
        """
        Train the bigram model on a corpus of text.
        
        This method tokenizes the corpus, builds the vocabulary, counts
        bigram occurrences, and computes conditional probabilities for
        each word following every other word.
        
        Args:
            corpus: A string containing the training text.
            
        Returns:
            None. The model's internal state is updated.
        """
        # STUDENT_COMPLETE: Train the bigram model
        # Step 1: Tokenize the corpus using self._tokenize()
        #
        # Step 2: Build vocabulary as set of unique tokens
        #
        # Step 3: Count bigram occurrences
        #         - Loop through tokens (except the last one)
        #         - For each token, count which tokens follow it
        #         - Store in self.bigram_counts as nested dict:
        #           {word: {next_word: count, ...}, ...}
        #
        # Step 4: Compute conditional probabilities
        #         - P(next_word | current_word) = count(current, next) / total_count(current)
        #         - Store in self.bigram_probs as nested dict:
        #           {word: {next_word: probability, ...}, ...}



        # Step 1: Tokenize the corpus into a list of words
        tokens = self._tokenize(corpus)

        # Step 2: Build the vocabulary as a set of unique tokens
        self.vocabulary = set(tokens)

        # Step 3: Count bigram occurrences
        self.bigram_counts = {}
        for i in range(len(tokens) - 1):
            current_word = tokens[i]
            next_word = tokens[i + 1]

            if current_word not in self.bigram_counts:
                self.bigram_counts[current_word] = {}

            self.bigram_counts[current_word][next_word] = (
                self.bigram_counts[current_word].get(next_word, 0) + 1
            )

        # Step 4: Compute conditional probabilities for each bigram
        #         P(next_word | current_word) = count(current, next) / total_count(current)
        self.bigram_probs = {}
        for word, next_words in self.bigram_counts.items():
            total_count = sum(next_words.values())
            self.bigram_probs[word] = {
                next_word: count / total_count 
                for next_word, count in next_words.items()
            }
        
        print(
            f"Training complete. "
            f"Vocabulary size: {len(self.vocabulary)}"
        )

    def predict_next_word(self, word: str) -> str:
        """
        Predict the next word given the current word using greedy decoding.
        Returns the word with the highest conditional probability.
        
        Args:
            word: The current word (string) to predict from.
            
        Returns:
            The predicted next word with highest probability.
            
        Raises:
            KeyError: If the input word is not in the vocabulary.
        """
        # STUDENT_COMPLETE: Implement greedy prediction
        # Step 1: Check if word exists in self.bigram_probs
        #         If not, raise KeyError with appropriate message
        #
        # Step 2: Get the probability distribution for words following input
        #
        # Step 3: Find and return the word with maximum probability
        #         (This is greedy decoding - always pick the most likely)



        # Step 1: Check if the input word is in the bigram probabilities
        if word not in self.bigram_probs:
            raise KeyError(
                f"Word '{word}' was not found during training. It has no probability distribution"
            )
        
        # Step 2: Get the probability distribution for words following the input word
        # Step 3: Find and return the word with the maximum probability
        next_probs_word = max(
            self.bigram_probs[word], 
            key=lambda x: self.bigram_probs[word][x]
        )
        return next_probs_word

    def infer(self, word: str, n: int) -> str:
        """
        Generate n words autoregressively starting from the given word.
        Each predicted word becomes the input for the next prediction.
        
        Args:
            word: The starting word for generation.
            n: The number of words to generate.
            
        Returns:
            A string containing the n generated words separated by spaces.
        """
        # STUDENT_COMPLETE: Implement autoregressive generation
        # Step 1: Initialize a list to store generated words
        # Step 2: Set current_word to the input word
        # Step 3: Loop n times:
        #         - Predict next word using predict_next_word()
        #         - Append predicted word to list
        #         - Update current_word to be the predicted word
        #         - Handle KeyError if word not in vocabulary (stop early)
        # Step 4: Join all generated words with spaces and return



        # Step 1: Initialize a list to store generated words
        generated_words: List[str] = []
        current_word = word

        # Step 2: Loop n times to generate words autoregressively
        for _ in range(n):
            try:
                next_word = self.predict_next_word(current_word)
            except KeyError:
                # If the current word has no successors in the model, stop generation early
                print(
                    f"Alert. '{current_word}' has no successors. Stopping generation early."
                )
                break
            generated_words.append(next_word)
            # Update current_word for the next iteration 
            current_word = next_word

        return ' '.join(generated_words)
    
    def save(self) -> None:
        """
        Serialize and save the trained bigram model to a pickle file.
        Saves vocabulary, bigram_counts, and bigram_probs.
        
        Returns:
            None. The model is saved to self.file_path.
        """
        # STUDENT_COMPLETE: Save model using pickle
        # Step 1: Create a dictionary containing all model data:
        #         - vocabulary
        #         - bigram_counts
        #         - bigram_probs
        # Step 2: Open file in binary write mode ('wb')
        # Step 3: Use pickle.dump() to serialize and save



        # Step 1: Bundle all model data into a dictionary for serialization
        data = {
            "vocabulary": self.vocabulary,
            "bigram_counts": self.bigram_counts,
            "bigram_probs": self.bigram_probs,
        }
        # Step 2 and 3: Open the specified file path in binary-write mode and save the model using pickle
        with open(self.file_path, "wb") as fx:
            pickle.dump(data, fx)
        print(f"Model saved to {self.file_path}")
    
    def _load(self) -> None:
        """
        Load a trained bigram model from a pickle file.
        
        Returns:
            None. The model's internal state is restored from file.
        """
        # STUDENT_COMPLETE: Load model using pickle
        # Step 1: Open file in binary read mode ('rb')
        # Step 2: Use pickle.load() to deserialize
        # Step 3: Restore model attributes from loaded data



        # Step 1 and 2: Open in binary-read mode and deserialize the model data using pickle
        with open(self.file_path, "rb") as fx:
            data = pickle.load(fx)
        
        # Step 3: Restore model attributes from the loaded data
        self.vocabulary = data["vocabulary"]
        self.bigram_counts = data["bigram_counts"]
        self.bigram_probs = data["bigram_probs"]
        print(f"Model loaded from {self.file_path}. Vocabulary size: {len(self.vocabulary)}")


# Main Entry Point for Code =============================================================================
def run_general_tests():
    """
    Run and display results for all general mode functions.
    This demonstrates basic Python programming concepts.
    """
    print("=" * 60)
    print("GENERAL MODE - Python Fundamentals Tests")
    print("=" * 60)
    
    # Test reverse_string
    print("\n1. reverse_string('hello'):")
    print(f"   Result: {reverse_string('hello')}")
    
    # Test count_vowels
    print("\n2. count_vowels('Hello World'):")
    print(f"   Result: {count_vowels('Hello World')}")
    
    # Test find_max
    print("\n3. find_max([3, 1, 4, 1, 5, 9, 2, 6]):")
    print(f"   Result: {find_max([3, 1, 4, 1, 5, 9, 2, 6])}")
    
    # Test is_palindrome
    print("\n4. is_palindrome('A man a plan a canal Panama'):")
    print(f"   Result: {is_palindrome('A man a plan a canal Panama')}")
    
    # Test word_frequency
    print("\n5. word_frequency('the cat and the dog'):")
    print(f"   Result: {word_frequency('the cat and the dog')}")
    
    # Test fibonacci
    print("\n6. fibonacci(7):")
    print(f"   Result: {fibonacci(7)}")
    
    # Test flatten_list
    print("\n7. flatten_list([[1, 2], [3, [4, 5]], 6]):")
    print(f"   Result: {flatten_list([[1, 2], [3, [4, 5]], 6])}")
    
    print("\n" + "=" * 60)


def main():
    """
    Main entry point for the bigram language model program.
    Parses command line arguments and executes the appropriate activity.
    """
    # STUDENT_COMPLETE: Implement command line interface using argparse
    # 
    # Required arguments:
    # 1. Positional argument 'activity' with choices: 'general', 'train', 'inference'
    #
    # 2. --data: Path to training data corpus (for 'train' activity)
    #
    # 3. --save: Path where model will be saved (for 'train' activity)
    #
    # 4. --load: Path to trained model (for 'inference' activity)
    #
    # 5. --word: Starting word for generation (for 'inference' activity)
    #
    # 6. --n: Number of words to generate (for 'inference' activity)
    #
    # Activity implementations:
    #
    # 'general': Call run_general_tests()
    #
    # 'train':
    #   - Validate that --data and --save are provided
    #   - Load corpus from file specified by --data
    #   - Create BigramModel with --save path
    #   - Train model on corpus
    #   - Save model
    #
    # 'inference':
    #   - Validate that --load, --word, and --n are provided
    #   - Load trained model from --load path
    #   - Generate n words starting from --word
    #   - Print starting word followed by generated text
    


    parser = argparse.ArgumentParser(
        description="CSC 7644 CA1 - Bigram Language Model CLI",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # 1: Positional activity selector with detailed help message
    parser.add_argument(
        "activity", 
        choices=["general", "train", "inference"], 
        help=(
            "Which activity to run:\n"
            "  general   — Run Python fundamentals demonstrations\n"
            "  train     — Train a bigram model (requires --data, --save)\n"
            "  inference — Generate text from a model "
            "(requires --load, --word, --n)"
        )
    )

    # 2. --data: Path to training data corpus (for 'train' activity)
    parser.add_argument(
        "--data", 
        type=str,
        default=None,
        help="Path to training corpus (required for 'train' activity)",
    )

    # 3. --save: Path where model will be saved (for 'train' activity)
    parser.add_argument(
        "--save", 
        type=str,
        default=None,
        help="Path to save trained model (.p file, required for 'train' activity)",
    )

    # 4. --load: Path to trained model (for 'inference' activity)
    parser.add_argument(
        "--load", 
        type=str,
        default=None,
        help="Path to a previously saved model",
    )

    # 5. --word: Starting word for generation (for 'inference' activity)
    parser.add_argument(
        "--word", 
        type=str,
        default=None,
        help="Starting word for autoregressive text generation",
    )

    # 6. --n: Number of words to generate (for 'inference' activity)
    parser.add_argument(
        "--n", 
        type=int,
        default=None,
        help="Number of words to generate (required for 'inference' activity)",
    )

    args = parser.parse_args()

    # -----------------------------------------------------------------------#
    # Activity: general
    # -----------------------------------------------------------------------#
    if args.activity == "general":
        run_general_tests()

    # -----------------------------------------------------------------------#            
    # Activity: train
    # -----------------------------------------------------------------------#
    elif args.activity == "train":
        # Validate that --data and --save are provided, if not print error and exit
        missing = [f"--{arg}" for arg in ["data", "save"] 
                   if getattr(args, arg) is None]
        if missing:
            parser.error(f"Missing required arguments for 'train' activity: {', '.join(missing)}")
        
        # Load corpus from the specified file path/disk
        with open(args.data, "r", encoding="utf-8") as f:
            corpus = f.read()
        print(f"Corpus loaded from {args.data}. Length: {len(corpus)} characters.")

        # Create, train, and save the bigram model
        model = BigramModel(file_path=args.save, load=False)
        model.train(corpus)
        model.save()

    # -----------------------------------------------------------------------#
    # Activity: inference
    # -----------------------------------------------------------------------#
    elif args.activity == "inference":
        # Validate that --load, --word, and --n are provided, if not print error and exit
        missing = [f"--{arg}" for arg in ["load", "word", "n"] 
                   if getattr(args, arg) is None]
        if missing:
            parser.error(
                f"Missing required arguments for 'inference' activity: {', '.join(missing)}"
            )

        # Load the trained model from file path/disk and generate text starting from the input word
        model = BigramModel(file_path=args.load, load=True)
        generated_text = model.infer(word=args.word, n=args.n)
        
        # Print the starting word and the generated text
        print(f"Starting word: {args.word}")
        print(f"Generated text: {generated_text}")

if __name__ == "__main__":
    main()