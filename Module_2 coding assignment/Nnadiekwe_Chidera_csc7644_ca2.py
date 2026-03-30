"""
CSC 7644 - LLM Application Development
Module 2 Coding Assignment: Byte Pair Encoding (BPE) Tokenizer

This module implements a Byte Pair Encoding tokenizer from scratch. BPE is a
subword tokenization algorithm used by modern LLMs to balance vocabulary size
with sequence length. The algorithm iteratively merges the most frequent pair
of adjacent tokens until a desired vocabulary size is reached.

Why BPE Matters:
- Tokens are the unit of accuracy, latency, and billing in LLM applications
- BPE balances vocabulary size with sequence length
- Understanding tokenization helps you design better prompts and control costs

INSTRUCTIONS:
- Complete all methods marked with # STUDENT_COMPLETE
- Do not modify method signatures or class structure
- Do not use any 3rd party libraries (only built-in modules)
- Follow PEP-8 style guidelines for comments and documentation
- Test your code using the command line interface

Author: [Chidera Nnadiekwe]
Date: [March 29, 2026]
"""

import pickle
import argparse
from typing import List, Tuple, Dict, Optional


class BPETokenizer:
    """
    A Byte Pair Encoding (BPE) tokenizer that learns subword units from a corpus.
    
    BPE works by:
    1. Starting with a character-level vocabulary
    2. Iteratively finding the most frequent adjacent pair of tokens
    3. Merging that pair into a new token and adding it to the vocabulary
    4. Repeating until the desired number of merges is reached
    
    The vocabulary preserves insertion order, which is critical for correct
    tokenization during inference.
    
    Attributes:
        vocab_path: Path where the vocabulary will be saved/loaded.
        vocabulary: List of tokens in merge order (preserves insertion order).
    """
    
    def __init__(self, vocab_path: str, load: bool = False):
        """
        Initialize the BPETokenizer.
        
        Args:
            vocab_path: Path to where the vocabulary will be saved or loaded from.
                        Should have .p or .pkl extension for pickle files.
            load: If True, load an existing vocabulary from vocab_path.
                  If False, initialize an empty vocabulary for training.
        """

        # Initialize tokenizer
        self.vocab_path = vocab_path

        if load:
            # Load previously trained vocabulary from file and merge rules from disk.
            self._load()
        else:
            # Initialize empty structure for training runs.
            # Regular dicts preserve insertion order.
            self.vocabulary = {} # token_string to token_id
    
    def _most_frequent_pair(self, corpus: List[str]) -> Optional[Tuple[str, str]]:
        """
        Find the most frequent pair of adjacent tokens in the corpus.
        
        Args:
            corpus: List of current tokens.
            
        Returns:
            The most frequent pair as a tuple (token1, token2), 
            or None if no pair occurs more than once.
        """

        # Find the most frequent adjacent pair of tokens in the corpus.
        if len(corpus) < 2:
            return None
        
        pair_counts = {}

        for i in range(len(corpus) - 1):
            pair = (corpus[i], corpus[i+1])
            pair_counts[pair] = pair_counts.get(pair, 0) + 1
            
        if not pair_counts:
            return None
        
        # Sort pairs by frequency in descending order
        sorted_pairs = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)

        # Get the most frequent pair and its count
        most_frequent_pair, count = sorted_pairs[0]
        return most_frequent_pair if count > 1 else None

    def _merge_pair(self, corpus: List[str], pair: Tuple[str, str]) -> List[str]:
        """
        Replace all occurrences of a pair with the merged token.
        
        Args:
            corpus: List of current tokens.
            pair: The pair of tokens to merge as (token1, token2).
            
        Returns:
            New list of tokens with the pair merged into a single token.
        """

        # Merge the specified pair in the corpus and return the new list of tokens.
        merged_token = ''.join(pair)
        result = []
        i = 0
        # Use a while loop to iterate through the corpus and merge pairs
        while i < len(corpus):
            if i < len(corpus) - 1 and corpus[i] == pair[0] and corpus[i + 1] == pair[1]:
                result.append(merged_token)
                i += 2
            else:
                result.append(corpus[i])
                i += 1

        return result
    
    def train(self, corpus: str, num_iter: int) -> None:
        """
        Train the BPE tokenizer on a corpus of text.
        
        This method:
        1. Initializes vocabulary with unique characters from the corpus
        2. Iteratively finds the most frequent pair of adjacent tokens
        3. Merges that pair and adds the new token to the vocabulary
        4. Repeats for num_iter iterations or until no more merges possible
        
        Args:
            corpus: A string containing the training text.
            num_iter: Number of merge operations (iterations) to perform.
            
        Returns:
            None. The vocabulary is stored internally.
        """

        # Initialize vocabulary with unique characters from the corpus
        self.vocabulary = dict.fromkeys([c for c in corpus])  # Preserve order
        tokens = list(corpus)  # Start with character-level tokens

        for _ in range(num_iter):
            pair = self._most_frequent_pair(tokens)
            if pair is None:
                break  # No more pairs to merge
            
            merged_token = ''.join(pair)
            self.vocabulary[merged_token] = None  # Add merged token to vocabulary
            
            tokens = self._merge_pair(tokens, pair)  # Merge the pair in the token list

        # Convert vocabulary to list for indexing during tokenization
        self.vocabulary = list(self.vocabulary.keys())
    
    def _apply_token_merge(self, tokenized: List[str], token: str) -> List[str]:
        """
        Apply a single vocabulary token to merge matching adjacent tokens.
        
        This is used during tokenization to apply learned merges in order.
        
        Args:
            tokenized: Current list of tokens.
            token: The merged token from vocabulary to apply.
            
        Returns:
            New list of tokens with matching pairs merged.
        """

        # Apply the given token as a merge rule to the tokenized list.
        result = []
        j = 0

        # Use a while loop to iterate through the tokenized list and apply the merge
        while j < len(tokenized):
            if j < len(tokenized) - 1:
                # Check if the current and next tokens can be merged into the given token
                combined = tokenized[j] + tokenized[j + 1]
                if combined == token:
                    # If they match, append the merged token and skip the next token
                    result.append(token)
                    j += 2
                    continue
            # If no merge occurs, append the current token and move to the next one
            result.append(tokenized[j])
            j += 1
            
        return result

    def tokenize(self, s: str) -> Tuple[List[str], List[int]]:
        """
        Tokenize a string using the learned BPE vocabulary.
        
        This method applies the learned merge rules (vocabulary tokens) in order
        to convert the input string into a sequence of tokens and their IDs.
        
        Args:
            s: The input string to tokenize.
            
        Returns:
            A tuple containing:
                - List of token strings
                - List of token IDs (indices in vocabulary)
        """

        # Tokenize the input string
        tokenized = [c for c in s]  # Start with character-level tokens

        # Apply each vocabulary token (merge rule) in order
        for token in self.vocabulary:
            if len(token) > 1:  # Only apply actual merges (tokens longer than 1 character)
                tokenized = self._apply_token_merge(tokenized, token)
        
        # Convert tokens to IDs
        token_ids = self._tokens_to_ids(tokenized)

        return tokenized, token_ids

    def _tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """
        Convert a list of tokens to their vocabulary IDs.
        
        Args:
            tokens: List of token strings.
            
        Returns:
            List of integer IDs corresponding to each token's index in vocabulary.
        """

        # Convert tokens to their corresponding IDs based on vocabulary indices
        token_ids = []

        for token in tokens:
            token_ids.append(self.vocabulary.index(token)) 
        
        return token_ids

    def save(self) -> None:
        """
        Serialize and save the trained BPE vocabulary to a pickle file.
        
        Returns:
            None. The vocabulary is saved to self.vocab_path.
        """

        # Save the trained vocabulary to a file using pickle
        with open(self.vocab_path, 'wb') as f:
            pickle.dump(self.vocabulary, f)
    
    def _load(self) -> None:
        """
        Load a trained BPE vocabulary from a pickle file.
        
        Returns:
            None. The vocabulary is restored from file.
        """
 
        # Load a trained vocabulary from a file using pickle
        with open(self.vocab_path, 'rb') as f:
            self.vocabulary = pickle.load(f)
    
    def get_vocab_size(self) -> int:
        """
        Get the current vocabulary size.
        
        Returns:
            The number of tokens in the vocabulary.
        """
        return len(self.vocabulary)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """
    Main entry point for the BPE tokenizer program.
    Parses command line arguments and executes the appropriate activity.
    """

    # command line interface implementation
    parser = argparse.ArgumentParser(description="BPE Tokenizer Command Line Interface")

    parser.add_argument("activity", choices=["train", "tokenize"])

    parser.add_argument("--data", type=str)
    parser.add_argument("--k", type=int)
    parser.add_argument("--save", type=str)

    parser.add_argument("--load", type=str)
    parser.add_argument("--s", type=str)

    args = parser.parse_args()
    
    # Execute activity based on parsed arguments
    if args.activity == "train":
        if not args.data or not args.k or not args.save:
            raise ValueError("train mode requires --data, --k, --save")

        with open(args.data, "r", encoding="utf-8") as f:
            corpus = f.read()

        tokenizer = BPETokenizer(args.save, load=False)
        tokenizer.train(corpus, args.k)
        tokenizer.save()

        print("Training complete.")
        print("Vocabulary size:", tokenizer.get_vocab_size())

    elif args.activity == "tokenize":
        if not args.load or not args.s:
            raise ValueError("tokenize mode requires --load and --s")

        tokenizer = BPETokenizer(args.load, load=True)
        tokens, ids = tokenizer.tokenize(args.s)

        print("Tokens:", tokens)
        print("Token IDs:", ids)


if __name__ == "__main__":
    main()