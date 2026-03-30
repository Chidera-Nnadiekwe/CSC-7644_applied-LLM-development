In this assignment, you will 

- Implement a Byte Pair Encoding (BPE) tokenizer from scratch in Python. 
Apply your BPE tokenizer to tokenize arbitrary text.
- Modern LLMs rely on learned subword tokenizers. Building BPE yourself clarifies why merge order and parsing fidelity affect vocabulary size, sequence length, cost, and downstream behavior. You’ll produce a deterministic, inspectable tokenizer you can later compare with off-the-shelf implementations. 

### Instructions 
You will write a single Python file named LastName_FirstName_csc7644_ca2.py, replacing LastName and FirstName with your last/first name, respectively.  

For this task, you cannot use third-party libraries, such as NumPy; you must only use built-in modules/libraries. Essentially, you cannot use any libraries that you installed via pip. 

I have provided you with a template code sample that you must complete. You can add new helper functions to the file as you see fit; however, you must complete the ones in the template, as these are the ones that will be graded. 

The code will have two modes: training and inference. The mode being used depends on the arguments passed to the command line. 

**train:** this mode will train a BPE tokenizer on the provided corpus and save the vocabulary to a pickle file 

**inference:** this mode will use the trained BPE vocabulary to tokenize text provided on the command line  

### Detailed Coding Instructions: 
1. Complete the functions provided in the example code. Locations in the code where the student must complete the code are noted in comments via # STUDENT_COMPLETE tags. The comments provide instructions for what actions each function must perform. 
2. Complete the provided Python class template for the BPE tokenizer (BPETokenizer) that can be trained on a string of data. Comments with additional instructions are provided in the template. The following methods in the class must be completed, and their intended functionality is provided below: 
    - **__init__:** The __init__ method accepts two arguments, the first, “vocab_path,” is a string that points to the path where the serialized BPE vocabulary will be/was saved. The second argument is “load”, a boolean that, if True, loads the vocabulary from “vocab_path”, and if False, initializes the vocabulary as empty. 
    - **train:** method that accepts two arguments: corpus, a string of data on which the model will be trained, and num_iter, the number of BPE training iterations (merges) to run before stopping. You are free to choose any data structure you like for storing the vocabulary, but remember that it must preserve the insertion order. 
    - **tokenize:** a method that accepts one argument, s, being a string of text that will be tokenized. The method will tokenize the string and return two outputs (a Tuple), the first being the token list and the second being the token ID list. 
    - **save:** method that serializes the trained BPE vocabulary using pickle and saves it to the path provided in __init__. 
3. Your code must have a *command-line* interface implemented using the built-in *argparse* library with: 
    - The first positional **(required)** argument should be a selector for which activity to perform. Options should be  “train” and “tokenize.” 
    - If the user selects “train,” then –data, --k, and --save need to be provided 
    - If the user selects “inference,” then –load and --s need to be provided 
    - An argument **(--data)** that points to the path (including filename and extension) to the training data corpus. This will only be used if the user selects the “train” activity. 
    - An argument **(--save)** that points to the path where the bigram model can be saved so that it can be loaded in the “inference” activity. The model must be serialized and saved using pickle. 
    - An argument **(--load)** that points to the path (string) where the trained bigram model was saved. The model object can be loaded using pickle. 
    - An argument **(--k)** that specifies the number of BPE iterations/merges to perform during training. 
    - An argument **(--s)** that is a string of text to be tokenized.  
The table below summarizes the required arguments for each mode. Refer to this while developing and testing your CLI. 

### Command Line Argument Requirements 

|Mode |Required Argument(s)|Description|
-----|--------------------|-----------|
|train| --data <path>  |Path to the training corpus file (e.g., corpus.txt)| 
|     | --k <int>           |Number of BPE merge operations to perform   |
|        | --save <path>       |Path where the trained BPE model (vocab) should be saved (e.g., model.pkl) |
|tokenize|--load <path>|Path to the trained BPE model file (e.g., model.pkl)|
|         | --s <"string to tokenize">         |The input string to tokenize using the loaded BPE model |

The program uses argparse to accept CLI arguments. The first positional argument (train or tokenize) determines the mode in which the tokenizer operates. Depending on the mode, different arguments are required. 

4. All code must be documented using comments in the manner that you would do for production-level code. Refer to the [PEP-8 style guide](https://pep8.org/). All functions and methods should have docstrings. Line-level comments should be provided for lines of code whose purpose is not immediately apparent. 

### Submission Guidelines 
The code (Python file) must be submitted through Moodle by 11:59 p.m. CST on the assigned due date. The Python file will be named **LastName_FirstName_csc7644_ca2.py**, where LastName and FirstName are replaced with your last/first names, respectively. The grade will be immediately provided to the student based on the correctness of the code. The user may submit their code up to three times. It is very important that method, function, and class names are not changed from those provided in the template. You may add additional helper methods. 

### Submission Checklist 
Before submitting, verify that you have met all the following requirements:

- **File Name** 
    - Your file is named exactly (replace LastName and FirstName):  
    *LastName_FirstName_csc7644_ca2.py* 
- **Class and Methods** 
    - You have implemented the BPETokenizer class with the following required methods: 
        - __init__(self, vocab_path, load) 
        - **train**(self, corpus, num_iter) 
        - **tokenize**(self, s) 
        - **save**(self) 
    - All required methods run without error and return the expected outputs. 
- **Command Line Interfac**e 
    - You are using argparse to handle CLI arguments. 
    - Your program supports two modes: 
        - train: requires --data, --k, --save 
        - tokenize: requires --load, --s 
    - CLI arguments match the assignment table. 
- **Serialization** 
    - The trained vocabulary is serialized with pickle and saved/loaded correctly. 
- **Style and Documentation** 
    - Code follows PEP-8 style guidelines. 
    - All methods contain docstrings. 
    - Line-level comments explain non-trivial logic. 
Once you’ve checked all items above, submit your .py file to Gradescope. 