# Description
To install Anaconda, create a virtual environment, and put into practice the fundamental Python concepts taught in the first lecture. In addition, you will begin to think about the concepts of probabilistic language models by implementing a bigram language model. 

### Why This Matters
Python is the de facto standard programming language for all machine learning work, including large language models. The ability to develop programs in Python is crucial for leveraging these technologies in your future career. Additionally, n-gram language models are the simplest example of a probabilistic language model. A bigram language model predicts the next word based on the previous word, creating a bigram. Large language models are very complex examples of probabilistic language models. By implementing a simple bigram model, you will better understand the probabilistic nature of all language models, including LLMs, which will help you design more effective and interpretable LLM solutions in your career. 

### Instructions 
1. You will write a single Python file named **LastName_FirstName_csc7644_ca1.py**, replacing LastName and FirstName with your last/first name, respectively.
2. For this task, you cannot use third-party libraries, such as NumPy; you must only use built-in modules/libraries. Create a new Python 3.12 environment to use for the assignment, but do not install any additional libraries. 
3. I have provided you with a template code sample that you must complete. You can add new helper functions to the file as you see fit; however, you must complete the ones in the template, as these are the ones that will be graded. 
4. The code will have three modes: general, training, and inference. The mode being used depends on the arguments passed to the command line. 
    - **general**: this mode runs a set of code to evaluate your general understanding of introductory Python. 
    - **train**: this mode will train a bigram language model on the provided corpus and save the trained conditional probability distribution structure (referred to as the bigram model) to a pickle file. 
    - **inference**: this mode will use the trained bigram language model from the pickle file and generate words based on command-line arguments. 
### Detailed Coding Instructions:
1. Complete the functions provided in the example code. Locations in the code where the student must complete the code are noted in comments via STUDENT_COMPLETE tags. The comments provide instructions for what actions each function must perform. 
2. Complete the provided Python class template for the bigram model (BigramModel) that can be trained on a string of data. Comments with additional instructions are provided in the template. The following methods in the class must be completed, and their intended functionality is provided below: 
    - **__init__:** The __init__ method accepts two arguments, the first, “file_path,” is a string that points to the path where the serialized bigram model will be saved. The second argument is “load”, a boolean that, if True, loads the bigram model from “file_path”, and if False, initializes the model.
    - **train:** method that accepts one positional argument, corpus, a string of data on which the model will be trained. The method must split the provided string into separate words (treat punctuation as separate words), identify the unique words in the corpus, identify which other words can follow each unique word, and quantify the probability that each word follows other words. For example, if the word “cats” is only followed by the words “has”, “are”, and “will” in the data, and if each of those words follows “cat” the same number of times, then all three words have the same probability of occurring (0.33 or 33%). 
    - **predict_next_word:** a method that accepts one argument, input, being a single word (string). The method will use this input to sample the next word from the probability distribution learned through training. The method should always sample and return only the word with the highest probability. 
    - **infer:** a method that accepts two arguments. The first argument is input, being a single word (string). The second is n, being the number of words to generate. The method will autoregressively predict the next word based on the previous one using the predict_next_word method and return the full string of n generated words. 
    - **save:** method that serializes the trained bigram using pickle and saves it to the path provided in __init__. 
3. Your code must have a command-line interface implemented using the built-in argparse library with: 
    - The first positional (required) argument should be a selector for which activity to perform. Options should be “general”, “train”, and “inference.” 
        1) If the user selects “general,” no other arguments need to be provided 
        2) If the user selects “train,” then --data and --save need to be provided 
        3) If the user selects “inference,” then --load, --word, and --n need to be provided 
    - An argument (--data) that points to the path (including filename and extension) to the training data corpus. This will only be used if the user selects the “train” activity. 
    - An argument (--save) that points to the path where the bigram model can be saved so that it can be loaded in the “inference” activity. The model must be serialized and saved using pickle. 
    - An argument (--load) that points to the path (string) where the trained bigram model was saved. The model object can be loaded using pickle. 
    - A string argument (--word) that specifies the first word (or words) used for the “inference” activity. 
    - An integer argument (--n) that specifies the number of words to predict for the “inference” activity. 
4. All code must be documented using comments in the manner that you would do for production-level code. Refer to the PEP-8 style guide. 
### Submission Guidelines  
The code (Python file) must be submitted through Moodle by 11:59 PM on the assigned due date. The Python file will be named LastName_FirstName_csc7644_ca1.py, where LastName and FirstName are replaced with your last/first names, respectively. It is very important that method, function, and class names are not changed from those provided in the template 