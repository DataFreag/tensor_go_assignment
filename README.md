# TensorGO Assignment
## Requirements
Before using the code, ensure the following dependencies:

- Python 3.7
- Required Python Packages install using '**pip install -r requirements.txt**'
- Requires the command-line tool [`FFmpeg`](https://ffmpeg.org/) to be installed on your system
- Requires OpenAI API key for question-answering chain. We will be using GPT 3.5 for this assignment, its API key can be accessed by signing up [here](https://openai.com/blog/openai-api)
## Description
The objective of this project is to enhance RAG by building a multilingual speech recognition system.
The goal is to enable RAG to perform various tasks, including speech recognition, translation, and summarization, in multiple languages without the need for additional training.
This will be achieved by leveraging a pre-trained multilingual speech recognition model, specifically Multilingual Whisper.
<br><br>
RAG follows a recall-augmented approach, combining elements of retrieval-based and generative model.
The Retriever will have Query Encoder and Document Index and Generative will be a seq2seq model.
The document index will be transcribed text from ASR whisper model.
## Installation
1. **Clone the repository**

   `git clone https://github.com/DataFreag/tensor_go_assignment.git`

2. **Install dependencies**

   `pip install -r requirements.txt`

3. **Install FFmpeg**
   The Whisper ASR model requires FFmpeg.<br>
   For windows download FFmpeg from the official website[`https://ffmpeg.org/download.html`](https://ffmpeg.org/download.html) and add its bin directory to your system's PATH.
## Usage
1. Load your CharOpenAI API key by editing the JSON file (key.json) and `API_KEY` parameter.
2. Specify the audio path in the main script.
3. Specify the language to be translate in the main script.
4. Run the main script.
## Code Explanation
The code is structured as follows:

- **main.py:** The main script to run the multilingual speech recognition system with RAG. It consists of the following sections:
  1. The code uses Whisper model from Whisper package to transcribe and print the audio specified in the input.
  2. The transcription will be splitted into chunks for document generation (**Recursive Character Text Splitter** is used).
  3. The generated documents will be embedded using a hugging face embedding model to create a Chroma vector
  4. This chroma vector will be used as retriever vector database which is transcipted audio files
  5. Creating a question-answering chain using the retriver and api_key to complete RAG chain
  6. We can pass questions to this question answering chain to get results related to vector database

- **key.json:** This file used to store configuration parameters, like API key.
- **requirements.txt:** file lists all the Python packages required for the project.
