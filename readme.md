# Question and Answer on PDF File

## Introduction
------------
Its is a simple application to perfomr QnA on providing a pdf file.

## How It Works
------------


The application follows these steps to provide responses to your questions:

1. PDF Loading: The app reads multiple PDF documents and extracts their text content.

2. Text Chunking: The extracted text is divided into smaller chunks that can be processed effectively.

3. Language Model: The application utilizes a language model to generate vector representations (embeddings) of the text chunks.

4. Similarity Matching: When you ask a question, the app compares it with the text chunks and identifies the most semantically similar ones.

5. Response Generation: The selected chunks are passed to the language model, which generates a response based on the relevant content of the PDFs.


## All Endpoints Usage
------------


1. /file [POST]    Get the file as an input and extract the text data and loads into the model.

2. /default_csv [GET]   Loads the default CSV by taking filename as query parameter.

3. /ask_question [POST]    Takes question as an input and input question into ML model and return ML models output as an answer(result).


   
## Data Processing
------------


1. For provided csv data I have made a separate csv according to the file names.

2. Each file is put as an input individually.

## Improvements
------------


1. I could have made a application with good rate-limiting feature becaus it is a file operation it better to avoid unnecessary IO operations.

2. Could have made better research on selection ML models and designs.

3. Focused on session management on multiple devices.

4. Could used fastapi as it provides good api response time.


