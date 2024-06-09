# RAG using Python

A Python-based implementation of Retrieval-Augmented Generation (RAG) for interacting with PDF documents through a conversational interface.

## Overview

This project leverages OpenAI embeddings, FAISS for vector storage, and Streamlit for creating a web application that allows users to chat with the contents of uploaded PDF files.

## Features

- Extract text from PDF files
- Split text into manageable chunks
- Generate embeddings using OpenAI
- Store embeddings in FAISS vector store
- Conversational interface using Streamlit

## Installation

### Prerequisites

- Python 3.x
- Required libraries (install using `pip`)

### Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/saiprasaad/RAG-using-python.git
    ```

2. Navigate to the project directory:
    ```bash
    cd RAG-using-python
    ```

3. Create a `.env` file and add your OpenAI API key:
    ```env
    OPENAI_API_KEY=your_openai_api_key
    ```

## Usage

Run the Streamlit app:
```bash
streamlit run app.py
