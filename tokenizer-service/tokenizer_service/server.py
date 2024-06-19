import os

from flask import Flask, request, jsonify
from functools import lru_cache
from transformers import AutoTokenizer, AutoConfig, PreTrainedTokenizer
from transformers.utils import logging

CACHE_SIZE = 20
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

app = Flask(__name__)


@lru_cache(maxsize=CACHE_SIZE)
def load_tokenizer(model_name):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=HUGGINGFACE_API_TOKEN,
                                                  trust_remote_code=True)
        return tokenizer
    except Exception as e:
        app.logger.error(f"Error loading the tokenizer for model '{model_name}': {e}")
        raise ValueError(f"Error loading the tokenizer for model '{model_name}'")


def tokenize(model_name, text):
    try:
        tokenizer = load_tokenizer(model_name)
        encoding = tokenizer(text, return_offsets_mapping=True)
        tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'])
        offsets = encoding['offset_mapping']
        tokens = tokens[1:-1]
        offsets = offsets[1:-1]
        return {
            "token_count": len(tokens),
            "tokens": tokens,
            "offsets": offsets,
        }
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        app.logger.error(f"Error tokenizing text with model '{model_name}': {e}")
        return {"error": "An error occurred while tokenizing the text"}


@app.route('/api/tokenize', methods=['POST'])
def tokenize_text():
    try:
        data = request.json
        model_name = data.get('model-name')
        text = data.get('text')

        if not model_name or not text:
            return jsonify({"error": "Both 'model-name' and 'text' are required"}), 400

        result = tokenize(model_name, text)
        if 'error' in result:
            return jsonify(result), 500

        return jsonify(result), 200

    except Exception as e:
        app.logger.error(f"Unexpected error: {e}")
        return jsonify({"error": "An unexpected error occurred"}), 500


if __name__ == "__main__":
    app.run(debug=True)
