import os
import typing

from flask import Flask, request, jsonify
from functools import lru_cache
from transformers import AutoTokenizer, PreTrainedTokenizer  # type: ignore

CACHE_SIZE = 20
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

# Flask is the quickest way to prototype an experimental service
app = Flask(__name__)


@lru_cache(maxsize=CACHE_SIZE)
def load_tokenizer(model_name: str) -> PreTrainedTokenizer:
    """
    Load tokenizer from HF

    It would be nice if there was a way to query their model hub instead trying and excepting
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_auth_token=HUGGINGFACE_API_TOKEN, trust_remote_code=True
        )
        return tokenizer
    except Exception as e:
        app.logger.error(f"Error loading the tokenizer for model '{model_name}': {e}")
        raise ValueError(f"Error loading the tokenizer for model '{model_name}'")


def tokenize(
    model_name: str, text: str
) -> typing.Optional[tuple[int, list[str], list[tuple[int, int]]]]:
    try:
        tokenizer = load_tokenizer(model_name)
        encoding = tokenizer(text, return_offsets_mapping=True)
        tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"])
        offsets = encoding["offset_mapping"]
        # ignore CLS and SEP
        tokens = tokens[1:-1]
        offsets = offsets[1:-1]
        return (len(tokens), tokens, offsets)
    except ValueError as e:
        return None
    except Exception as e:
        app.logger.error(f"Error tokenizing text with model '{model_name}': {e}")
        return None


# Request schema -
#   model-name: str
#   text: str
# Response schema -
#   token_count: Optional[int]
#   tokens: Optional[list[str]]
#   offsets: Optional[list[tuple[int, int]]]
#   error: Optional[str]
# ether error or the other three fields are not None
@app.route("/api/tokenize", methods=["POST"])
def tokenize_text():
    try:
        data: dict = request.json
        model_name: typing.Optional[str] = data.get("model-name")
        text: typing.Optional[str] = data.get("text")

        if not model_name or not text:
            return jsonify({"error": "Both 'model-name' and 'text' are required"}), 400

        tokenization_result: typing.Optional[
            tuple[int, list[str], list[tuple[int, int]]]
        ] = tokenize(model_name, text)
        if tokenization_result is None:
            return jsonify({"error": "There was an error during tokenization"}), 500

        tokens_count, tokens, offsets = tokenization_result
        return (
            jsonify(
                {
                    "token_count": tokens_count,
                    "tokens": tokens,
                    "offsets": offsets,
                }
            ),
            200,
        )

    except Exception as e:
        app.logger.error(f"Unexpected error: {e}")
        return jsonify({"error": "An unexpected error occurred"}), 500


if __name__ == "__main__":
    app.run(debug=True)
