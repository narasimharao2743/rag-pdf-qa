import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from rag_pipeline import load_and_index, load_existing_store, query

app = Flask(__name__)

UPLOAD_FOLDER = "./uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename.endswith(".pdf"):
        return jsonify({"error": "Only PDF files are supported"}), 400

    filename = secure_filename(file.filename)
    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)

    load_and_index(path)
    return jsonify({"message": f"{filename} indexed successfully"})


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    if not data or "question" not in data:
        return jsonify({"error": "question field is required"}), 400

    if not os.path.exists(CHROMA_DIR := "./chroma_store"):
        return jsonify({"error": "No documents indexed yet. Upload a PDF first."}), 400

    vectorstore = load_existing_store()
    result = query(data["question"], vectorstore)
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, port=8000)
