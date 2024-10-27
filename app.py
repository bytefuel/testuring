import streamlit as st
import os
import faiss
from sentence_transformers import SentenceTransformer
import tree_sitter_python as tspython
from tree_sitter import Language, Parser
from openai import OpenAI
from github import Github

# Initialize OpenAI client and Sentence Transformer model
client = OpenAI()
client.api_key = os.getenv("OPENAI_API_KEY")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Parse codebase using Tree-sitter
def parse_code(filepath):
    PY_LANGUAGE = Language(tspython.language())
    parser = Parser(PY_LANGUAGE)
    with open(filepath, 'r') as f:
        code = f.read()
    tree = parser.parse(bytes(code, 'utf8'))
    root_node = tree.root_node

    # Extract function definitions
    def extract_functions(node):
        functions = []
        if node.type == 'function_definition':
            func_name = node.child_by_field_name('name').text.decode('utf8')
            start_line = node.start_point[0]
            end_line = node.end_point[0]
            functions.append((func_name, start_line, end_line))
        for child in node.children:
            functions.extend(extract_functions(child))
        return functions

    return extract_functions(root_node)

# Embed code snippets
def embed_code_snippets(functions, code_lines):
    snippets = []
    for func_name, start, end in functions:
        snippet = '\n'.join(code_lines[start:end + 1])
        snippets.append((func_name, snippet))
    snippets_text = [s[1] for s in snippets]
    embeddings = model.encode(snippets_text, convert_to_tensor=True)
    return snippets, embeddings

# Store embeddings in FAISS
def store_embeddings_in_faiss(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.cpu().detach().numpy())
    return index

# Query FAISS for relevant code snippets
def query_faiss(index, query, snippets):
    query_embedding = model.encode([query], convert_to_tensor=True)
    D, I = index.search(query_embedding.cpu().detach().numpy(), k=5)
    return [snippets[i] for i in I[0]]

# Generate test case with GPT
def generate_test_case_with_gpt(query, code_snippets):
    snippets_text = '\n'.join([snippet for _, snippet in code_snippets])
    prompt = (
        f"Here is the relevant code:\n\n"
        f"```python\n{snippets_text}\n```\n\n"
        f"Generate a test case for {query}.\n"
        "Please respond with the test case code only, without any additional explanations."
    )
    stream = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    output = ""
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            output += chunk.choices[0].delta.content
    return output

# Streamlit App UI
st.title("GitHub Code QA")
github_repo = st.text_input("Enter GitHub repository URL:")
query = st.text_input("Enter your question (e.g., 'Generate a test case for user login'):")

if st.button("Analyze and Generate Test Case"):
    if github_repo and query:
        g = Github(os.getenv("GITHUB_TOKEN"))
        repo_name = github_repo.split("github.com/")[-1]
        repo = g.get_repo(repo_name)
        code_lines = []
        for content_file in repo.get_contents(""):
            if content_file.path.endswith(".py"):  # Only parse Python files
                file_content = content_file.decoded_content.decode('utf-8')
                code_lines.extend(file_content.splitlines())
        with open("temp_code.py", "w") as temp_code_file:
            temp_code_file.write("\n".join(code_lines))
        functions = parse_code("temp_code.py")
        snippets, embeddings = embed_code_snippets(functions, code_lines)
        index = store_embeddings_in_faiss(embeddings)
        relevant_snippets = query_faiss(index, query, snippets)
        test_case = generate_test_case_with_gpt(query, relevant_snippets)
        st.code(test_case, language="python")
    else:
        st.error("Please enter both GitHub repository URL and a query.")
