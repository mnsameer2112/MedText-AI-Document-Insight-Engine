from flask import Flask, render_template, request
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from huggingface_hub import login
from PIL import Image
import easyocr
import io
import pymupdf
import fitz  # PyMuPDF
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Authenticate with your Hugging Face API token
login(token="hf_YOURTOKEN")


model_name = "ruslanmv/Medical-Llama3-8B"
device_map = 'auto'
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    trust_remote_code=True,
    use_cache=False,
    device_map=device_map
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

def askme(question, context):
    sys_message = '''
    You are an AI Medical Assistant trained on a vast dataset of health information. Please be thorough and
    provide an informative answer. If you don't know the answer to a specific medical inquiry, advise seeking professional help.
    I will provide you with my medical document alongside a set of queries.
    Respond to each question sequentially, in relevance with the provided document.
    If you don't know the answer to a specific medical inquiry, advise seeking professional help.
    Provide only the relevant answer and refrain from disclosing personal information like name, age, address, etc.
    '''

    prompt = f"Medical document:\n{context}\nQuestion(s):\n{question}"
    messages = [{"role": "system", "content": sys_message}, {"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=600,  # Adjusted to a reasonable length
        use_cache=True,
        temperature=0.85,  # Adjusted for randomness
        # top_k=50,  # Top-k sampling
        # top_p=0.9,  # Top-p sampling
        repetition_penalty=1.15  # Repetition penalty
    )
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    print("RESPONSE TEXT:")
    print(response_text)

    start_marker = '<|im_start|>assistant'
    end_marker1 = '<|im_end|>'
    end_marker2 = 'Question(s):'

    # Find the start marker index
    start_index = response_text.find(start_marker)
    if start_index == -1:
        extracted_words = ""  # Set to empty string if start marker not found
    else:
        # Find the end marker index after the start marker
        end_index1 = response_text.find(end_marker1, start_index + len(start_marker))
        end_index2 = response_text.find(end_marker2, start_index + len(start_marker))
        print(f"End marker 1:{end_index1} End marker 2:{end_index2}")

        if end_index1 == -1 or end_index2 == -1:  # If at least one is not found
            end_index = max(end_index1, end_index2)  # Take the valid one (or -1)
        elif end_index1 == -1:
            end_index = end_index2
        elif end_index2 == -1:
            end_index = end_index1
        else:
            end_index = min(end_index1, end_index2)  # Both found, take earlier one

        if end_index == -1:  # No end marker found after start
            return response_text[start_index + len(start_marker):].strip()
    return response_text[start_index + len(start_marker):end_index].strip()



def extract_text_from_image(image_path):
    # Initialize EasyOCR reader
    reader = easyocr.Reader(['en'])

    # Read text from image
    result = reader.readtext(image_path)

    # Extract recognized text
    recognized_text = [entry[1] for entry in result]

    # Join the recognized text into a single string
    extracted_text = '\n'.join(recognized_text)

    return extracted_text



def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

@app.route('/', methods=['GET', 'POST'])
def home():
    answer = ''
    if request.method == 'POST':
        query = request.form['query']
        file = request.files['file']
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            if file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                context = extract_text_from_image(file_path)
            elif file.filename.lower().endswith('.pdf'):
                context = extract_text_from_pdf(file_path)
            else:
                context = "Unsupported file type."
            answer = askme(query, context)
        else:
            answer = "Please upload a valid image or PDF file."
    return render_template('index.html', answer=answer)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)



