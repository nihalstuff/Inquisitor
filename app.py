from flask import Flask, jsonify, request, render_template
import spacy
from transformers import BloomTokenizerFast, BloomForCausalLM

app = Flask(__name__)

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load BLOOM model
model_name = "bigscience/bloom-7b1"
tokenizer = BloomTokenizerFast.from_pretrained(model_name)
model = BloomForCausalLM.from_pretrained(model_name)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_questions', methods=['POST'])
def get_questions():
    description = request.json['description']
    questions = generate_questions_from_description(description)
    return jsonify(questions)

@app.route('/submit_answers', methods=['POST'])
def submit_answers():
    answers = request.json['answers']
    description = request.json['description']
    crime_info = extract_crime_scene_info(description)
    analysis = analyze_conversation_history(answers, crime_info)
    guilt_percentage = calculate_guilt_percentage(analysis)
    return jsonify({'guilt_percentage': guilt_percentage})

def generate_questions_from_description(description):
    prompt = f"As a detective interrogating a suspect in connection with the following crime scene, you need to ask direct and probing questions to assess their involvement. Craft questions that inquire about their whereabouts during the crime, their relationships with other potential suspects and the victim, their possible motives, and any evidence linking them to the crime scene.\n\nCrime Scene Description:\n{description}\n\nQuestions:\n1."
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs,
        max_length=500,
        num_return_sequences=1,
        no_repeat_ngram_size=2,  # Helps to reduce repetition
        temperature=0.7,  # Controls randomness in generation
        top_p=0.9,  # Nucleus sampling
        top_k=50,  # Top-k sampling
        pad_token_id=tokenizer.eos_token_id
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    questions = text.split('\n')

    # Filter to ensure entries are questions
    filtered_questions = [q.strip() for q in questions if q.strip() and q.strip().endswith('?')]

    # Return up to 15 questions
    return filtered_questions[:15]

def extract_crime_scene_info(description):
    doc = nlp(description)
    entities = {ent.label_: [] for ent in doc.ents}
    for ent in doc.ents:
        entities[ent.label_].append(ent.text)
    return entities

def analyze_conversation_history(responses, crime_info):
    analysis = {"suspects": 0, "evidence": 0, "events": 0}
    for response in responses:
        doc = nlp(response)
        for ent in doc.ents:
            if ent.text in crime_info.get("PERSON", []):
                analysis["suspects"] += 1
            if ent.text in crime_info.get("ORG", []) + crime_info.get("GPE", []) + crime_info.get("LOC", []):
                analysis["events"] += 1
            if ent.text in crime_info.get("NORP", []) + crime_info.get("FAC", []) + crime_info.get("PRODUCT", []) + crime_info.get("EVENT", []) + crime_info.get("WORK_OF_ART", []) + crime_info.get("LAW", []) + crime_info.get("LANGUAGE", []):
                analysis["evidence"] += 1
    return analysis

def calculate_guilt_percentage(analysis):
    guilt_score = (analysis["suspects"] * 10 + analysis["evidence"] * 5 + analysis["events"] * 5)
    return min(guilt_score, 100)  # Cap the score at 100%

if __name__ == '__main__':
    app.run(debug=True)
