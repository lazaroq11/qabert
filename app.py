from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline
import os

app = Flask(__name__)
CORS(app)

qa_context = ""
qa_model = pipeline('question-answering', model='distilbert-base-cased-distilled-squad', tokenizer='distilbert-base-cased-distilled-squad')

@app.route('/set_context', methods=['POST'])
def set_context():
    global qa_context
    data = request.json
    qa_context = data.get('context', '')
    return jsonify({'message': 'Contexto definido com sucesso'})

@app.route('/chat', methods=['POST'])
def chat():
    global qa_context
    data = request.json
    user_input = data['user_input']
    model_type = data.get('model_type', 'gpt')  # Padrão para geração de texto

    if model_type == 'qa':
        try:
            context = qa_context
            answer = qa_model({'question': user_input, 'context': context})
            response = answer['answer']
        except Exception as e:
            response = f"Erro ao processar a pergunta: {str(e)}"
    else:
        response = "Modelo não suportado"

    return jsonify({'response': response})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
