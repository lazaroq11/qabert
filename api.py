from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)
app.config['CORS_HEADERS'] = ['Content-Type', 'Access-Control-Allow-Origin']
# Inicialize o contexto para o modelo de Perguntas e Respostas
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
        context = qa_context
        answer = qa_model({'question': user_input, 'context': context})
        response = answer['answer']
    else:
        # Adicione a lógica para outros modelos, se necessário
        response = "Modelo não suportado"

    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
