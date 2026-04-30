print("Debug: Iniciando App.py...")
from flask import Flask, request, jsonify
from flask_cors import CORS
from .ml_models import RegressionAI, FinancialRecommendationsAI
from .utils import validate_positive_int
from .models import session, Sale

app = Flask("IAs_Autolote")
CORS(app, resources={r"/*": {"origins": "*"}})

regression_ai = RegressionAI()
recommendations_ai = FinancialRecommendationsAI()

@app.route('/test_db', methods=['GET'])
def test_db():
    try:
        count = session.query(Sale).count()
        return jsonify({'message': f'Conectado. Ventas en BD: {count}'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/regression/train_and_predict', methods=['POST'])
def train_and_predict():
    try:
        data = request.json or {}
        n_months = validate_positive_int(data.get('n_months', 6), 'n_months') or 6  # prediccion a N meses
        # Entrena y predice (la IA carga datos internamente)
        training_results = regression_ai.train_models()  # entrena
        predictions = regression_ai.predict_future_months(n_months)  # predice
        return jsonify({
            'training_results': training_results,
            'predictions': predictions
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 400

@app.route('/recommendations', methods=['POST'])
def get_recommendations():
    try:
        data = request.json or {}
        metrics = data.get('metrics', {})
        if not metrics:
            raise ValueError("Debe proporcionar las metricas calculadas en 'metrics'.")
        recommendation = recommendations_ai.generate_recommendation(metrics)
        return jsonify({
            'metrics_recibidas': metrics,
            'recommendation': recommendation
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
