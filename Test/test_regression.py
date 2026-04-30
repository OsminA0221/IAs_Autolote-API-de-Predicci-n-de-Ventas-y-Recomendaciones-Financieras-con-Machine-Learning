
from ias.ml_models import RegressionAI

# Inicializar IA de regresión
reg_ai = RegressionAI(model_type='decision_tree')

# Entrenar modelos
resultados = reg_ai.train_models()
print("Resultados del entrenamiento:")
print(resultados)

# Predecir los próximos 6 meses
predicciones = reg_ai.predict_future_months(n_months=6)
print("\nPredicciones para los próximos 6 meses:")
for pred in predicciones:
    print(pred)
