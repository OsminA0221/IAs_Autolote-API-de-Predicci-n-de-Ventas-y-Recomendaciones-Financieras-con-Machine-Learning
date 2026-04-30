# IAs_Autolote — API de Predicción y Recomendaciones Financieras

API desarrollada en Python que implementa modelos de Machine Learning para la **predicción de ingresos y ventas** y la **generación de recomendaciones financieras automatizadas**.

Este proyecto corresponde a la capa de Inteligencia Artificial de un sistema de gestión para autolotes, enfocado en **analítica predictiva y toma de decisiones basada en datos**.

---

##  Descripción general

El sistema consume datos históricos de ventas desde una base de datos PostgreSQL, entrena modelos de regresión y genera:

* Predicciones de ingresos futuros
* Estimaciones de volumen de ventas
* Recomendaciones financieras basadas en métricas del negocio

La API está diseñada para integrarse fácilmente con aplicaciones frontend (por ejemplo, React) u otros servicios.

---

## Características principales

*  Predicción de ingresos y ventas por periodos mensuales
*  Modelos de Machine Learning (Decision Tree y Random Forest)
*  Evaluación de modelos con métricas (MAE, RMSE, R²)
*  Generación de recomendaciones financieras automatizadas
*  API REST lista para integración
*  Arquitectura modular y escalable

---

## 🏗️ Arquitectura del proyecto

```
ias/
├── __init__.py
├── App.py              # API REST (Flask)
├── ml_models.py        # Lógica de Machine Learning
├── models.py           # Modelos y conexión a BD (SQLAlchemy)
└── utils.py            # Validaciones y utilidades
```

### Componentes:

* **Capa API**: Manejo de rutas y solicitudes HTTP
* **Capa de IA**: Entrenamiento, evaluación y predicción
* **Capa de datos**: Acceso a base de datos mediante ORM
* **Utilidades**: Validaciones y funciones auxiliares

---

## ⚙️ Tecnologías utilizadas

* Python
* Flask
* SQLAlchemy
* PostgreSQL
* scikit-learn
* pandas
* numpy

---

## 📡 Endpoints principales

###  Verificación de conexión

```
GET /test_db
```

Verifica la conexión con la base de datos y retorna el número de registros.

---

###  Entrenamiento y predicción

```
POST /regression/train_and_predict
```

Entrena los modelos de Machine Learning y genera predicciones.

**Body:**

```json
{
  "n_months": 6
}
```

---

###  Recomendaciones financieras

```
POST /recommendations
```

Genera recomendaciones financieras en función de las métricas enviadas por el cliente.

---

## Flujo de funcionamiento

1. El cliente envía una solicitud a la API
2. Se validan los datos de entrada
3. Se consultan datos históricos desde PostgreSQL
4. Se entrenan los modelos de Machine Learning
5. Se generan predicciones
6. Se analizan métricas financieras
7. Se devuelve la respuesta en formato JSON

---

## ⚙️ Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/tu-usuario/ias-autolote.git
cd ias-autolote
```

---

### 2. Crear entorno virtual

```bash
python -m venv .venv
```

Activar entorno:

* Windows:

```bash
.venv\Scripts\activate
```

* Linux/macOS:

```bash
source .venv/bin/activate
```

---

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

---

### 4. Ejecutar la API

```bash
python -m ias.App
```

---

##  Requisitos de base de datos

* PostgreSQL
* Esquema: `ac`
* Tabla principal: `Sale`

Campos esperados:

* `SaleId`
* `VehicleId`
* `Fecha`
* `Precio`

---

##  Casos de uso

* Proyección de ingresos en negocios de venta de vehículos
* Análisis de tendencias de ventas
* Optimización de decisiones financieras
* Apoyo a estrategias basadas en datos

---

##  Valor del proyecto

Este proyecto demuestra:

* Aplicación de Machine Learning en un contexto real de negocio
* Integración de modelos predictivos en una API REST
* Procesamiento y análisis de datos históricos
* Diseño de soluciones escalables y mantenibles
* Enfoque en generación de valor a partir de datos
