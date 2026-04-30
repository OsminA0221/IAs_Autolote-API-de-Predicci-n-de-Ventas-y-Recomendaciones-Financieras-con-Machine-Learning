import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from datetime import datetime
from dateutil.relativedelta import relativedelta
from .models import session, Sale
import logging

logging.basicConfig(level=logging.INFO)


def _clean_typeid(val):
    """Devuelve 1 si es credito, 0 si es contado."""
    if val is None:
        return 0
    s = str(val).strip().upper()
    if s in ['2', 'CRÉDITO', 'CREDITO', 'CREDITO', 'CREDITO']:
        return 1
    try:
        return 1 if int(float(s)) == 2 else 0
    except Exception:
        return 0


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Precio'] = df['Precio'].fillna(0)
    df['year_month'] = pd.to_datetime(df['Fecha']).dt.to_period('M')
    df_agg = df.groupby('year_month').agg(
        ingresos=('Precio', 'sum'),
        ventas=('Precio', 'count'),
        precio_prom=('Precio', 'mean')
    ).reset_index()
    df_agg['month_number'] = range(len(df_agg))
    df_agg['month_of_year'] = df_agg['year_month'].dt.month
    df_agg['rm_ingresos_3'] = df_agg['ingresos'].rolling(window=3, min_periods=1).mean()
    df_agg['rm_ingresos_6'] = df_agg['ingresos'].rolling(window=6, min_periods=1).mean()
    df_agg['rm_ventas_3'] = df_agg['ventas'].rolling(window=3, min_periods=1).mean()
    df_agg['rm_ventas_6'] = df_agg['ventas'].rolling(window=6, min_periods=1).mean()
    df_agg['rm_precio_3'] = df_agg['precio_prom'].rolling(window=3, min_periods=1).mean()
    df_agg['rm_precio_6'] = df_agg['precio_prom'].rolling(window=6, min_periods=1).mean()
    df_agg['precio_prom'] = df_agg['precio_prom'].fillna(df_agg['precio_prom'].mean() or 0)
    return df_agg


class RegressionAI:
    def __init__(self, model_type='decision_tree', max_leaf_nodes=None):
        self.model_type = model_type
        self.max_leaf_nodes = max_leaf_nodes
        # modelos por tipo: contado y credito
        self.models = {
            'contado': {'ing': None, 'ven': None, 'base_ing': None, 'base_ven': None},
            'credito': {'ing': None, 'ven': None, 'base_ing': None, 'base_ven': None},
        }

    def _get_model(self):
        if self.model_type == 'decision_tree':
            return DecisionTreeRegressor(max_leaf_nodes=self.max_leaf_nodes, random_state=42)
        elif self.model_type == 'random_forest':
            return RandomForestRegressor(random_state=42)
        else:
            raise ValueError("Tipo de modelo no valido.")

    def _evaluate_model(self, y_true, y_pred):
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float('nan')
        return {'MAE': mae, 'RMSE': rmse, 'R2': r2}

    def _load_raw(self):
        today = datetime.now().date()
        query = session.query(Sale).filter(Sale.Fecha <= today)
        df = pd.read_sql(query.statement, session.bind)
        df = df.dropna(subset=['Fecha'])
        if df.empty:
            raise ValueError("No hay datos de ventas hasta la fecha actual.")
        df['es_credito'] = df.get('TypeId', pd.Series([None]*len(df))).apply(_clean_typeid)
        return df

    def _train_single(self, df_agg):
        resultados = {}
        feature_cols = [
            'month_number', 'month_of_year',
            'rm_ingresos_3', 'rm_ingresos_6',
            'rm_ventas_3', 'rm_ventas_6',
            'precio_prom', 'rm_precio_3', 'rm_precio_6'
        ]
        X = df_agg[feature_cols].fillna(df_agg[feature_cols].mean())
        y_ingresos = df_agg['ingresos']
        y_ventas = df_agg['ventas']

        if len(df_agg) < 2:
            modelo_ing = None
            modelo_ven = None
            base_ing = float(y_ingresos.mean())
            base_ven = float(y_ventas.mean())
            resultados['ingresos'] = {'MAE': 0.0, 'RMSE': 0.0, 'R2': float('nan')}
            resultados['ventas'] = {'MAE': 0.0, 'RMSE': 0.0, 'R2': float('nan')}
            return modelo_ing, modelo_ven, base_ing, base_ven, resultados

        X_train, X_val, y_train, y_val = train_test_split(X, y_ingresos, test_size=0.2, random_state=42)
        modelo_ing = self._get_model()
        modelo_ing.fit(X_train, y_train)
        pred_ing = modelo_ing.predict(X_val)
        resultados['ingresos'] = self._evaluate_model(y_val, pred_ing)

        X_train, X_val, y_train, y_val = train_test_split(X, y_ventas, test_size=0.2, random_state=42)
        modelo_ven = self._get_model()
        modelo_ven.fit(X_train, y_train)
        pred_ven = modelo_ven.predict(X_val)
        resultados['ventas'] = self._evaluate_model(y_val, pred_ven)

        base_ing = float(y_ingresos.mean())
        base_ven = float(y_ventas.mean())
        return modelo_ing, modelo_ven, base_ing, base_ven, resultados

    def train_models(self):
        df_raw = self._load_raw()
        df_contado_raw = df_raw[df_raw['es_credito'] == 0]
        df_credito_raw = df_raw[df_raw['es_credito'] == 1]

        df_contado = _aggregate(df_contado_raw) if not df_contado_raw.empty else pd.DataFrame()
        df_credito = _aggregate(df_credito_raw) if not df_credito_raw.empty else pd.DataFrame()

        resultados = {'contado': {}, 'credito': {}}

        # Contado
        if df_contado.empty:
            self.models['contado'] = {'ing': None, 'ven': None, 'base_ing': 0.0, 'base_ven': 0.0}
            resultados['contado'] = {'ingresos': {'MAE': 0, 'RMSE': 0, 'R2': float('nan')},
                                     'ventas': {'MAE': 0, 'RMSE': 0, 'R2': float('nan')}}
        else:
            m_ing, m_ven, b_ing, b_ven, res = self._train_single(df_contado)
            self.models['contado'] = {'ing': m_ing, 'ven': m_ven, 'base_ing': b_ing, 'base_ven': b_ven}
            resultados['contado'] = res

        # Credito
        if df_credito.empty:
            self.models['credito'] = {'ing': None, 'ven': None, 'base_ing': 0.0, 'base_ven': 0.0}
            resultados['credito'] = {'ingresos': {'MAE': 0, 'RMSE': 0, 'R2': float('nan')},
                                     'ventas': {'MAE': 0, 'RMSE': 0, 'R2': float('nan')}}
        else:
            m_ing, m_ven, b_ing, b_ven, res = self._train_single(df_credito)
            self.models['credito'] = {'ing': m_ing, 'ven': m_ven, 'base_ing': b_ing, 'base_ven': b_ven}
            resultados['credito'] = res

        logging.info("Modelos por tipo entrenados.")
        return resultados

    def _predict_for_type(self, df_raw, tipo_clave, n_months):
        df = _aggregate(df_raw) if not df_raw.empty else pd.DataFrame()
        history_len = len(df)
        # Si no hay datos, retorna vacio
        if df.empty:
            return []

        last_period = df['year_month'].max()
        try:
            last_month_date = last_period.to_timestamp() if hasattr(last_period, 'to_timestamp') else datetime.now()
        except Exception:
            last_month_date = datetime.now()
        if pd.isna(last_month_date):
            last_month_date = datetime.now()

        last_month_number = df['month_number'].max()
        try:
            last_month_number = int(last_month_number)
        except Exception:
            last_month_number = 0
        if pd.isna(last_month_number):
            last_month_number = 0

        last_rm_ing3 = float(df['rm_ingresos_3'].iloc[-1])
        last_rm_ing6 = float(df['rm_ingresos_6'].iloc[-1])
        last_rm_ven3 = float(df['rm_ventas_3'].iloc[-1])
        last_rm_ven6 = float(df['rm_ventas_6'].iloc[-1])
        last_precio_prom = float(df['precio_prom'].iloc[-1])
        last_rm_precio_3 = float(df['rm_precio_3'].iloc[-1])
        last_rm_precio_6 = float(df['rm_precio_6'].iloc[-1])

        modelo_ing = self.models[tipo_clave]['ing']
        modelo_ven = self.models[tipo_clave]['ven']
        base_ing = self.models[tipo_clave]['base_ing']
        base_ven = self.models[tipo_clave]['base_ven']

        predictions = []
        for i in range(1, n_months + 1):
            future_month_date = last_month_date + relativedelta(months=i)
            X_future = pd.DataFrame({
                'month_number': [last_month_number + i],
                'month_of_year': [future_month_date.month],
                'rm_ingresos_3': [last_rm_ing3],
                'rm_ingresos_6': [last_rm_ing6],
                'rm_ventas_3': [last_rm_ven3],
                'rm_ventas_6': [last_rm_ven6],
                'precio_prom': [last_precio_prom],
                'rm_precio_3': [last_rm_precio_3],
                'rm_precio_6': [last_rm_precio_6],
            })
            if modelo_ing is None or modelo_ven is None:
                pred_ingresos = float(base_ing)
                pred_ventas = float(base_ven)
            else:
                pred_ingresos = float(modelo_ing.predict(X_future)[0])
                pred_ventas = float(modelo_ven.predict(X_future)[0])
            predictions.append({
                'month': future_month_date.strftime('%Y-%m'),
                'pred_ingresos': pred_ingresos,
                'pred_ventas': pred_ventas,
            })

        ingresos_var = df['ingresos'].nunique()
        ventas_var = df['ventas'].nunique()
        # Permitir hasta n_months aun con poco historial; si la serie es totalmente plana, recorta suavemente
        if ingresos_var <= 1 and ventas_var <= 1:
            return predictions[: min(len(predictions), 3)]
        return predictions

    def predict_future_months(self, n_months=6):
        try:
            n_months = int(n_months)
            if n_months <= 0:
                n_months = 6
        except Exception:
            n_months = 6

        df_raw = self._load_raw()
        df_contado_raw = df_raw[df_raw['es_credito'] == 0]
        df_credito_raw = df_raw[df_raw['es_credito'] == 1]

        preds_cont = self._predict_for_type(df_contado_raw, 'contado', n_months)
        preds_cred = self._predict_for_type(df_credito_raw, 'credito', n_months)

        # combinar por mes sumando ingresos y ventas
        combined = {}
        for p in preds_cont:
            combined.setdefault(p['month'], {'month': p['month'], 'pred_ingresos': 0.0, 'pred_ventas': 0.0})
            combined[p['month']]['pred_ingresos'] += p['pred_ingresos']
            combined[p['month']]['pred_ventas'] += p['pred_ventas']
        for p in preds_cred:
            combined.setdefault(p['month'], {'month': p['month'], 'pred_ingresos': 0.0, 'pred_ventas': 0.0})
            combined[p['month']]['pred_ingresos'] += p['pred_ingresos']
            combined[p['month']]['pred_ventas'] += p['pred_ventas']

        # ordenar meses y limitar a n_months
        sorted_months = sorted(combined.values(), key=lambda x: x['month'])

        # filtrar meses que no cambian para evitar mostrar datos planos por falta de historial
        filtered = []
        eps = 1e-6
        for pred in sorted_months:
            if not filtered:
                filtered.append(pred)
                continue
            prev = filtered[-1]
            delta_ing = abs(pred['pred_ingresos'] - prev['pred_ingresos'])
            delta_ven = abs(pred['pred_ventas'] - prev['pred_ventas'])
            if delta_ing > eps or delta_ven > eps:
                filtered.append(pred)

        # si todo plano, deja solo el primer mes para no inducir error
        if not filtered and sorted_months:
            filtered = [sorted_months[0]]

        return filtered[:n_months]


class FinancialRecommendationsAI:
    def generate_recommendation(self, metrics):
        try:
            revenue = metrics.get('revenue', 0)
            cogs = metrics.get('cogs', 0)
            gross_profit = metrics.get('grossProfit', revenue - cogs)
            expenses = metrics.get('expenses', 0)
            net_income = metrics.get('netIncome', gross_profit - expenses)
            vehicles_sold = metrics.get('vehiclesSold', 0)
            avg_days_in_inventory = metrics.get('avgDaysInInventory', 0)
            cash_flow = metrics.get('cashFlow', 0)

            gross_margin_pct = (gross_profit / revenue) * 100 if revenue > 0 else 0
            expense_ratio_pct = (expenses / revenue) * 100 if revenue > 0 else 0

            recommendation = ""

            if net_income < 0:
                recommendation = (
                    f"Prioridad ALTA: La empresa tiene perdidas netas (C$ {net_income:,.0f}). "
                    "Revisa costos, optimiza ventas y ajusta precios para lograr rentabilidad."
                )
            elif cash_flow < 0:
                recommendation = (
                    f"Prioridad ALTA: Flujo de caja negativo (C$ {cash_flow:,.0f}). "
                    "Gestiona cobros y pagos para mejorar liquidez."
                )
            elif gross_margin_pct < 15:
                recommendation = (
                    f"Prioridad MEDIA: Margen bruto bajo ({gross_margin_pct:.1f}%). "
                    "Aumenta precios o reduce COGS para mejorar rentabilidad."
                )
            elif expense_ratio_pct > 30:
                recommendation = (
                    f"Prioridad MEDIA: Gastos operativos altos ({expense_ratio_pct:.1f}% de ingresos). "
                    "Optimiza gastos para mejorar eficiencia financiera."
                )
            elif avg_days_in_inventory > 90:
                recommendation = (
                    f"Prioridad MEDIA: Inventario lento ({avg_days_in_inventory:.0f} dias). "
                    "Promociona ventas, reduce stock o ajusta compras."
                )
            elif gross_margin_pct < 25:
                recommendation = (
                    f"Mejorar: Margen bruto moderado ({gross_margin_pct:.1f}%). "
                    "Optimiza precios y costos."
                )
            elif expense_ratio_pct > 20:
                recommendation = (
                    f"Mejorar: Gastos algo elevados ({expense_ratio_pct:.1f}%). "
                    "Revisa eficiencia operativa y control de costos."
                )
            elif avg_days_in_inventory > 60:
                recommendation = (
                    f"Mejorar: Inventario con salida relativamente lenta ({avg_days_in_inventory:.0f} dias). "
                    "Evalua promociones y rotacion de stock."
                )
            else:
                recommendation = (
                    "Excelente: KPIs en rangos optimos. Manten estrategias actuales y crecimiento sostenido."
                )

            logging.info("Recomendacion generada correctamente.")
            return recommendation

        except Exception as e:
            logging.error(f"Error generando recomendacion financiera: {e}")
            raise
