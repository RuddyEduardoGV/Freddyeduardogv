import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.neural_network import MLPRegressor
from sklearn import svm


def load_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == '.xlsx':
        try:
            df = pd.read_excel(path, engine='openpyxl', index_col='FECHA')
            return df
        except Exception as e:
            raise RuntimeError(f"No se pudo leer XLSX '{path}': {e}")
    elif ext == '.csv':
        try:
            df = pd.read_csv(path)
            # Definir índice si existe la columna FECHA
            if 'FECHA' in df.columns:
                df['FECHA'] = pd.to_datetime(df['FECHA'], errors='coerce')
                df = df.set_index('FECHA')
            return df
        except Exception as e:
            raise RuntimeError(f"No se pudo leer CSV '{path}': {e}")
    else:
        raise ValueError(f"Extensión no soportada: {ext}. Usa .xlsx o .csv")


def prepare_data(df: pd.DataFrame):
    df = df.copy()
    df = df[pd.to_numeric(df['Detalle'], errors='coerce').notnull()]
    df['Detalle'] = df['Detalle'].astype('float')

    division1_df = df.iloc[0:2886, 0:9]
    division2_df = df.iloc[2887:3608, 0:9]

    feature_cols = ['Movilveintiuno', 'Movilcincocinco', 'Movilunocuatrocuatro',
                    'Momentdiez', 'Momentsetenta', 'Momenttrescerocero']

    missing = [c for c in feature_cols + ['Detalle'] if c not in df.columns]
    if missing:
        raise KeyError(f"Faltan columnas requeridas: {missing}")

    X = np.asanyarray(division1_df[feature_cols])
    X2 = np.asanyarray(division2_df[feature_cols])

    Y = np.asanyarray(division1_df['Detalle'].astype('float'))
    Y2 = np.asanyarray(division2_df['Detalle'].astype('float'))

    return X, Y, X2, Y2


def train_models(X, Y):
    clf = svm.SVC(kernel='poly', degree=3)
    clf.fit(X, Y)

    nn = MLPRegressor(activation='logistic', hidden_layer_sizes=(200), max_iter=1000, solver='adam', random_state=42)
    nn.fit(X, Y)
    return clf, nn


def evaluate(nn: MLPRegressor, X_test, Y_test, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    y_pred = nn.predict(X_test)

    fpr, tpr, thresholds = metrics.roc_curve(Y_test, y_pred)
    auc = metrics.roc_auc_score(Y_test, y_pred)

    roc = pd.DataFrame({
        'fpr': fpr,
        'tpr': tpr,
        '1-fpr': 1 - fpr,
        'tf': tpr - (1 - fpr),
        'thresholds': thresholds,
    })
    roc_opt = roc.iloc[(roc['tf'] - 0).abs().argsort()[:1]].copy()
    opt_threshold = float(roc_opt['thresholds'].values[0])

    # Guardar predicciones y etiqueta binaria según umbral óptimo
    results = pd.DataFrame({
        'Actual': Y_test,
        'Predicted': y_pred,
        'Signal_UP': (y_pred >= opt_threshold).astype(int),
    })
    results.to_csv(os.path.join(outdir, 'predicciones.csv'), index=False)

    # Curva ROC
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC={auc:.4f}")
    plt.plot([0, 1], [0, 1], 'k--', linewidth=0.8)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc=4)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'roc_curve.png'), dpi=150)
    plt.close()

    # Métricas adicionales en CSV
    metrics_df = pd.DataFrame({
        'metric': ['AUC', 'optimal_threshold', 'mean_predicted', 'mean_actual'],
        'value': [auc, opt_threshold, float(np.mean(y_pred)), float(np.mean(Y_test))],
    })
    metrics_df.to_csv(os.path.join(outdir, 'metrics.csv'), index=False)

    return auc, opt_threshold


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='/workspace/Base1.xlsx', help='Ruta a .xlsx o .csv con columnas requeridas')
    parser.add_argument('--outdir', default='/workspace/output', help='Carpeta de salida para resultados')
    args = parser.parse_args()

    df = load_table(args.input)
    X, Y, X2, Y2 = prepare_data(df)
    clf, nn = train_models(X, Y)
    auc, opt_threshold = evaluate(nn, X2, Y2, args.outdir)
    print(f"AUC={auc:.4f}, optimal_threshold={opt_threshold:.6f}")


if __name__ == '__main__':
    main()

