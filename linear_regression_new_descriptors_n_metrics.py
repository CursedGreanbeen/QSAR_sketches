import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score, LeaveOneOut
from mordred_selection import select_optimal_combinations


def build_final_model():
    # best_combo = select_optimal_combinations()

    # if not best_combo:
    #     print("Не удалось найти подходящую комбинацию дескрипторов")
    #     return

    # best_features = best_combo['features']
    # best_features = ['Xch-7d', 'GGI7', 'IC2', 'CIC5', 'NaaN', 'SsssCH']   # топ-1
    # best_features = ['Xch-7d', 'GGI7', 'IC2', 'CIC5', 'SsssCH', 'SMR_VSA3']    # топ-2
    # best_features = ['GGI7', 'IC2', 'AATS8d', 'SlogP_VSA1', 'MDEC-33']    # топ 3
    # best_features = ['Xch-7d', 'GGI7', 'IC2', 'GATS5c', 'MDEC-33']    # топ 4
    best_features = ['GGI7', 'IC2', 'AATS8d', 'AATS5i', 'SlogP_VSA1', 'MDEC-33']
    print(f"Лучшая комбинация: {', '.join(best_features)}")

    df = pd.read_csv("data_descriptors_new.csv")
    df = df.drop(['ligand', 'mol'], axis=1, errors='ignore')
    # df_prime = df.copy()
    df_test = pd.read_csv('test_2_data_descriptors.csv')
    df_test = df_test.drop(['mol'], axis=1, errors='ignore')

    df['Energy 6wha'] = df['Energy 6wha'].astype(str).str.replace(',', '.').astype(float)
    df = df.dropna(subset=['Energy 6wha'])
    df = df.dropna()
    # df_unknown = df_prime[df_prime['Energy 6wha'].isna()]

    selected_columns = ['SMILES', 'Energy 6wha'] + best_features
    df_final = df[selected_columns]

    X = df_final[best_features]
    y = df_final['Energy 6wha']
    # x_unknown = df_unknown[best_features]
    x_test_unknown = df_test[best_features]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=32
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    # x_unknown_pred = model.predict(x_unknown)
    x_test_unknown_pred = model.predict(x_test_unknown)
    # df_unknown.loc[:, 'Predicted'] = x_unknown_pred
    df_test.loc[:, 'Predicted'] = x_test_unknown_pred

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    # Q² через кросс-валидацию на всех данных
    # q2 = cross_val_score(LinearRegression(), X, y, cv=5, scoring='r2').mean()

    print(f"\nМЕТРИКИ КАЧЕСТВА")
    print(f"R²: {r2:.3f}")
    # print(f"Q² (кросс-валидация): {q2:.3f}")
    print(f"MSE: {mse:.3f}")
    print(f"Intercept: {model.intercept_:.3f}")    # значение целевой переменной, когда все признаки равны нулю.

    def calculate_q2_simple(model, X, y):
        y_pred = model.predict(X)
        ss_res = ((y - y_pred) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        return 1 - (ss_res / ss_tot)

    q2_simple = calculate_q2_simple(model, X, y)
    print(f"Q² (упрощенный из-за малой выборки): {q2_simple:.3f}")

    comparison = pd.DataFrame({
        'Actual': y_test.values,
        'Predicted': y_pred
    }).round(3)

    print(f"\nСравнение:")
    print(comparison.head(10))

    # print(f"\n12 молекул без энергии:")
    print(f"\nПредсказание энергии для новых тестовых молекул: ")
    # for i, (smiles, pred) in enumerate(zip(df_unknown['SMILES'], x_unknown_pred)):
    for i, (smiles, pred) in enumerate(zip(df_test['SMILES'], x_test_unknown_pred)):
        print(f"{i + 1}. {smiles}: {pred:.3f}")

    results_df = pd.DataFrame({
        'SMILES': df_final.loc[y_test.index, 'SMILES'],
        'Actual': y_test.values,
        'Predicted': y_pred
    })

    results_df.to_excel('final_model_predictions.xlsx', index=False)
    print(f"\nРезультаты сохранены в 'final_model_predictions.xlsx'")

    df_test.to_excel('test_5_predictions.xlsx', index=False)

    for feat, coef in zip(best_features, model.coef_):
        print(f"{feat}: {coef:.4f}")


if __name__ == "__main__":
    build_final_model()
