
import re
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import (
    ENGLISH_STOP_WORDS,
    TfidfVectorizer,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# 1. Путь к датасету
DATA_PATH = Path("data") / "SMSSpamCollection"


# 2. Очистка и нормализация текста
def clean_text(text):
    # Приводим текст к нижнему регистру
    text = str(text).lower()

    # Удаляем пунктуацию и цифры,
    # оставляя только английские буквы и пробелы
    text = re.sub(r"[^a-z\s]", " ", text)

    # Токенизация: разбиваем текст на отдельные слова
    tokens = text.split()

    # Удаляем стоп-слова
    tokens = [
        word for word in tokens
        if word not in ENGLISH_STOP_WORDS
    ]

    # Возвращаем очищенный текст
    return " ".join(tokens)


def main():
    # 3. Загружаем датасет
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Не найден файл: {DATA_PATH}. "
            "Проверь, что он находится в папке data."
        )

    df = pd.read_csv(
        DATA_PATH,
        sep="\t",
        header=None,
        names=["label", "text"],
        encoding="utf-8",
        on_bad_lines="skip",
    )

    # Оставляем только корректные строки
    df = df.dropna(subset=["label", "text"])
    df = df[df["label"].isin(["ham", "spam"])].copy()

    print("Первые 5 строк датасета:")
    print(df.head())

    print("\nКоличество сообщений:", len(df))
    print("\nКоличество сообщений каждого класса:")
    print(df["label"].value_counts())

    # 4. Применяем очистку
    df["clean_text"] = df["text"].apply(clean_text)

    print("\nПример исходного сообщения:")
    print(df["text"].iloc[0])

    print("\nПример очищенного сообщения:")
    print(df["clean_text"].iloc[0])

    # Удаляем пустые тексты после очистки
    df = df[df["clean_text"].str.strip() != ""].copy()

    X = df["clean_text"]
    y = df["label"]

    # 5. Разделяем данные на обучение и тест
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # 6. Преобразуем слова в числа с помощью TF-IDF
    vectorizer = TfidfVectorizer(
        min_df=2,
        max_df=0.95,
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print("\nРазмер обучающей матрицы:", X_train_tfidf.shape)
    print("Размер тестовой матрицы:", X_test_tfidf.shape)
    print("Количество слов в словаре:", len(vectorizer.vocabulary_))

    # 7. Обучаем классификатор
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)

    # 8. Получаем предсказания
    y_pred = model.predict(X_test_tfidf)

    print("\nТочность модели (Accuracy):")
    print(round(accuracy_score(y_test, y_pred), 4))

    print("\nОтчёт о качестве классификации:")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("Матрица ошибок (порядок классов: ham, spam):")
    print(confusion_matrix(y_test, y_pred, labels=["ham", "spam"]))

    # 9. Анализируем пять ошибок модели
    errors = pd.DataFrame({
        "text": X_test,
        "actual": y_test,
        "predicted": y_pred,
    })

    errors = errors[errors["actual"] != errors["predicted"]]

    print("\nКоличество ошибочных предсказаний:", len(errors))
    print("\nПервые 5 ошибок модели:")

    if len(errors) == 0:
        print("Ошибок на тестовой выборке нет.")
    else:
        five_errors = errors.head(5).copy()
        five_errors["hypothesis"] = five_errors.apply(
            lambda row: (
                "Спам ошибочно принят за обычное сообщение: "
                "возможно, текст похож на обычную переписку."
                if row["actual"] == "spam"
                else
                "Обычное сообщение ошибочно принято за спам: "
                "возможно, есть слова, похожие на рекламу."
            ),
            axis=1,
        )

        print(
            five_errors[
                ["text", "actual", "predicted", "hypothesis"]
            ].to_string(index=False)
        )

        five_errors.to_csv(
            "five_errors.csv",
            index=False,
            encoding="utf-8-sig",
        )
        print("\nПять ошибок сохранены в five_errors.csv")

    print("\nПрактическая работа завершена.")


if __name__ == "__main__":
    main()
