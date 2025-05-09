import pandas as pd
from cleantext import clean
import nltk
from nltk.corpus import stopwords
import pymorphy3
from sklearn.model_selection import train_test_split
from pandarallel import pandarallel
import argparse
from pathlib import Path
import os
pandarallel.initialize()
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

RANDOM_STATE = 42

def process_data(path):
    df = pd.read_csv(path)

    def clean_russian_news(text):
        return clean(
            text,
            fix_unicode=True,
            to_ascii=False,
            lower=True,
            no_line_breaks=True,
            no_urls=True,
            no_emails=True,
            no_phone_numbers=True,
            no_numbers=False,
            no_digits=False,
            no_currency_symbols=False,
            no_punct=True,
            replace_with_url="<URL>",
            replace_with_email="<EMAIL>",
            replace_with_phone_number="<PHONE>",
            replace_with_punct="",
            lang='ru'
        )

    for block in ("title", "text"):
        df[block] = df[block].parallel_apply(clean_russian_news)

    russian_stopwords = set(stopwords.words('russian'))
    morph_analyzer = pymorphy3.MorphAnalyzer()

    def process_russian_text(text):
        tokens = nltk.word_tokenize(text, language='russian')
        filtered_tokens = [
            token for token in tokens 
            if token.lower() not in russian_stopwords and token.isalpha()
        ]
        lemmatized_tokens = [
            morph_analyzer.parse(token)[0].normal_form 
            for token in filtered_tokens
        ]
        return " ".join(lemmatized_tokens)

    for block in ("title", "text"):
        df[block] = df[block].parallel_apply(process_russian_text)

    df.dropna(subset=["text"], inplace=True)

    train_df, temp_df = train_test_split(
        df,
        test_size=0.4,
        stratify=df["topic"],
        random_state=RANDOM_STATE
    )
    
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df["topic"],
        random_state=RANDOM_STATE
    )
    
    return train_df, val_df, test_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path')
    parser.add_argument('--output-path')
    args = parser.parse_args()
    train_df, val_df, test_df = process_data(args.input_path)

    Path(args.output_path).mkdir(parents=True, exist_ok=True)

    train_df.to_csv(os.path.join(args.output_path, "train_df.csv"), index=False)
    val_df.to_csv(os.path.join(args.output_path, "val_df.csv"), index=False)
    test_df.to_csv(os.path.join(args.output_path, "test_df.csv"), index=False)