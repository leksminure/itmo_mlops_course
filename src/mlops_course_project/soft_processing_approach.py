import pandas as pd
from cleantext import clean
import nltk
from sklearn.model_selection import train_test_split
from pandarallel import pandarallel
from pathlib import Path
import os 

import argparse

pandarallel.initialize()
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
            lower=False,
            no_line_breaks=False,
            no_urls=False,
            no_emails=False,
            no_phone_numbers=False,
            no_numbers=False,
            no_digits=False,
            no_currency_symbols=False,
            no_punct=False,
            lang='ru'
        )

    for block in ("title", "text"):
        df[block] = df[block].parallel_apply(clean_russian_news)

    def process_russian_text(text):
        tokens = nltk.word_tokenize(text, language='russian')
        filtered = [t for t in tokens if len(t) > 1]
        return " ".join(filtered)

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