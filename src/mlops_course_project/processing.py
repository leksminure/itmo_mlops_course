import pandas as pd
from cleantext import clean
import nltk
from nltk.corpus import stopwords
import pymorphy3
from sklearn.model_selection import train_test_split
from pandarallel import pandarallel
from pathlib import Path
import os
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

pandarallel.initialize()
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

def process_data(
    path: str,
    text_column_names: list,
    target_column_name: str,
    use_lemmatize: bool,
    keep_only_nouns: bool,
    cleaning_params: dict,
    initial_test_size: float,
    val_test_ratio: float,
    random_state: int
):
    df = pd.read_csv(path)
    
    def clean_russian_news(text):
        return clean(text, **cleaning_params)

    for block in text_column_names:
        df[block] = df[block].parallel_apply(clean_russian_news)

    russian_stopwords = set(stopwords.words('russian'))
    morph_analyzer = pymorphy3.MorphAnalyzer()

    def process_russian_text(text):
        tokens = nltk.word_tokenize(text, language='russian')
        if use_lemmatize:
            filtered_tokens = [
                token for token in tokens 
                if token.lower() not in russian_stopwords and token.isalpha()
            ]
            tokens = [
                morph_analyzer.parse(token)[0].normal_form 
                for token in filtered_tokens
            ]
        else:
            tokens = [t for t in tokens if len(t) > 1]

        if keep_only_nouns:
            lemmas = []
            for t in tokens:
                parsed = morph_analyzer.parse(t)[0]
                if 'NOUN' in parsed.tag:
                    lemmas.append(parsed.normal_form)
            return " ".join(lemmas)
        return " ".join(tokens)

    for block in text_column_names:
        df[block] = df[block].parallel_apply(process_russian_text)

    df.dropna(inplace=True)

    train_df, temp_df = train_test_split(
        df,
        test_size=initial_test_size,
        stratify=df[target_column_name],
        random_state=random_state
    )
    
    val_df, test_df = train_test_split(
        temp_df,
        test_size=val_test_ratio,
        stratify=temp_df[target_column_name],
        random_state=random_state
    )
    
    return train_df, val_df, test_df

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    train_df, val_df, test_df = process_data(
        path=to_absolute_path(cfg.paths.raw_data),
        text_column_names=cfg.processing.text_columns,
        target_column_name=cfg.processing.target_column,
        use_lemmatize=cfg.processing.lemmatization.enabled,
        keep_only_nouns=cfg.processing.lemmatization.keep_only_nouns,
        cleaning_params=cfg.processing.cleaning,
        initial_test_size=cfg.splitting.initial_test_size,
        val_test_ratio=cfg.splitting.val_test_ratio,
        random_state=cfg.splitting.random_state
    )

    output_dir = to_absolute_path(cfg.paths.processed_data)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    train_df.to_csv(os.path.join(output_dir, cfg.processing.name, "train_df.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, cfg.processing.name, "val_df.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, cfg.processing.name, "test_df.csv"), index=False)

if __name__ == "__main__":
    main()