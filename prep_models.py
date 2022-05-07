import gensim
import gensim.downloader
import fasttext
import fasttext.util
import wget
import gzip
import shutil
import os
import argparse


def download_preproc_fasttext(emb_dim=300, vectors_dir="word_vectors"):
    if os.path.isfile("f{vectors_dir}/cc.en.300.bin"):
        print("\ncc.en.300.bin exists\n")
    else:
        print("Downloading 300-dim FastText model...\n")
        ft_model_url = (
            "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz"
        )
        wget.download(ft_model_url, out=f"{vectors_dir}/")
        print("\n")

        with gzip.open(f"{vectors_dir}/cc.en.300.bin.gz", "rb") as f_in:
            with open(f"{vectors_dir}/cc.en.300.bin", "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(f"{vectors_dir}/cc.en.300.bin.gz")

    if emb_dim == 200:
        ft = fasttext.load_model(f"{vectors_dir}/cc.en.300.bin")
        print("\nLoaded 300-dim FastText model")
        fasttext.util.reduce_model(ft, 200)
        ft.save_model(f"{vectors_dir}/cc.en.200.bin")
        print("Saved 200-dim FastText model\n")


def download_preproc_glove(vectors_dir="word_vectors"):
    if os.path.isfile(f"{vectors_dir}/glove-wiki-gigaword-200.bin"):
        print("\nglove-wiki-gigaword-200.bin exists\n")
    else:
        print("\nDownloading 200-dim GloVe model...\n")
        glove_vectors = gensim.downloader.load("glove-wiki-gigaword-200")
        glove_vectors.save_word2vec_format(
            f"{vectors_dir}/glove-wiki-gigaword-200.bin", binary=True
        )
        print("Saved 200-dim GloVe model\n")


def download_preproc_w2v(vectors_dir="word_vectors"):
    if os.path.isfile(f"{vectors_dir}/word2vec-google-news-300.bin"):
        print("word2vec-google-news-300.bin exists")
    else:
        print("\nDownloading 300-dim Word2Vec model...\n")
        glove_vectors = gensim.downloader.load("word2vec-google-news-300")
        glove_vectors.save_word2vec_format(
            f"{vectors_dir}/word2vec-google-news-300.bin", binary=True
        )
        print("Saved 300-dim Word2Vec model\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        type=str,
        default=["w2v", "glove", "fasttext"],
        nargs="*",
        help="Which model to download and preprocess, default: w2v, glove, fasttext (all of them)",
    )
    parser.add_argument(
        "--vectors_dir",
        type=str,
        default="word_vectors",
        nargs="?",
        help="Where to save the preprocessed models, default: word_vectors",
    )

    args = parser.parse_args()

    if "w2v" in args.models:
        download_preproc_w2v(vectors_dir=args.vectors_dir)

    if "glove" in args.models:
        download_preproc_glove(vectors_dir=args.vectors_dir)

    if "fasttext" in args.models:
        download_preproc_fasttext(emb_dim=200, vectors_dir=args.vectors_dir)

    if "fasttext_300" in args.models:
        download_preproc_fasttext(emb_dim=300, vectors_dir=args.vectors_dir)
