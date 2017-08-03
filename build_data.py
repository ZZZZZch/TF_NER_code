from config import Config
from data_utils import Dataset, get_vocabs, UNK, NUM, \
    get_glove_vocab, write_vocab, load_vocab, get_char_vocab, \
    export_trimmed_glove_vectors, get_processing_word, write_clear_data


def build_data(config):
    """
    Procedure to build data

    Args:
        config: defines attributes needed in the function
    Returns:
        creates vocab files from the datasets
        creates a npz embedding file from trimmed glove vectors
    """
    processing_word = get_processing_word(lowercase=True)
    processing_word = get_processing_word(lowercase=True)

    # clean data
    train_filepath, dev_filepath_a = write_clear_data(
        config.train_filename,
        build_dev=config.build_dev_from_trainset,
        dev_ratio=config.dev_ratio)
    test_filepath, dev_filepath_b = write_clear_data(
        config.test_filename,
        build_dev=config.build_dev_from_testset,
        dev_ratio=config.dev_ratio)
    dev_filepath = dev_filepath_a or dev_filepath_b

    # Generators
    dev = Dataset(dev_filepath, processing_word)
    test = Dataset(test_filepath, processing_word)
    train = Dataset(train_filepath, processing_word)

    # Build Word and Tag vocab
    vocab_words, vocab_tags = get_vocabs([train, dev, test])
    vocab_glove = get_glove_vocab(config.glove_filename)

    vocab = vocab_words & vocab_glove
    vocab.add(UNK)
    vocab.add(NUM)

    # Save vocab
    write_vocab(vocab, config.words_filename)
    write_vocab(vocab_tags, config.tags_filename)

    # Trim GloVe Vectors
    vocab = load_vocab(config.words_filename)
    export_trimmed_glove_vectors(vocab, config.glove_filename,
                                 config.trimmed_filename, config.dim)

    # Build and save char vocab
    train = Dataset(config.train_filename)
    vocab_chars = get_char_vocab(train)
    write_vocab(vocab_chars, config.chars_filename)


if __name__ == "__main__":
    config = Config()
    build_data(config)
