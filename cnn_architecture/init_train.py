import logging

from cnn_rnn_architecture import train_with_wv as cnn

LOG_FILENAME = 'train.log'
logging.basicConfig(level=logging.INFO)

# Data path
train_data_path = "../data/train_data.csv"
test_data_path = "../data/test_data.csv"

logging.info("\n\n\n==========================================================\n")

# cnn.initiate_training(train_data_path, test_data_path, sequence_length=100, num_epochs=20, embedding_dim=50, context=25)

cnn.initiate_training(train_data_path, test_data_path, sequence_length=100, num_epochs=10, embedding_dim=100,
                      context=20)

# cnn.initiate_training(train_data_path, test_data_path, sequence_length=100, num_epochs=20, embedding_dim=100, context=25)

cnn.initiate_training(train_data_path, test_data_path,
                      sequence_length=100,
                      embedding_dim=50,
                      filter_sizes=[2, 3, 4, 5, 6, 7, 8],
                      num_filters=100,
                      dropout_prob=[0.6, 0.7],
                      hidden_dims=400,
                      batch_size=50,
                      num_epochs=15,
                      min_word_count=1,
                      context=20)

logging.info("\n\n\n==========================================================\n")

logging.info("\n\n\n==========================================================\n")
cnn.initiate_training(train_data_path, test_data_path,
                      sequence_length=100,
                      embedding_dim=100,
                      filter_sizes=[3, 5],
                      num_filters=300,
                      dropout_prob=[0.5, 0.7],
                      hidden_dims=100,
                      batch_size=50,
                      num_epochs=15,
                      min_word_count=1,
                      context=20)

logging.info("\n\n\n==========================================================\n")
cnn.initiate_training(train_data_path, test_data_path,
                      sequence_length=100,
                      embedding_dim=100,
                      filter_sizes=[5, 8],
                      num_filters=600,
                      dropout_prob=[0.5, 0.7],
                      hidden_dims=200,
                      batch_size=50,
                      num_epochs=15,
                      min_word_count=1,
                      context=20)

logging.info("\n\n\n==========================================================\n")

cnn.initiate_training(train_data_path, test_data_path,
                      sequence_length=100,
                      embedding_dim=100,
                      filter_sizes=[3, 5, 8],
                      num_filters=600,
                      dropout_prob=[0.5, 0.7],
                      hidden_dims=200,
                      batch_size=50,
                      num_epochs=15,
                      min_word_count=1,
                      context=20)

logging.info("\n\n\n==========================================================\n")

cnn.initiate_training(train_data_path, test_data_path,
                      sequence_length=100,
                      embedding_dim=100,
                      filter_sizes=[3, 8],
                      num_filters=600,
                      dropout_prob=[0.5, 0.7],
                      hidden_dims=200,
                      batch_size=50,
                      num_epochs=15,
                      min_word_count=1,
                      context=20)

logging.info("\n\n\n==========================================================\n")

cnn.initiate_training(train_data_path, test_data_path,
                      sequence_length=100,
                      embedding_dim=100,
                      filter_sizes=[3, 5],
                      num_filters=600,
                      dropout_prob=[0.5, 0.7],
                      hidden_dims=200,
                      batch_size=50,
                      num_epochs=15,
                      min_word_count=1,
                      context=20)

logging.info("\n\n\n==========================================================\n")

cnn.initiate_training(train_data_path, test_data_path,
                      sequence_length=100,
                      embedding_dim=100,
                      filter_sizes=[2, 4],
                      num_filters=300,
                      dropout_prob=[0.5, 0.7],
                      hidden_dims=200,
                      batch_size=50,
                      num_epochs=15,
                      min_word_count=1,
                      context=20)

logging.info("\n\n\n==========================================================\n")
cnn.initiate_training(train_data_path, test_data_path,
                      sequence_length=100,
                      embedding_dim=50,
                      filter_sizes=[3, 5],
                      num_filters=300,
                      dropout_prob=[0.5, 0.7],
                      hidden_dims=100,
                      batch_size=50,
                      num_epochs=15,
                      min_word_count=1,
                      context=20)

logging.info("\n\n\n==========================================================\n")
cnn.initiate_training(train_data_path, test_data_path,
                      sequence_length=100,
                      embedding_dim=50,
                      filter_sizes=[5, 8],
                      num_filters=600,
                      dropout_prob=[0.5, 0.7],
                      hidden_dims=200,
                      batch_size=50,
                      num_epochs=15,
                      min_word_count=1,
                      context=20)

logging.info("\n\n\n==========================================================\n")

cnn.initiate_training(train_data_path, test_data_path,
                      sequence_length=100,
                      embedding_dim=50,
                      filter_sizes=[3, 5, 8],
                      num_filters=600,
                      dropout_prob=[0.5, 0.7],
                      hidden_dims=200,
                      batch_size=50,
                      num_epochs=15,
                      min_word_count=1,
                      context=20)

logging.info("\n\n\n==========================================================\n")

cnn.initiate_training(train_data_path, test_data_path,
                      sequence_length=100,
                      embedding_dim=50,
                      filter_sizes=[3, 8],
                      num_filters=600,
                      dropout_prob=[0.5, 0.7],
                      hidden_dims=200,
                      batch_size=50,
                      num_epochs=15,
                      min_word_count=1,
                      context=20)

logging.info("\n\n\n==========================================================\n")

cnn.initiate_training(train_data_path, test_data_path,
                      sequence_length=100,
                      embedding_dim=50,
                      filter_sizes=[3, 5],
                      num_filters=600,
                      dropout_prob=[0.5, 0.7],
                      hidden_dims=200,
                      batch_size=50,
                      num_epochs=15,
                      min_word_count=1,
                      context=20)

logging.info("\n\n\n==========================================================\n")

cnn.initiate_training(train_data_path, test_data_path,
                      sequence_length=100,
                      embedding_dim=50,
                      filter_sizes=[2, 4],
                      num_filters=300,
                      dropout_prob=[0.5, 0.7],
                      hidden_dims=200,
                      batch_size=50,
                      num_epochs=15,
                      min_word_count=1,
                      context=20)

logging.info("\n\n\n==========================================================\n")
cnn.initiate_training(train_data_path, test_data_path,
                      sequence_length=100,
                      embedding_dim=50,
                      filter_sizes=[3, 5],
                      num_filters=300,
                      dropout_prob=[0.6, 0.8],
                      hidden_dims=200,
                      batch_size=50,
                      num_epochs=15,
                      min_word_count=1,
                      context=20)

logging.info("\n\n\n==========================================================\n")
cnn.initiate_training(train_data_path, test_data_path,
                      sequence_length=100,
                      embedding_dim=50,
                      filter_sizes=[4, 8],
                      num_filters=600,
                      dropout_prob=[0.5, 0.7],
                      hidden_dims=200,
                      batch_size=50,
                      num_epochs=15,
                      min_word_count=1,
                      context=20)

logging.info("\n\n\n==========================================================\n")
