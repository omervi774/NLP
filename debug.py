import loader as ld

train_dataset, test_dataset, num_words, input_size = ld.get_data_set(32, toy=True)
toy_dataset = ld.get_toy_data()

for labels, reviews, reviews_text in toy_dataset:
    print(reviews_text)
