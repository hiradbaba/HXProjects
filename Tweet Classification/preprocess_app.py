import data_process
'''converts raw data and stores transformed data'''
p = data_process.Preprocessor('data/tweet_data.csv')
p.nlp_data(save=True)
p.transform_data('d2v',save=True)
