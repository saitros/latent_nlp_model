import time
from nltk.corpus import stopwords as stop_words

from contextualized_topic_models.models.ctm import CombinedTM
from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation
from contextualized_topic_models.utils.preprocessing import WhiteSpacePreprocessingStopwords as sp

from task.preprocessing.data_load import total_data_load

def topic_(args):

    start_time = time.time()

    #===================================#
    #==============Logging==============#
    #===================================#

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    handler = TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter(" %(asctime)s - %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
    logger.propagate = False

    #===================================#
    #=============Data Load=============#
    #===================================#

    write_log(logger, 'Start preprocessing!')

    nltk.download('stopwords')
    stopwords = list(stop_words.words("english"))

    src_list, trg_list = total_data_load(args)

    #===================================#
    #==========Topic Modeling===========#
    #===================================#

    sp_train_out = sp(src_list['train'], stopwords_list=stopwords).preprocess()
    sp_valid_out = sp(src_list['valid'], stopwords_list=stopwords).preprocess()
    sp_test_out = sp(src_list['test'], stopwords_list=stopwords).preprocess()

    qt = TopicModelDataPreparation("all-mpnet-base-v2")
    train_dataset = qt.fit(text_for_contextual=sp_train_out[0], text_for_bow=sp_train_out[1])
    valid_dataset = qt.transform(text_for_contextual=sp_valid_out[0], text_for_bow=sp_valid_out[1])
    test_dataset = qt.transform(text_for_contextual=sp_test_out[0], text_for_bow=sp_test_out[1])

    ctm = CombinedTM(bow_size=len(qt.vocab), contextual_size=768, n_components=args.n_components, num_epochs=30) # 7 topics
    ctm.fit(train_dataset) # run the model
    ctm.save(models_dir=args.preprocess_path) # save the model

    topic_keywords_list = ctm.get_topic_lists(7)
    for i in range(args.n_components):
        write_log(logger, topic_keywords_list[i])