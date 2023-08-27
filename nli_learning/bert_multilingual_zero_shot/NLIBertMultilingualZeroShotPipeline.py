from nli_learning.Dataset.NLIDataModuleMultilanguageZeroShot import NLIDataModuleMultilanguageZeroShot
from nli_learning.bert_multilingual_zero_shot.bert_multilingual_zero_shot import train_bert_multilingual_zero_shot


NUM_EPOCHS = 10

def bert_multilingual_zero_shot_train_pipeline(dataset_path):


    print('Loading dataset')
    dataset = NLIDataModuleMultilanguageZeroShot(data_source_folder=dataset_path,dataset_name='bert_dataset', batch_size=256,transform=None)
    dataset.prepare_data()
    dataset.setup()

    print("MNLI Bert Zero Shot classifier")
    train_bert_multilingual_zero_shot(datamodule=dataset, num_epochs=NUM_EPOCHS)