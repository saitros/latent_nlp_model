# IIPL Latent-variable machine translation project
This project is an NLP project conducted by IIPL. This project, which intends to apply latent-variables to various fields of NLP, aims to improve the performance of various NLP tasks such as low-resource machine translation, text style transfer, and dataset shift.

## Dependencies

This code is written in Python. Dependencies include

* python == 3.6
* PyTorch == 1.6
* Transformers == 4.8.1

## Preprocessing

A step by step series of examples that tell you how to get a development env running

Say what the step will be

```
python main.py --preprocessing
```

Available options are tokenizer(--tokenizer), SentencePiece model type(--sentencepiece_model; If tokenizer is spm),  source text vocabulary size(--src_vocab_size), target text vocabulary size(--trg_vocab_size), padding token id(--pad_id), unknown token id(--unk_id), start token id(--bos_id) and end token id(--eos_id).

```
python main.py --preprocessing --tokenizer=spm --sentencepiece_model=unigram --src_vocab_size=8000 --trg_vocab_size=8000 --pad_id=0 --unk_id=3 --bos_id=1 --eos_id=2
```

## Training

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc

