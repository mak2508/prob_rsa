# Probabilistic RSA-presupposition
This is an implementation of RSA from [Presupposition Triggering Reflects Pragmatic Reasoning About Utterance Utility](https://wvvw.easychair.org/publications/preprint/WZwz8) using probabilitic programming to achieve greater scalability of the examples with respect to `world`, `context`, `utterance` and `qud` sizes.

The examples and the code are adapted/guided from the original codebase from the above linked paper, which can be [here](https://github.com/alexwarstadt/RSA-presupposition).

The `search_inference` library is taken from the [pyro](https://pyro.ai/) package. The original source can be found [here](https://github.com/pyro-ppl/pyro/blob/dev/examples/rsa/search_inference.py).


## Getting Started
### Environment Setup
Make sure you have python installed. The codebase is tested with `python 3.12`, but should work with earlier versions too.

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Examples
To make use of the library, a config file representing the presupposition to be studied must be created.
Some examples can be found in the `sample_configs` folder.
Explanations of different class methods are documented within the class.
```
rsa = RSA(<path_to_config_file>)
```
You can find example usage in notebooks in the `examples` folder. 
