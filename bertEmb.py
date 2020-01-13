"""BertEmbedding class."""
import collections
import re

import modeling
import tokenization
import tensorflow as tf

from extract_features import (InputExample, input_fn_builder, model_fn_builder,
                              convert_examples_to_features)


class BertModel(object):
    """BertEmbedding Class."""

    def __init__(self,
                 bert_config_file=None,
                 layers="-1",
                 max_seq_length=128,
                 init_checkpoint=None,
                 vocab_file=None,
                 do_lower_case=True,
                 batch_size=32,
                 use_tpu=False,
                 master=None,
                 num_tpu_cores=8,
                 use_one_hot_embeddings=False):
        """Init Method."""
        self.max_seq_length = max_seq_length

        self.tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file,
                                                    do_lower_case=do_lower_case)

        self.layer_indexes = [int(x) for x in layers.split(",")]

        bert_config = modeling.BertConfig.from_json_file(bert_config_file)

        # TPU Config
        is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
        tpu_config = tf.contrib.tpu.TPUConfig(num_shards=num_tpu_cores,
                                              per_host_input_for_training=is_per_host)
        run_config = tf.contrib.tpu.RunConfig(master=master, tpu_config=tpu_config)

        model_fn = model_fn_builder(bert_config=bert_config,
                                    init_checkpoint=init_checkpoint,
                                    layer_indexes=self.layer_indexes,
                                    use_tpu=use_tpu,
                                    use_one_hot_embeddings=use_one_hot_embeddings)

        # If TPU is not available, this will fall back to normal Estimator on CPU
        # or GPU.
        self.estimator = tf.contrib.tpu.TPUEstimator(use_tpu=use_tpu,
                                                     model_fn=model_fn,
                                                     config=run_config,
                                                     predict_batch_size=batch_size)

    def get_examples(self, sents):
        """
        Read sentences.

        Args:
            sents (str): A list of sentences

        Return:
            A list of InputExample
        """
        examples = []
        unique_id = 0
        for sent in sents:
            line = tokenization.convert_to_unicode(sent)
            line = line.strip()
            text_a = None
            text_b = None
            m = re.match(r"^(.*) \|\|\| (.*)$", line)
            if m is None:
                text_a = line
            else:
                text_a = m.group(1)
                text_b = m.group(2)
            examples.append(InputExample(unique_id=unique_id,
                                         text_a=text_a,
                                         text_b=text_b))
            unique_id += 1
        return examples

    def predict(self, sentences):
        """
        Predic Model.

        Args:
            sentences (str): A list of sentences

        Return:
            A dict of word embedding
        """
        examples = self.get_examples(sentences)

        features = convert_examples_to_features(examples=examples,
                                                seq_length=self.max_seq_length,
                                                tokenizer=self.tokenizer)

        unique_id_to_feature = {}
        for feature in features:
            unique_id_to_feature[feature.unique_id] = feature

        input_fn = input_fn_builder(features=features,
                                    seq_length=self.max_seq_length)

        predictions = []
        for result in self.estimator.predict(input_fn, yield_single_examples=True):
            unique_id = int(result["unique_id"])
            feature = unique_id_to_feature[unique_id]
            output_json = collections.OrderedDict()
            output_json["linex_index"] = unique_id
            all_features = []
            for (i, token) in enumerate(feature.tokens):
                all_layers = []
                for (j, layer_index) in enumerate(self.layer_indexes):
                    layer_output = result["layer_output_%d" % j]
                    layers = collections.OrderedDict()
                    layers["index"] = layer_index
                    layers["values"] = [
                        round(float(x), 6) for x in layer_output[i:(i + 1)].flat
                    ]
                    all_layers.append(layers)
                features = collections.OrderedDict()
                features["token"] = token
                features["layers"] = all_layers
                all_features.append(features)
            output_json["features"] = all_features
            predictions.append(output_json)

        # Post processing
        result = []
        for i in range(len(predictions)):
            prediction = [(elem["token"], elem["layers"][0]["values"])
                          for elem in predictions[i]["features"]]
            result.append(prediction)

        return result

    def post_processing(self, orig_tokens):
        """Get the mapping with the orign tokens."""
        bert_tokens = []

        # Token map will be an int -> int mapping between the `orig_tokens` index and
        # the `bert_tokens` index.
        orig_to_tok_map = []

        bert_tokens.append("[CLS]")
        for orig_token in orig_tokens:
            orig_to_tok_map.append(len(bert_tokens))
            bert_tokens.extend(self.tokenizer.tokenize(orig_token))
        bert_tokens.append("[SEP]")
        # bert_tokens == ["[CLS]", "john", "johan", "##son", "'", "s", "house", "[SEP]"]
        # orig_to_tok_map == [1, 2, 4, 6]
        return orig_to_tok_map
