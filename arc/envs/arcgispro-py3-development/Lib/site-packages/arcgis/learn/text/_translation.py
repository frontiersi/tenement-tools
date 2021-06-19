import traceback
from .._data import _raise_fastai_import_error
HAS_TRANSFORMER = True

try:
    import torch
    from transformers import pipeline, logging
    from .._utils.common import _get_device_id
    from fastprogress.fastprogress import progress_bar
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
except Exception as e:
    transformer_exception = "\n".join(traceback.format_exception(type(e), e, e.__traceback__))
    HAS_TRANSFORMER = False


class TextTranslator:
    """
    Creates a `TextTranslator` Object.
    Based on the Hugging Face transformers library
    To learn more about the available models for translation task,
    kindly visit:- https://huggingface.co/models?filter=translation
    
    =====================   ===========================================
    **Argument**            **Description**
    ---------------------   -------------------------------------------
    source_language         Optional string. Specify the language of the
                            text you would like to get the translation of.
                            Default value is 'es' (Spanish)
    ---------------------   -------------------------------------------
    target_language         Optional string. The language into which one
                            wishes to translate the input text.
                            Default value is 'en' (English)
    =====================   ===========================================

    :returns: `TextTranslator` Object
    """

    #: supported transformer backbones
    supported_backbones = ["MarianMT"]

    def __init__(self, source_language="es", target_language="en"):
        if not HAS_TRANSFORMER:
            _raise_fastai_import_error(import_exception=transformer_exception)

        logger = logging.get_logger()
        logger.setLevel(logging.ERROR)
        self._source_lang = source_language
        self._target_lang = target_language

        self._device_id = _get_device_id()
        self._device = torch.device("cpu" if self._device_id < 0 else "cuda:{}".format(self._device_id))
        self._task = f"translation_{self._source_lang}_to_{self._target_lang}"
        self._tokenizer = AutoTokenizer.from_pretrained(f"Helsinki-NLP/opus-mt-{source_language}-{target_language}")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(f"Helsinki-NLP/opus-mt-{source_language}-{target_language}")
        self.model.to(self._device)

    def translate(self, text_or_list, **kwargs):
        """
        Translate the given text or list of text into the target language

        =====================   ===========================================
        **Argument**            **Description**
        ---------------------   -------------------------------------------
        text_or_list            Required string or list. A text/passage
                                or a list of texts/passages to translate.
        =====================   ===========================================

        **kwargs**

        =====================   ===========================================
        **Argument**            **Description**
        ---------------------   -------------------------------------------
        min_length              Optional integer. The minimum length of the
                                sequence to be generated.
                                Default value is set to to `min_length` parameter
                                of the model config.
        ---------------------   -------------------------------------------
        max_length              Optional integer. The maximum length of the
                                sequence to be generated.
                                Default value is set to to `max_length` parameter
                                of the model config.
        ---------------------   -------------------------------------------
        num_return_sequences    Optional integer. The number of independently
                                computed returned sequences for each element
                                in the batch.
                                Default value is set to 1.
        ---------------------   -------------------------------------------
        num_beams               Optional integer. Number of beams for beam
                                search. 1 means no beam search.
                                Default value is set to 1.
        ---------------------   -------------------------------------------
        length_penalty          Optional float. Exponential penalty to the
                                length. 1.0 means no penalty. Set to values < 1.0
                                in order to encourage the model to generate
                                shorter sequences, to a value > 1.0 in order to
                                encourage the model to produce longer sequences.
                                Default value is set to 1.0.
        ---------------------   -------------------------------------------
        early_stopping          Optional bool. Whether to stop the beam search
                                when at least ``num_beams`` sentences are
                                finished per batch or not.
                                Default value is set to False.
        =====================   ===========================================

        :returns: a list or a list of list containing the translation of the input prompt(s) / sentence(s) to the target language
        """
        results = []
        num_return_sequences = kwargs.get("num_return_sequences", 1)
        if not isinstance(text_or_list, (list, tuple)): text_or_list = [text_or_list]
        for i in progress_bar(range(len(text_or_list))):
            inputs = self._tokenizer.encode(f"{text_or_list[i]} {self._tokenizer.eos_token}", return_tensors="pt").\
                to(self._device)
            outputs = self.model.generate(inputs, **kwargs)
            result = self._tokenizer.batch_decode(outputs, skip_special_tokens=True)
            result = [{"translated_text": x} for x in result]
            if num_return_sequences == 1: result = result[0]
            results.append(result)
        return results
