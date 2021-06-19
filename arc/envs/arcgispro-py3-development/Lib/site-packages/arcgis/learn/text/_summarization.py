import traceback
from .._data import _raise_fastai_import_error
HAS_TRANSFORMER = True

try:
    import torch
    from transformers import pipeline, logging
    from .._utils.common import _get_device_id
    from fastprogress.fastprogress import progress_bar
    from transformers.modeling_auto import MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING
    EXPECTED_MODEL_TYPES = [x.__name__.replace('Config', '') for x in MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING.keys()]
except Exception as e:
    transformer_exception = "\n".join(traceback.format_exception(type(e), e, e.__traceback__))
    HAS_TRANSFORMER = False
    EXPECTED_MODEL_TYPES = []

class TextSummarizer:
    """
    Creates a `TextSummarizer` Object.
    Based on the Hugging Face transformers library

    =====================   ===========================================
    **Argument**            **Description**
    ---------------------   -------------------------------------------
    backbone                Optional string. Specify the HuggingFace
                            transformer model name which will be used to
                            summarize the text.

                            To learn more about the available models for
                            summarization task, kindly visit:-
                            https://huggingface.co/models?filter=summarization
    =====================   ===========================================

    :returns: `TextSummarizer` Object
    """

    #: supported transformer backbones
    supported_backbones = EXPECTED_MODEL_TYPES

    def __init__(self, backbone=None):
        if not HAS_TRANSFORMER:
            _raise_fastai_import_error(import_exception=transformer_exception)

        logger = logging.get_logger()
        logger.setLevel(logging.ERROR)
        self._device = _get_device_id()
        self._task = "summarization"
        try:
            self.model = pipeline(self._task, model=backbone, device=self._device)
        except Exception as e:
            error_message = (f"Model - `{backbone}` cannot be used for {self._task} task.\n"
                             f"Model type should be one of {EXPECTED_MODEL_TYPES}.")
            raise Exception(error_message)

    def summarize(self, text_or_list, **kwargs):
        """
        Summarize the given text or list of text

        =====================   ===========================================
        **Argument**            **Description**
        ---------------------   -------------------------------------------
        text_or_list            Required string or list. A text/passage
                                or a list of texts/passages to generate the
                                summary for.
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

        :returns: a list or a list of list containing the summary/summaries for the input prompt(s) / sentence(s)
        """
        results = []
        num_return_sequences = kwargs.get("num_return_sequences", 1)
        if not isinstance(text_or_list, (list, tuple)): text_or_list = [text_or_list]
        for i in progress_bar(range(len(text_or_list))):
            result = self.model(text_or_list[i], **kwargs)
            if num_return_sequences == 1: result = result[0]
            results.append(result)
        return results
