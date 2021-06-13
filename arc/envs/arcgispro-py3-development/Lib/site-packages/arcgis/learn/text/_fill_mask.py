import traceback
from .._data import _raise_fastai_import_error
HAS_TRANSFORMER = True

try:
    import torch
    from transformers import pipeline, logging
    from .._utils.common import _get_device_id
    from fastprogress.fastprogress import progress_bar
    from transformers.modeling_auto import MODEL_FOR_MASKED_LM_MAPPING
    EXPECTED_MODEL_TYPES = [x.__name__.replace('Config', '') for x in MODEL_FOR_MASKED_LM_MAPPING.keys()]
except Exception as e:
    transformer_exception = "\n".join(traceback.format_exception(type(e), e, e.__traceback__))
    HAS_TRANSFORMER = False
    EXPECTED_MODEL_TYPES = []

class FillMask:
    """
    Creates a `FillMask` Object.
    Based on the Hugging Face transformers library

    =====================   ===========================================
    **Argument**            **Description**
    ---------------------   -------------------------------------------
    backbone                Optional string. Specify the HuggingFace
                            transformer model name which will be used to
                            generate the suggestion token.

                            To learn more about the available models for
                            fill-mask task, kindly visit:-
                            https://huggingface.co/models?filter=lm-head
    =====================   ===========================================

    :returns: `FillMask` Object
    """

    #: supported transformer backbones
    supported_backbones = EXPECTED_MODEL_TYPES

    def __init__(self, backbone=None):
        if not HAS_TRANSFORMER:
            _raise_fastai_import_error(import_exception=transformer_exception)

        logger = logging.get_logger()
        logger.setLevel(logging.ERROR)
        self._device = _get_device_id()
        self._task = "fill-mask"
        try:
            self.model = pipeline(self._task, model=backbone, device=self._device, topk=10)
        except Exception as e:
            error_message = (f"Model - `{backbone}` cannot be used for {self._task} task.\n"
                             f"Model type should be one of {EXPECTED_MODEL_TYPES}.")
            raise Exception(error_message)

    def predict_token(self, text_or_list, num_suggestions=5):
        """
        Summarize the given text or list of text

        =====================   ===========================================
        **Argument**            **Description**
        ---------------------   -------------------------------------------
        text_or_list            Required string or list. A text/sentence
                                or a list of texts/sentences for which on wishes
                                to generate the recommendations for masked-token.
        =====================   ===========================================

        :returns: A list or a list of list of :obj:`dict`: Each result comes as list of dictionaries with the following keys:

            - **sequence** (:obj:`str`)  -- The corresponding input with the mask token prediction.
            - **score** (:obj:`float`)   -- The corresponding probability.
            - **token_str** (:obj:`str`) -- The predicted token (to replace the masked one).
        """
        results = []
        if not isinstance(text_or_list, (list, tuple)): text_or_list = [text_or_list]
        self._do_sanity(text_or_list)
        for i in progress_bar(range(len(text_or_list))):
            text = text_or_list[i].replace('__', self.model.tokenizer.mask_token)
            result = self._process_result(self.model(text)[:num_suggestions])
            if num_suggestions == 1: result = result[0]
            results.append(result)
        return results

    @staticmethod
    def _do_sanity(text_list):
        for text in text_list:
            if '__' not in text:
                error_message = (
                    f"Text - `{text}` is mising `__` token. Use the `__` token to "
                    "specify where you want to generate the suggestions in the text."
                )
                raise Exception(error_message)

    def _process_result(self, result_list):
        for item in result_list:
            _ = item.pop("token", -1)
            token = item["token_str"]
            sequence = item["sequence"]
            item["token_str"] = self.model.tokenizer.convert_tokens_to_string(token).strip()
            item["sequence"] = self.model.tokenizer.decode(
                self.model.tokenizer.encode(sequence, add_special_tokens=False), skip_special_tokens=True)

        return result_list
