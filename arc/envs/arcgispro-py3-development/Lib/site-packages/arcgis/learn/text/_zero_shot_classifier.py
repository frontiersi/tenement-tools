import traceback
from .._data import _raise_fastai_import_error
HAS_TRANSFORMER = True

try:
    import torch
    from transformers import pipeline, logging
    from .._utils.common import _get_device_id
    from fastprogress.fastprogress import progress_bar
    from transformers.modeling_auto import MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING
    EXPECTED_MODEL_TYPES = [x.__name__.replace('Config', '') for x in MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING.keys()]
except Exception as e:
    transformer_exception = "\n".join(traceback.format_exception(type(e), e, e.__traceback__))
    HAS_TRANSFORMER = False
    EXPECTED_MODEL_TYPES = []


class ZeroShotClassifier:
    """
    Creates a `ZeroShotClassifier` Object.
    Based on the Hugging Face transformers library

    =====================   ===========================================
    **Argument**            **Description**
    ---------------------   -------------------------------------------
    backbone                Optional string. Specifying the HuggingFace
                            transformer model name which will be used to
                            predict the answers from a given passage/context.

                            To learn more about the available models for
                            zero-shot-classification task, kindly visit:-
                            https://huggingface.co/models?search=nli
    =====================   ===========================================

    :returns: `ZeroShotClassifier` Object
    """

    #: supported transformer backbones
    supported_backbones = EXPECTED_MODEL_TYPES

    def __init__(self, backbone=None):
        if not HAS_TRANSFORMER:
            _raise_fastai_import_error(import_exception=transformer_exception)

        logger = logging.get_logger()
        logger.setLevel(logging.ERROR)
        self._device = _get_device_id()
        self._task = "zero-shot-classification"
        try:
            self.model = pipeline(self._task, model=backbone, device=self._device)
        except Exception as e:
            error_message = (f"Model - `{backbone}` cannot be used for {self._task} task.\n"
                             f"Model type should be one of {EXPECTED_MODEL_TYPES}.")
            raise Exception(error_message)
        
    def predict(self, text_or_list, candidate_labels, **kwargs):
        """
        Predicts the class label(s) for the input text

        =====================   ===========================================
        **Argument**            **Description**
        ---------------------   -------------------------------------------
        text_or_list            Required string or list. The sequence or a
                                list of sequences to classify.
        ---------------------   -------------------------------------------
        candidate_labels        Required string or list. The set of possible
                                class labels to classify each sequence into.
                                Can be a single label, a string of
                                comma-separated labels, or a list of labels.
        =====================   ===========================================

        **kwargs**

        =====================   ===========================================
        **Argument**            **Description**
        ---------------------   -------------------------------------------
        multi_class             Optional boolean. Whether or not multiple
                                candidate labels can be true.
                                Default value is set to False.
        ---------------------   -------------------------------------------
        hypothesis              Optional string. The template used to turn each
                                label into an NLI-style hypothesis. This template
                                must include a {} or similar syntax for the
                                candidate label to be inserted into the template.
                                Default value is set to `"This example is {}."`.
        =====================   ===========================================

        :returns: a list of :obj:`dict`: Each result comes as a dictionary with the following keys:

            - **sequence** (:obj:`str`) -- The sequence for which this is the output.
            - **labels** (:obj:`List[str]`) -- The labels sorted by order of likelihood.
            - **scores** (:obj:`List[float]`) -- The probabilities for each of the labels.
        """
        results = []
        multi_class = kwargs.get("multi_class", False)
        hypothesis = kwargs.get("hypothesis", "This example is {}.")
        if not isinstance(text_or_list, (list, tuple)): text_or_list = [text_or_list]
        for i in progress_bar(range(len(text_or_list))):
            result = self.model(text_or_list[i], candidate_labels,
                                multi_class=multi_class, hypothesis_template=hypothesis)
            results.append(result)
        from IPython.display import clear_output
        clear_output()
        return results
