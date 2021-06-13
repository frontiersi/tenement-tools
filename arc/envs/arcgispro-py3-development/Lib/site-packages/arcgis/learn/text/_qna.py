import traceback
import warnings
warnings.filterwarnings("ignore", module='transformers')
from .._data import _raise_fastai_import_error
HAS_TRANSFORMER = True

try:
    import torch
    from transformers import pipeline, logging
    from .._utils.common import _get_device_id
    from fastprogress.fastprogress import progress_bar
    from transformers.modeling_auto import MODEL_FOR_QUESTION_ANSWERING_MAPPING
    EXPECTED_MODEL_TYPES = [x.__name__.replace('Config', '') for x in MODEL_FOR_QUESTION_ANSWERING_MAPPING.keys()]
except Exception as e:
    transformer_exception = "\n".join(traceback.format_exception(type(e), e, e.__traceback__))
    HAS_TRANSFORMER = False
    EXPECTED_MODEL_TYPES = []


class QuestionAnswering:
    """
    Creates a `QuestionAnswering` Object.
    Based on the Hugging Face transformers library

    =====================   ===========================================
    **Argument**            **Description**
    ---------------------   -------------------------------------------
    backbone                Optional string. Specify the HuggingFace
                            transformer model name which will be used to
                            extract the answers from a given passage/context.

                            To learn more about the available models for
                            question-answering task, kindly visit:-
                            https://huggingface.co/models?filter=question-answering
    =====================   ===========================================

    :returns: `QuestionAnswering` Object
    """

    #: supported transformer backbones
    supported_backbones = EXPECTED_MODEL_TYPES

    def __init__(self, backbone=None):
        if not HAS_TRANSFORMER:
            _raise_fastai_import_error(import_exception=transformer_exception)

        logger = logging.get_logger()
        logger.setLevel(logging.ERROR)
        self._device = _get_device_id()
        self._task = "question-answering"
        try:
            self.model = pipeline(self._task, model=backbone, device=self._device)
        except Exception as e:
            error_message = (f"Model - `{backbone}` cannot be used for {self._task} task.\n"
                             f"Model type should be one of {EXPECTED_MODEL_TYPES}.")
            raise Exception(error_message)

    def get_answer(self, text_or_list, context, **kwargs):
        """
        Find answers for the asked questions from the given passage/context

        =====================   ===========================================
        **Argument**            **Description**
        ---------------------   -------------------------------------------
        text_or_list            Required string or list. Questions or a list
                                of questions one wishes to seek an answer for.
        ---------------------   -------------------------------------------
        context                 Required string. The context associated with
                                the question(s) which contains the answers.
        =====================   ===========================================

        **kwargs**

        =====================   ===========================================
        **Argument**            **Description**
        ---------------------   -------------------------------------------
        num_answers             Optional integer. The number of answers to
                                return. The answers will be chosen by order
                                of likelihood.
                                Default value is set to 1.
        ---------------------   -------------------------------------------
        max_answer_length       Optional integer. The maximum length of the
                                predicted answers.
                                Default value is set to 15.
        ---------------------   -------------------------------------------
        max_question_length     Optional integer. The maximum length of the
                                question after tokenization. Questions will be
                                truncated if needed.
                                Default value is set to 64.
        ---------------------   -------------------------------------------
        impossible_answer       Optional bool. Whether or not we accept impossible
                                as an answer.
                                Default value is set to False
        =====================   ===========================================

        :returns: a list or a list of list containing the answer(s) for the input question(s)
        """
        results, kwargs_dict = [], {}

        kwargs_dict["topk"] = kwargs.get("num_answers", 1)
        kwargs_dict["max_answer_len"] = kwargs.get("max_answer_length", 15)
        kwargs_dict["max_question_len"] = kwargs.get("max_question_length", 64)
        kwargs_dict["handle_impossible_answer"] = kwargs.get("impossible_answer", False)

        if not isinstance(text_or_list, (list, tuple)): text_or_list = [text_or_list]
        for i in progress_bar(range(len(text_or_list))):
            results.append(self.model(question=text_or_list[i], context=context, **kwargs_dict))

        return self._process_result(results, text_or_list)

    @staticmethod
    def _process_result(result_list, question_list):
        processed_results = []
        for result, question in zip(result_list, question_list):
            if isinstance(result, dict):
                tmp_dict = {"question": question, "answer": result["answer"], "score": result["score"]}
                processed_results.append(tmp_dict)
            elif isinstance(result, list):
                item_list = [{"question": question, "answer": item["answer"], "score": item["score"]} for item in result]
                processed_results.append(item_list)

        return processed_results
