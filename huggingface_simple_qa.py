# stolen from https://huggingface.co/transformers/task_summary.html#question-answering
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

from httprest_client import HttpTritonClient


def get_sample_context_questions():
    text = r"""
    🤗 Transformers (formerly known as pytorch-transformers and pytorch-pretrained-bert) provides general-purpose
    architectures (BERT, GPT-2, RoBERTa, XLM, DistilBert, XLNet…) for Natural Language Understanding (NLU) and Natural
    Language Generation (NLG) with over 32+ pretrained models in 100+ languages and deep interoperability between
    TensorFlow 2.0 and PyTorch.
    """
    questions = [
        "How many pretrained models are available in 🤗 Transformers?",
        "What does 🤗 Transformers provide?",
        "🤗 Transformers provides interoperability between which frameworks?",
    ]
    return text, questions


def get_start_end(outputs):
    answer_start_scores = outputs.start_logits
    answer_end_scores = outputs.end_logits
    # Get the most likely beginning of answer with the argmax of the score
    answer_start = torch.argmax(answer_start_scores)
    # Get the most likely end of answer with the argmax of the score
    answer_end = torch.argmax(answer_end_scores) + 1
    return answer_start, answer_end


if __name__ == "__main__":

    model_name = "deepset/bert-base-cased-squad2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    text, questions = get_sample_context_questions()

    model_name = "deepset_bert_base_cased_squad2"
    with HttpTritonClient(model_name) as client:

        for question in questions:
            inputs = tokenizer(question, text, add_special_tokens=True, return_tensors="pt")
            input_ids = inputs["input_ids"].tolist()[0]
            # text_tokens = tokenizer.convert_ids_to_tokens(input_ids)
            inputs_numpy = {k:x.numpy() for k,x in inputs.items()}
            outputs = client(inputs_numpy)
            answer_start,answer_end = get_start_end(outputs)
            answer = tokenizer.convert_tokens_to_string(
                tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end])
            )
            print(f"Question: {question}")
            print(f"Answer: {answer}")
