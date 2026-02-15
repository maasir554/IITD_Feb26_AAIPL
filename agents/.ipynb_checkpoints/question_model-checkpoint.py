# Starting with Unsloth (load trained LoRA adapter)
import time
from typing import Optional, Union, List
from unsloth import FastLanguageModel


class QAgent(object):
    def __init__(self, **kwargs):
        model_path = kwargs.get("model_path", "/workspace/AAIPL/outputs/final")
        # Allow long input; output capped at 1024. Total context = max_seq_length.
        self.max_seq_length = kwargs.get("max_seq_length", 2048)

        # load with Unsloth for inference (trained adapter)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=self.max_seq_length,
        )
        FastLanguageModel.for_inference(model)

        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "left"
        self.model = model
        print("Loaded successfully ✅")

    def generate_response(
        self, message: str | List[str], system_prompt: Optional[str] = None, **kwargs
    ) -> str:
        # Decoder-only models require left padding for correct batch generation
        self.tokenizer.padding_side = "left"
        if system_prompt is None:
            system_prompt = "You are a helpful assistant."
        if isinstance(message, str):
            message = [message]
        # Prepare all messages for batch processing
        all_messages = []
        for msg in message:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": msg},
            ]
            all_messages.append(messages)

        # convert all messages to text format
        texts = []
        for messages in all_messages:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            texts.append(text)

        # tokenize: truncate so input + max_new_tokens <= max_seq_length (output cap 1024)
        max_new_tokens = kwargs.get("max_new_tokens", 1024)
        max_input_length = self.max_seq_length - max_new_tokens
        model_inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=max_input_length,
        ).to(self.model.device)

        tgps_show_var = kwargs.get("tgps_show", False)
        # conduct batch text completion
        if tgps_show_var:
            start_time = time.time()
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        if tgps_show_var:
            generation_time = time.time() - start_time

        # decode the batch
        batch_outs = []
        if tgps_show_var:
            token_len = 0
        for i, (input_ids, generated_sequence) in enumerate(
            zip(model_inputs.input_ids, generated_ids)
        ):
            # extract only the newly generated tokens
            output_ids = generated_sequence[len(input_ids) :].tolist()

            # compute total tokens generated
            if tgps_show_var:
                token_len += len(output_ids)

            # remove thinking content using regex
            # result = re.sub(r'<think>[\s\S]*?</think>', '', full_result, flags=re.DOTALL).strip()
            index = (
                len(output_ids) - output_ids[::-1].index(151668)
                if 151668 in output_ids
                else 0
            )

            # decode the full result
            content = self.tokenizer.decode(
                output_ids[index:], skip_special_tokens=True
            ).strip("\n")
            batch_outs.append(content)
        if tgps_show_var:
            return (
                batch_outs[0] if len(batch_outs) == 1 else batch_outs,
                token_len,
                generation_time,
            )
        return batch_outs[0] if len(batch_outs) == 1 else batch_outs, None, None


if __name__ == "__main__":
    # Single example generation
    model = QAgent()
    prompt = f"""
    Question: Generate a hard MCQ based question as well as their 4 choices and its answers on the topic, Number Series.
    Return your response as a valid JSON object with this exact structure:

        {{
            "topic": Your Topic,
            "question": "Your question here ending with a question mark?",
            "choices": [
                "A) First option",
                "B) Second option", 
                "C) Third option",
                "D) Fourth option"
            ],
            "answer": "A",
            "explanation": "Brief explanation of why the correct answer is right and why distractors are wrong"
        }}
    """

    response, tl, tm = model.generate_response(
        prompt,
        tgps_show=True,
        max_new_tokens=512,
        temperature=0.1,
        top_p=0.9,
        do_sample=True,
    )
    print("Single example response:")
    print("Response: ", response)
    print(
        f"Total tokens: {tl}, Time taken: {tm:.2f} seconds, TGPS: {tl/tm:.2f} tokens/sec"
    )
    print("+-------------------------------------------------\n\n")

    # Multi example generation
    prompts = [
        "What is the capital of France?",
        "Explain the theory of relativity.",
        "What are the main differences between Python and Java?",
        "What is the significance of the Turing Test in AI?",
        "What is the capital of Japan?",
    ]
    responses, tl, tm = model.generate_response(
        prompts,
        tgps_show=True,
        max_new_tokens=512,
        temperature=0.1,
        top_p=0.9,
        do_sample=True,
    )
    print("\nMulti example responses:")
    for i, resp in enumerate(responses):
        print(f"Response {i+1}: {resp}")
    print(
        f"Total tokens: {tl}, Time taken: {tm:.2f} seconds, TGPS: {tl/tm:.2f} tokens/sec"
    )
