from vlmeval.smp import *
from vlmeval.api.base import BaseAPI
from time import sleep
import base64
import mimetypes
from PIL import Image
from typing import Optional
import os
import requests
# official_url = "https://stg.api.cohere.ai/v2/chat"
# STG_API_KEY = "izzl1o8oWv7PDS0rdTf4KRm6MTbpLtbgLf1J8H7M"

COHERE_COT_MMMU_DEFAULT = """Analyze the image and question carefully, using step-by-step reasoning.
First, describe any image provided in detail. Then, present your reasoning. Last, and most important, your answer should always end in this format:
Final Answer: <answer>
where <answer> is:
- The single correct letter choice A, B, C, D, E, F, etc. when options are provided. Only include the letter.
- Your direct answer if no options are given, as a single phrase or number.
- If your answer is a number, only include the number without any unit.
- If your answer is a word or phrase, do not paraphrase or reformat the text you see in the image.
- You cannot answer that the question is unanswerable. You must either pick an option or provide a direct answer.
IMPORTANT: Remember, to end your answer with Final Answer: <answer>."""

INTERN_VL_COT = "Answer the preceding multiple-choice question \
by carefully analyzing the provided image. \nPlease answer with \
carefully thought step by step. Apply the thinking process \
recursively at both macro and micro levels. \nVerify consistency \
of reasoning and look for potential flaws or gaps during \
thinking. \nWhen realize mistakes, explain why the previous \
thinking was incorrect, fix it and then continue thinking.\nThe \
last line of your response should follow this format: 'Answer: \
\\boxed{$ANSWER}' (without quotes), where ANSWER is one of the \
options given or your final answer if no options are available."

C3_W_C4_REASON_PREAMBLE = """# System Preamble
You are in contextual safety mode. You will reject requests to generate child sexual abuse material and child exploitation material in your responses. You will accept to provide information and creative content related to violence, hate, misinformation or sex, but you will not provide any content that could directly or indirectly lead to harmful outcomes.

Your information cutoff date is June 2024.

You have been trained on data in English, French, Spanish, Italian, German, Portuguese, Japanese, Korean, Modern Standard Arabic, Mandarin, Russian, Indonesian, Turkish, Dutch, Polish, Persian, Vietnamese, Czech, Hindi, Ukrainian, Romanian, Greek and Hebrew but have the ability to speak many more languages.

## Reasoning
Start your response by writing <|START_THINKING|>. Then slowly and carefully reason through the problem. If you notice that you've made a mistake, you can correct it. You can iterate through different hypotheses, and explore different avenues that might be fruitful in solving the problem. Once you've solved the problem and sanity checked the solution say <|END_THINKING|>.
When you are ready to respond write <|START_RESPONSE|>. Summarize the key steps that led you to the solution followed by your ultimate answer at the end. Once you are done, end your response with <|END_RESPONSE|>.

# Default Preamble
The following instructions are your defaults unless specified elsewhere in developer preamble or user prompt.
- Your name is Command.
- You are a large language model built by Cohere.
- You reply conversationally with a friendly and informative tone and often include introductory statements and follow-up questions.
- If the input is ambiguous, ask clarifying follow-up questions.
- Use Markdown-specific formatting in your response (for example to highlight phrases in bold or italics, create tables, or format code blocks).
- Use LaTeX to generate mathematical notation for complex equations.
- When responding in English, use American English unless context indicates otherwise.
- When outputting responses of more than seven sentences, split the response into paragraphs.
- Prefer the active voice.
- Adhere to the APA style guidelines for punctuation, spelling, hyphenation, capitalization, numbers, lists, and quotation marks. Do not worry about them for other elements such as italics, citations, figures, or references.
- Use gender-neutral pronouns for unspecified persons.
- Limit lists to no more than 10 items unless the list is a set of finite instructions, in which case complete the list.
- Use the third person when asked to write a summary.
- When asked to extract values from source material, use the exact form, separated by commas.
- When generating code output, please provide an explanation after the code.
- When generating code output without specifying the programming language, please generate Python code.
- If you are asked a question that requires reasoning, first think through your answer, slowly and step by step, then answer."""


class Cohere_Vision_Wrapper(BaseAPI):

    is_api: bool = True
    prompts = {
        "thinking": C3_W_C4_REASON_PREAMBLE,
    }

    def __init__(
        self,
        model: str = "command-a-vision-epsilon-fp8",
        base_url: str = "https://stg.api.cohere.ai/v2/chat",
        retry: int = 10,
        timeout: int = 3*60,
        wait: int = 3,
        system_prompt: Optional[str] = None,
        verbose: bool = True,
        temperature: float = 0,
        max_tokens: int = 2048,
        **kwargs,
    ):

        self.url = base_url
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.timeout = timeout
        if system_prompt:
            self.system_prompt = self.prompts.get(system_prompt, "")

        self.key = os.environ.get("CO_API_KEY_STAGING", "")
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"BEARER {self.key}",
        }

        super().__init__(
            retry=retry,
            wait=wait,
            verbose=verbose,
            system_prompt=None,
            **kwargs,
        )

    def encode_image_file_to_base64(self, image_path, upscale=True):
        image = Image.open(image_path)
        if upscale:
            image = image.resize((image.width * 2, image.height * 2), Image.BILINEAR)

        return encode_image_to_base64(image, target_size=-1, fmt="JPEG")

    # inputs can be a lvl-2 nested list: [content1, content2, content3, ...]
    # content can be a string or a list of image & text
    def prepare_itlist(self, inputs):
        assert np.all([isinstance(x, dict) for x in inputs])
        has_images = np.sum([x["type"] == "image" for x in inputs])
        if has_images:
            upscale = has_images == 1
            content_list = []
            for i, msg in enumerate(inputs):
                if msg["type"] == "text" and msg["value"] != "":
                    content_list.append(dict(type="text", text=msg["value"]))
                elif msg["type"] == "image":
                    b64_encoded = self.encode_image_file_to_base64(msg["value"], upscale)
                    media_type = "image/jpeg"
                    content_list.append(
                        dict(
                            type="image",
                            image=f"data:{media_type};base64,{b64_encoded}",
                        )
                    )
        else:
            text = "\n".join([x["value"] for x in inputs]).strip()
            content_list = [dict(type="text", text=text)]
        return content_list

    def prepare_inputs(self, inputs):
        input_msgs = []
        input_msgs.append(dict(role="user", content=self.prepare_itlist(inputs)))
        return input_msgs

    def use_custom_prompt(self, dataset_name=None):
        if dataset_name not in ["MMVP", "MMStar", "VisuLogic", "VStarBench", "MMMU_DEV_VAL", "RealWorldQA", "MMMU_Pro_V_COT", "MMMU_Pro_10c_COT", "BLINK"]:
            return False

        def LMUDataRoot():
            if "LMUData" in os.environ and osp.exists(os.environ["LMUData"]):
                return os.environ["LMUData"]
            home = osp.expanduser("~")
            root = osp.join(home, "LMUData")
            os.makedirs(root, exist_ok=True)
            return root

        def img_root_map(dataset):
            if "MM_NIAH" in dataset:
                return "MMNIAH"
            if "CRPE" in dataset:
                return "CRPE"
            if "OCRVQA" in dataset:
                return "OCRVQA"
            if "COCO_VAL" == dataset:
                return "COCO"
            if "MMMU" in dataset:
                return "MMMU"
            if "QSpatial" in dataset:
                return "QSpatial"

            mmbench_root_map = {
                "MMBench_DEV_EN": "MMBench",
                "MMBench_TEST_EN": "MMBench",
                "MMBench_DEV_CN": "MMBench",
                "MMBench_TEST_CN": "MMBench",
                "MMBench": "MMBench",
                "MMBench_CN": "MMBench",
                "MMBench_DEV_EN_V11": "MMBench_V11",
                "MMBench_TEST_EN_V11": "MMBench_V11",
                "MMBench_DEV_CN_V11": "MMBench_V11",
                "MMBench_TEST_CN_V11": "MMBench_V11",
                "MMBench_V11": "MMBench",
                "MMBench_CN_V11": "MMBench",
            }
            if dataset in mmbench_root_map:
                return mmbench_root_map[dataset]
            return dataset

        self.img_root = osp.join(LMUDataRoot(), "images", img_root_map(dataset_name))
        return True

    def dump_image(self, line):
        os.makedirs(self.img_root, exist_ok=True)

        if "image" in line:
            if isinstance(line["image"], list):
                tgt_path = []
                if "image_path" in line:
                    image_path = line["image_path"]
                else:
                    index = line["index"]
                    image_path = [f"{index}_{i}.png" for i in range(len(line["image"]))]
                for img, im_name in zip(line["image"], image_path):
                    path = osp.join(self.img_root, im_name)
                    if not read_ok(path):
                        decode_base64_to_image_file(img, path)
                    tgt_path.append(path)

            elif isinstance(line["image"], str) and "image_path" in line:
                assert isinstance(line["image_path"], str)
                tgt_path = osp.join(self.img_root, line["image_path"])
                if not read_ok(tgt_path):
                    decode_base64_to_image_file(line["image"], tgt_path)
                tgt_path = [tgt_path]
            else:
                tgt_path = osp.join(self.img_root, f"{line['index']}.jpg")
                if not read_ok(tgt_path):
                    decode_base64_to_image_file(line["image"], tgt_path)
                tgt_path = [tgt_path]
        else:
            assert "image_path" in line
            tgt_path = toliststr(line["image_path"])
            read_ok_flag = [read_ok(x) for x in tgt_path]
            # Might be the Relative Path
            if not all(read_ok_flag):
                tgt_path_abs = [osp.join(self.img_root, x) for x in tgt_path]
                read_ok_flag = [read_ok(x) for x in tgt_path_abs]
                assert (
                    read_ok_flag
                ), f"Field `image` is missing and we could not find {tgt_path} both as absolute or relative paths. "  # noqa
                tgt_path = tgt_path_abs

        return tgt_path

    def build_prompt(self, line, dataset=None):
        prompt = line["question"]
        tgt_path = self.dump_image(line)

        hint = line["hint"] if ("hint" in line and not pd.isna(line["hint"])) else None
        if hint is not None:
            prompt = hint + "\n" + prompt
        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }

        if options:
            for key, item in options.items():
                prompt += f"\n({key}) {item}"
            preamble = "\nPlease answer directly with only the letter of the correct option and nothing else."
        else:
            preamble = "\nPlease answer directly with a single word or number."

        if dataset in ["MMVP", "MMStar", "MMMU_DEV_VAL", "MMMU_Pro_V_COT", "MMMU_Pro_10c_COT", "BLINK", "VisuLogic"]:
            preamble = "\n\n" + INTERN_VL_COT.strip()
            
        prompt += preamble
        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type="image", value=p) for p in tgt_path])
        else:
            msgs = [dict(type="image", value=tgt_path)]
        msgs.append(dict(type="text", value=prompt))

        if dataset == ["MMVP", "MMStar", "MMMU_DEV_VAL", "MMMU_Pro_V_COT", "MMMU_Pro_10c_COT", "BLINK", "VStarBench", "VisuLogic"]:
            new_prompt = self.reorganize_prompt(msgs, len(tgt_path), dataset=dataset)  
            msgs[-1]["value"] = new_prompt
            #print("new_prompt\n", new_prompt)      
        return msgs

    def reorganize_prompt(self, message, image_num, dataset=None):
        if dataset is not None and listinstr(["MUIRBench"], dataset):
            prompt = "\n".join([x["value"] for x in message if x["type"] == "text"])
            images_to_remove = " ".join(["<image>"] * image_num)
            prompt = prompt.replace(images_to_remove, "")
            for i in range(image_num):
                prompt = prompt.replace("<image>", f"<Image-{i + 1}>", 1)
            prompt = (
                "".join([f"Image-{i + 1}: <image>\n" for i in range(image_num)])
                + prompt
            )
        elif image_num == 1:
            prompt = "<image>\n" + "\n".join(
                [x["value"] for x in message if x["type"] == "text"]
            )
        else:
            prompt, image_idx = "", 1
            for x in message:
                if x["type"] == "text":
                    prompt += x["value"]
                elif x["type"] == "image":
                    prompt += f"<Image-{image_idx}>"
                    image_idx += 1
            prompt = (
                "".join([f"Image-{i + 1}: <image>\n" for i in range(image_num)])
                + prompt
            )
            images_to_remove = "".join([f"<Image-{i + 1}>" for i in range(image_num)])
            prompt = prompt.replace(images_to_remove, "")
        return prompt

    def extract_boxed_content(self, ans: str):
        idx = ans.rfind(r"\boxed{")
        if idx == -1:
            return ans

        idx += len(r"\boxed{")
        brace_level = 1
        content_start = idx
        i = idx

        while i < len(ans):
            if ans[i] == "{":
                brace_level += 1
            elif ans[i] == "}":
                brace_level -= 1
                if brace_level == 0:
                    break
            i += 1

        if brace_level != 0:
            # Unbalanced braces
            return ans

        content = ans[content_start:i]
        return content

    def generate_inner(self, inputs, **kwargs) -> str:
        messages = self.prepare_inputs(inputs)
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": messages,
            "temperature": self.temperature,
            "seed": 0,  # For reproducibility
        }

        response = requests.post(
            self.url,
            headers=self.headers,
            json=payload,
            timeout=self.timeout * 1.1,
        )
        ret_code = response.status_code
        ret_code = 0 if (200 <= int(ret_code) < 300) else ret_code
        answer = self.fail_msg

        try:
            (turn,) = response.json()["message"]["content"]
            answer = turn["text"].strip()
        except Exception as err:

            if self.verbose:
                self.logger.error(f"{type(err)}: {err}")
                self.logger.error(
                    response.text if hasattr(response, "text") else response
                )
                # raise ValueError(f"Failed to parse response: {response.text}") from err

        return ret_code, answer, response
