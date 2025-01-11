import re
import tyro
import time
import enum
import datetime
import polars as pl
from math import ceil
from pathlib import Path
from ollama import Client
from tqdm.auto import tqdm
from textwrap import dedent
from bs4 import BeautifulSoup
from toolz import partial, dissoc
from pydantic import BaseModel, Field
from typing import Annotated, Iterable


MODELS = {
    "smollm2": "smollm2",
    "smollm2-360m": "smollm2:360m",
    "granite3_1-dense-2b": "granite3.1-dense:2b",
    "falcon3_3b": "falcon3:3b",
    "qwen-1_5b": "qwen2.5:1.5b-instruct-q8_0",
    "llama3_2": "llama3.2:3b-instruct-q8_0",
    "granite3_1-dense-8b": "granite3.1-dense",
    "granite3_2-dense-2b": "granite3.2:2b-instruct-q8_0",
    "granite3_2-dense-8b": "granite3.2:8b",
    "llama3_2-1b": "llama3.2:1b",
    "tulu3": "tulu3",
    "tulu3_1": "hf.co/bartowski/allenai_Llama-3.1-Tulu-3.1-8B-GGUF:Q6_K",
    "llama3_1": "llama3.1:8b-instruct-q4_K_M",
    "qwen-3b": "qwen2.5:3b",
    "qwen-7b": "qwen2.5:7b",
    "falcon3_7b": "falcon3",
    "smallthinker": "smallthinker",
    "dolphin3": "dolphin3",
    "marco-o1": "marco-o1",
    "nemo": "mistral-nemo:12b-instruct-2407-q4_K_M",
    "command-r7b": "command-r7b",
    "internlm3": "lly/InternLM3-8B-Instruct:8b-instruct-q4_k_m",
    "deepseek-r1_8b": "deepseek-r1:8b",
    "deepseek-r1_7b": "deepseek-r1:7b",
    "deepseek-r1_1.5b": "deepseek-r1:1.5b",
    "qwen-0_5b": "qwen2.5:0.5b-instruct-q8_0",
    "phi4-mini": "phi4-mini:3.8b-q8_0",
    "granite3_1-moe-1b": "granite3.1-moe:1b",  # oddly slow
    "phi4": "phi4",
    "qwq": "qwq",
    "qwen-32b": "qwen2.5:32b-instruct-q3_K_M",
    "gemma2": "gemma2",
    "granite3_1-moe-3b": "granite3.1-moe",  # too slow - 4sec per email
    "olmo2": "olmo2",  # slow
    "mistral-small": "mistral-small",
    "llama3.3": "llama3.3:70b-instruct-q3_K_M",
}


class Config(BaseModel):
    chunk_size: int = 10
    full_dataset: bool = False
    """Flag that determines whether to use the full email dataset or the supervised subset."""
    select_models: tuple[str, ...] = tuple(MODELS)
    """A list of model names to run, taken from the MODELS dict."""


def dict_take(d: dict, keys: Iterable[str]) -> dict:
    """returns a copy of the dictionary with only the specified keys."""
    return dissoc(d, *(set(d) - set(keys)))


class Labels(str, enum.Enum):
    """Enumeration of possible pre-defined email labels."""

    # Keep labels
    correspondence = "Personal or professional correspondence"
    financial = "Financial information"
    medical = "Medical information"
    project = "Programming, educational, and technical information"
    news = "News alerts and newsletters"
    scheduling = "Travel, scheduling and calendar events"
    shopping = "Shopping and order confirmations"
    other = "Other"

    # Discard labels
    promotional = "Promotional emails (marketing, sales, advertisements)"
    customer_service = "Customer service and support"
    security = "Account security and privacy"

    @classmethod
    def list(cls):
        return list(x.value for x in cls)

    @classmethod
    def keep(cls):
        return list(x.value for x in cls)[:8]


class Prediction(BaseModel):
    """Correctly identified class labels and alternative label of the given email."""

    class_label: Labels
    alternative_label: Annotated[
        str,
        Field(
            pattern="^[a-z]+( [a-z]+){0,2}$",
            description="1-3 word alternative categorical label",
        ),
    ]
    subject_suggestion: Annotated[
        str, Field(pattern="^[a-z]+$", description="1 word subject suggestion")
    ]


def remove_urls(body):
    if body is None:
        return ""
    pattern = re.compile(r"https?://\S+")

    return pattern.sub("", body)


def remove_multi_newlines(body):
    return re.sub(r"[\n\r]+", "\n", body)


def clean_html(body):
    if body is None:
        return ""
    soup = BeautifulSoup(body, "html.parser")
    for sos in soup(["script", "style"]):
        sos.decompose()
    text = soup.get_text()
    lines = remove_multi_newlines(text)
    text = "\n".join(line.strip() for line in lines.splitlines())

    return re.sub(r"\s+", " ", text)


def categorize_email(
    subject: str,
    body: str,
    model_choice: str,
    from_email: str,
    email_date: datetime.date,
    client: Client,
    temperature: float = 0,
) -> Prediction:
    email_string = dedent(
        """\
    Categorize the following email using one of the provided labels. If the email does not fit any of the provided labels, use the "Other" label.
    Provide an alternative label using one to three words of your choice. This alternative label must be a fitting yet general category for the email.
    Also, provide a suggestion for the email subject using only a single word.

    Here are examples of the provided labels:
    - Personal or professional correspondence: emails with family, friends, or colleagues. They have a more colloquial tone.
    - Financial information: bank statements, investment updates, bills, receipts, financial aid, scholarships, or grants.
    - Medical information: doctor's appointments, prescriptions, medical test results, or health insurance information.
    - Programming, educational, and technical information: coding projects, GitHub, educational resources, google scholar, or technical documentation.
    - News alerts and newsletters: news updates, newsletters, or subscriptions.
    - Travel, scheduling and calendar events: flight confirmations, reservations, scheduling, or calendar events.
    - Shopping and order confirmations: online or food shopping, order confirmations, delivery updates.
    - Promotional emails (marketing, sales, advertisements): promotional emails, marketing, sales, or advertisements.
    - Customer service and support: customer service, support, or help desk.
    - Account security and privacy: account security, privacy, or data protection.
    - Other: emails that do not fit any of the provided labels.

    Here is the cleaned email body:
    <email>
    From: {from_}
    Date: {email_date}
    Subject: {subject}

    Body:
    <body>
    {email}
    </body>
    </email>

    Again, categorize the following email using one of the provided labels. If the email does not fit any of the provided labels, use the "Other" label.
    Provide an alternative label using one to three words of your choice. This alternative label must be a fitting yet general category for the email.
    Also, provide a suggestion for the email subject using only a single word.
    """
    )
    cleaned_body = clean_html(remove_urls(body)).strip()
    if re.sub(r"\s+", "", cleaned_body).strip() == "":
        cleaned_body = body

    message_list = [
        {
            "role": "system",
            "content": (
                "You are an expert email categorizer and summarizer. "
                "You must categorize the following emails with a provided label. "
                "Emails are found within XML <email> tags. Respond with JSON."
            ),
        },
        {
            "role": "user",
            "content": email_string.format(
                subject=subject,
                email=cleaned_body,
                date=datetime.date.today(),
                email_date=email_date,
                from_=from_email,
            ),
        },
    ]

    response = client.chat(
        messages=message_list,
        model=model_choice,
        format=Prediction.model_json_schema(),
        options={"temperature": temperature, "num_ctx": 13_000},
    )
    out = Prediction.model_validate_json(response.message.content)
    return out


def process_email(
    row,
    model_choice,
    client,
    temperature: float = 0,
):
    start = time.time()
    try:
        out = categorize_email(
            row["subject"],
            row["body"],
            model_choice,
            row["from"],
            row["date"].date(),
            client,
            temperature=temperature,
        )
        end = time.time()
        return {
            **row,
            "primary_category": out.class_label.value,
            "subject_suggestion": out.subject_suggestion,
            "alternative_category": out.alternative_label,
            "inference_duration": end - start,
            "model": model_choice,
        }
    except Exception as e:
        print("process_email:")
        print(f"\n\nError: {e}")
        return {
            **row,
            "primary_category": "N/A",
            "subject_suggestion": "N/A",
            "alternative_category": "N/A",
            "inference_duration": time.time() - start,
            "model": model_choice,
        }


def load_supervised_dataset():
    files = sorted(Path(".").glob("training_dataset*_v*.parquet"))
    assert len(files) > 0, "No supervised datasets found."

    df = pl.concat(
        (
            pl.read_parquet(file)
            .drop_nulls(subset=["supervised_label"])
            .select(["from", "body", "date", "subject", "supervised_label"])
            for file in files
        )
    ).sort("date")

    return df


def run_predictions(dataset: pl.DataFrame, config: Config):
    keep_models = dict_take(MODELS, config.select_models)
    folder = "structured_output"
    if not config.full_dataset:
        folder += "_supervised"

    client = Client(timeout=120)
    for j, frame in tqdm(
        enumerate(dataset.iter_slices(n_rows=config.chunk_size)),
        desc=f"Processing chunks...",
        total=ceil(len(dataset) / config.chunk_size),
    ):
        for model_name, ollama_name in keep_models.items():
            path = Path(
                folder,
                model_name,
                f"categorized_emails_{model_name}_subset_{j:03d}.parquet",
            )
            path.parent.mkdir(exist_ok=True, parents=True)

            # get list of models
            response = client.list()

            # check to see if model_choice is within
            if not any(map(lambda x: x.model == ollama_name, response.models)):
                print(
                    f"Model {ollama_name} not found. Downloading...(this may take a while)"
                )
                client.pull(ollama_name)

            process_with_model = partial(
                process_email, model_choice=ollama_name, temperature=0, client=client
            )
            if not path.exists():
                results = list(
                    map(
                        process_with_model,
                        tqdm(
                            frame.iter_rows(named=True),
                            total=len(frame),
                            leave=False,
                            desc=f"Processing emails with {model_name}",
                        ),
                    )
                )
                new_frame = pl.DataFrame(results)
                new_frame.write_parquet(path)


def main(config: Config):

    if config.full_dataset:
        # shuffle for uniform sampling across time
        df = (
            pl.read_parquet("emails.parquet")
            .sort("date")
            .sample(fraction=1, shuffle=True, seed=0)
        )
    else:
        df = load_supervised_dataset().sort("date")

    run_predictions(df, config)


if __name__ == "__main__":
    config = tyro.cli(Config)
    main(config)
