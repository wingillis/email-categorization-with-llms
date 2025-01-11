"""
Script to manually categorize a subset of the emails in the dataset
"""

import re
import polars as pl
from pathlib import Path
from bs4 import BeautifulSoup
from email_struct_output import Labels
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Button, Static
from textual.containers import Horizontal, VerticalScroll, Vertical


SAVE_FREQUENCY = 10  # Save every 10 emails


def clean_html(body):
    if body is None:
        return ""
    soup = BeautifulSoup(body, "html.parser")
    for sos in soup(["script", "style"]):
        sos.decompose()
    text = soup.get_text()
    text = "\n".join(line.strip() for line in text.splitlines())

    return re.sub(r"\s+", " ", text)


class EmailCategorizationApp(App):
    CSS = """
    #email-summary {
        border: solid $primary;
        padding: 1 2;
    }

    #email-scroll {
        width: 60%;
        border: solid green;
    }

    #email-content {
        padding: 1 1;
        border: solid $primary;
    }

    .cat-button {
        margin: 1 1;
        width: 100%;
    }
    """

    BINDINGS = [("q", "quit", "Quit")]

    def __init__(self):
        super().__init__()
        self.title = "Email Categorizer"
        dataset = Path("training_dataset.parquet")

        if dataset.with_stem("training_dataset_with_labels").exists():
            self.df = pl.read_parquet(dataset.with_stem("training_dataset_with_labels"))
        else:
            # copy dataset
            self.df = pl.read_parquet(dataset)
            self.df = self.df.sample(fraction=1, shuffle=True, seed=0)
            self.df = self.df.with_columns(
                pl.Series("supervised_label", [None] * len(self.df))
            )
            self.df.write_parquet(dataset.with_stem("training_dataset_with_labels"))

        new_index = (
            self.df.select(pl.arg_where(pl.col("supervised_label").is_not_null()))
            .max()
            .item()
        )
        self.email_index = 0 if new_index is None else new_index + 1

        self.supervised_data = self.df["supervised_label"].to_list()

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static(id="num-emails")
        with Horizontal():
            with Vertical(id="email-scroll"):
                yield Static(id="email-summary", expand=True)
                with VerticalScroll():
                    yield Static(id="email-content", expand=True)

            with VerticalScroll():
                for label in sorted(Labels.list()):
                    cleaned_label = re.sub(
                        r"[\(\)]", "", label.replace(" ", "-").replace(",", "")
                    )
                    yield Button(
                        label, id=cleaned_label, variant="success", classes="cat-button"
                    )
        yield Footer()

    def on_mount(self):
        self.load_email()

    def load_email(self):
        try:
            index = self.email_index
            if index >= len(self.df):
                self.exit()
                return
            _from = self.df["from"][index]
            body = self.df["body"][index]
            subject = self.df["subject"][index]

            body = f"{subject}\n\n{body}"

            self.query_one("#email-summary").update(f"Email from:\n{_from}")
            self.query_one("#email-content").update(clean_html(body))
            self.query_one("#num-emails").update(f"Total emails: {index}")
        except Exception as e:
            self.email_index += 1
            self.load_email()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.supervised_data[self.email_index] = str(event.button.label)

        if (
            self.email_index % SAVE_FREQUENCY == 0
            or self.email_index == len(self.df) - 1
        ):
            pl_df = self.df.with_columns(
                pl.Series("supervised_label", self.supervised_data)
            )
            pl_df.write_parquet("training_dataset_with_labels.parquet")

        self.email_index += 1
        self.load_email()

    def on_unmount(self, event=None):
        pl_df = self.df.with_columns(
            pl.Series("supervised_label", self.supervised_data)
        )
        pl_df.write_parquet("training_dataset_with_labels.parquet")


if __name__ == "__main__":
    app = EmailCategorizationApp()
    app.run()
