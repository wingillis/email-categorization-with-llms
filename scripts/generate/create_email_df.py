import tyro
import email
import imaplib
import tomllib
import chardet
import datetime
import tenacity
import polars as pl
from tqdm.auto import tqdm
from pydantic import BaseModel
from email.header import decode_header
from email.utils import parsedate_to_datetime

class Config(BaseModel):
    credentials: str = "credentials.toml"


def decode(payload):
    if isinstance(payload, str):
        return payload
    elif isinstance(payload, bytes):
        try:
            return payload.decode("utf-8")
        except UnicodeDecodeError:
            try:
                detected = chardet.detect(payload)
                if detected["encoding"]:
                    return payload.decode(detected["encoding"])
                else:
                    # If chardet couldn't detect the encoding, try common encodings
                    for encoding in ["iso-8859-1", "windows-1252", "ascii"]:
                        try:
                            return payload.decode(encoding)
                        except UnicodeDecodeError:
                            continue
                    # If all else fails, decode with 'utf-8' and replace undecodable characters
                    return payload.decode("utf-8", errors="replace")
            except Exception as e:
                print(f"Error decoding payload: {e}")
                return payload.decode("utf-8", errors="replace")
    else:
        return str(payload)


def get_body(msg):

    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() in (
                "text/plain",
                "text/html",
            ) and "attachment" not in str(part.get("Content-Disposition")):
                return decode(part.get_payload(decode=True))

    return decode(msg.get_payload(decode=True))


def is_attachment(part):
    return (
        part.get("Content-Disposition") is not None
        and part.get_content_maintype() != "multipart"
    )


def get_attachments(msg):
    attachments = []

    for part in filter(is_attachment, msg.walk()):
        filename = part.get_filename()
        if bool(filename):
            attachments.append(filename)
    return attachments


def decode_component(component):
    try:
        data, _ = decode_header(component)[0]
        return decode(data)
    except TypeError:
        return ""


@tenacity.retry(wait=tenacity.wait_fixed(10))
def parse_email(imap, num: int):
    _, msg_data = imap.fetch(str(num), "(RFC822)")

    # get the tuples out of msg_data - those are the responses
    for response in filter(lambda x: isinstance(x, tuple), msg_data):
        email_body = email.message_from_bytes(response[1])
        subject = decode_component(email_body.get("Subject", ""))
        from_ = decode_component(email_body.get("From", ""))

        date_str = email_body.get("Date", "")
        if date_str:
            date = parsedate_to_datetime(date_str)
        else:
            # if date is not found, set an absurdly old date
            date = datetime.datetime(year=1800, month=1, day=1)
        return {
            "subject": subject,
            "from": from_,
            "date": date,
            "date_str": date_str,
            "size": len(response[1]),
            "attachments": len(get_attachments(email_body)),
            "multipart": email_body.is_multipart(),
            "body": get_body(email_body),
        }


def get_emails(email_address, password):
    email_df_name = "emails.parquet"

    # connect to the Gmail IMAP server
    imap = imaplib.IMAP4_SSL("imap.gmail.com")

    # login to the account
    imap.login(email_address, password)

    try:
        # select the mailbox to read
        _, message_count = imap.select('"[Gmail]/All Mail"', readonly=True)

        emails = []

        # fetch and process each email
        for i in tqdm(
            range(1, int(message_count[0].decode())), desc="Processing emails..."
        ):
            emails.append(parse_email(imap, i))

            # save dataframe every 500 emails - helpful for early stopping
            if i % 500 == 0:
                email_df = pl.DataFrame(emails)
                email_df.write_parquet(email_df_name)

        email_df = pl.DataFrame(emails)
        email_df.write_parquet(email_df_name, compression_level=4)
    finally:
        # close the connection
        imap.close()
        imap.logout()


if __name__ == "__main__":
    config = tyro.cli(Config)

    with open(config.credentials, "rb") as f:
        credentials = tomllib.load(f)

    get_emails(credentials["email"], credentials["password"])
