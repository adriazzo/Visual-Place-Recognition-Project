import json
import os
import re
import shlex
import subprocess
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Iterable
import urllib.parse
import urllib.request




TOKEN = "placeholder"
CHAT_ID = "placeholder"

TELEGRAM_RETRIES = 3

PYTHON_EXE_PATH = r"placeholder"

LOG_FILE = "risultati_test.log"
CLEAN_LOG_FILE = "risultati_test_pulito.log"
RESULT_JSON = "risultati_formattati.json"

DELAY_BETWEEN_TESTS = 3

COMMAND_TEMPLATE = (
    "VPR-methods-evaluation/main.py "
    "--num_workers 16 "
    "--batch_size 32 "
    "--log_dir log_dir "
    "--method={method} "
    "--backbone={backbone} "
    "--descriptors_dimension={descriptors_dimension} "
    "--image_size 512 512 "
    "--database_folder {database_folder} "
    "--queries_folder {queries_folder} "
    "--num_preds_to_save 20 "
    "--recall_values 1 5 10 20 "
    "--save_for_uncertainty "
    "--metric {metric}"
)

TQDM_PATTERN = re.compile(r"(\d+%)|(\d+/\d+)|(\|.*\|)|it/s|s/it")





ANSI_ESCAPE = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')


USEFUL_PATTERNS = [

    # command header
    r"^=+",
    r"^Esecuzione:",

    # config
    r"^Namespace\(",

    # test info
    r"Testing with",
    r"Testing on <",

    # dataset search
    r"Searching test images",

    # cache/model
    r"Returning .* model",
    r"Using cache found",

    # metrics
    r"R@\d+:",

    # output path
    r"The outputs are being saved",

    # final
    r"Execution time:",
]


GARBAGE_PATTERNS = [

    # tqdm
    r"\d+%\|",
    r"it/s",
    r"s/it",

    # prediction save spam
    r"Saving preds in",

    # empty tqdm leftovers
    r"^\s*$",

    # amd noise
    r"amdgpu.ids",
]


USEFUL_REGEX = [re.compile(p) for p in USEFUL_PATTERNS]
GARBAGE_REGEX = [re.compile(p) for p in GARBAGE_PATTERNS]




def clean_output(text: str) -> str:
    """
    Rimuove righe spazzatura
    """
    cleaned_lines = []

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if "\r" in line:
            print(f"--> RIGUZZATA DA bacslesh errre: {repr(line)}") # <--- Guarda qui la console!
            continue
        if TQDM_PATTERN.search(line):
            print(f"--> RIGUZZATA DA TQDM: {repr(line)}") # <--- Guarda qui la console!
            continue
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)



def clean_output(text: str) -> str:

    cleaned_lines = []

    seen = set()

    for raw_line in text.splitlines():
        line = ANSI_ESCAPE.sub("", raw_line)

        line = line.strip()

        if not line:
            continue

        if any(r.search(line) for r in GARBAGE_REGEX):
            continue

        if not any(r.search(line) for r in USEFUL_REGEX):
            continue

        if line in seen:
            continue

        seen.add(line)

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)

def compact_summary(text: str) -> str:

    lines = clean_output(text).splitlines()

    important = []

    for line in lines:
        if (
            "Testing with" in line
            or "R@" in line
            or "Execution time:" in line
        ):
            important.append(line)

    return "\n".join(important)


def telegram_request(method: str, payload=None, files=None, retries=TELEGRAM_RETRIES):

    url = f"https://api.telegram.org/bot{TOKEN}/{method}"

    for attempt in range(retries):
        try:
            if files:
                boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
                body = bytearray()

                for key, value in payload.items():
                    body.extend(f"--{boundary}\r\n".encode())
                    body.extend(
                        (
                            f'Content-Disposition: form-data; '
                            f'name="{key}"\r\n\r\n{value}\r\n'
                        ).encode()
                    )

                for field_name, filepath in files.items():
                    filename = os.path.basename(filepath)

                    with open(filepath, "rb") as f:
                        data = f.read()

                    body.extend(f"--{boundary}\r\n".encode())
                    body.extend(
                        (
                            f'Content-Disposition: form-data; '
                            f'name="{field_name}"; '
                            f'filename="{filename}"\r\n'
                        ).encode()
                    )

                    body.extend(
                        b"Content-Type: application/octet-stream\r\n\r\n"
                    )

                    body.extend(data)
                    body.extend(b"\r\n")

                body.extend(f"--{boundary}--\r\n".encode())
                req = urllib.request.Request(url, data=body)

                req.add_header(
                    "Content-Type",
                    f"multipart/form-data; boundary={boundary}"
                )
            else:
                data = urllib.parse.urlencode(payload).encode()
                req = urllib.request.Request(url, data=data)

            with urllib.request.urlopen(req, timeout=20) as response:
                print("Messaggio inviato!", url, payload, files, "", sep="\n")
                return response.read().decode()
        

        except (
                urllib.error.URLError,
                ConnectionResetError,
                TimeoutError,
            ) as e:

                print(
                    f"[Telegram Error] Tentativo "
                    f"{attempt + 1}/{retries}: {e}"
                )

                time.sleep(5 * (attempt + 1))

        except Exception as e:
            print(f"[Telegram Error] {e}")
            print("Richiesta fallita", url, payload, files, "", sep="\n")
    
        finally:
            if attempt >= (retries):
                print("Non sono riuscito ad inviare il messaggio", url, payload, files, "", sep="\n")


def send_message(text: str):
    return telegram_request(
        "sendMessage",
        payload={
            "chat_id": CHAT_ID,
            "text": text[:4000]
        }
    )


def send_document(filepath: str):

    return telegram_request(
        "sendDocument",
        payload={
            "chat_id": CHAT_ID
        },
        files={
            "document": filepath
        }
    )


@dataclass
class CommandResult:
    command: str
    return_code: int
    execution_time: float
    output: str


def run_command(python_exe: str, command: str) -> CommandResult:

    full_command = [python_exe] + shlex.split(command)
    start = time.time()
    process = subprocess.Popen(
        full_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace"
    )

    collected_output = []

    for line in process.stdout:
        print(line, end="")
        collected_output.append(line)

    process.wait()
    elapsed = time.time() - start
    output = "".join(collected_output)

    return CommandResult(
        command=command,
        return_code=process.returncode,
        execution_time=elapsed,
        output=output
    )


def extract_short_name(command: str) -> str:

    method_match = re.search(r"--method=(\w+)", command)
    backbone_match = re.search(r"--backbone=(\w+)", command)

    method = method_match.group(1) if method_match else "unknown"
    backbone = backbone_match.group(1) if backbone_match else "unknown"

    return f"{method} ({backbone})"


def run_commands(
    python_exe: str,
    commands: Iterable[str],
    log_file: str,
    delay: int = 2
):

    with open(log_file, "w", encoding="utf-8") as log:

        total = len(commands)

        for idx, cmd in enumerate(commands, start=1):

            short_name = extract_short_name(cmd)
            send_message(
                f"[{idx}/{total}] Avvio test:\n{short_name}"
            )

            print("\n" + "=" * 80)
            print(f"TEST {idx}/{total}")
            print(short_name)
            print("=" * 80 + "\n")

            result = run_command(python_exe, cmd)
            cleaned_output = clean_output(result.output)
            header = (
                f"\n{'=' * 80}\n"
                f"COMMAND:\n{cmd}\n"
                f"{'=' * 80}\n"
            )

            log.write(header)
            log.write(cleaned_output)
            log.write(
                f"\nExecution time: {result.execution_time:.2f}s\n"
            )
            log.flush()

            execution_time_str = str(
                timedelta(seconds=int(result.execution_time))
            )

            summary = (
                f"Test completato\n\n"
                f"Test: {short_name}\n"
                f"Tempo: {execution_time_str}\n"
                f"Exit code: {result.return_code}"
            )

            send_message(summary)

            if result.return_code != 0:
                send_message(
                    f"Errore durante esecuzione:\n{short_name}"
                )
                break

            print(f"\nCooldown {delay} secondi...\n")
            time.sleep(delay)

def parse_multiple_logs(file_path):

    if not os.path.exists(file_path):
        print(f"[-] Errore: Il file '{file_path}' non esiste.")
        return []

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        log_text = f.read()

    raw_tests = re.split(r'(?=Namespace\()', log_text)
    parsed_tests = []
    test_id = 1

    for raw_test in raw_tests:
        if "Namespace(" not in raw_test:
            continue
        if "Execution time:" not in raw_test:
            continue
        test_data = {
            "test_number": test_id
        }

        # =====================================================
        # PARAMETERS
        # =====================================================

        namespace_match = re.search(
            r'Namespace\((.*?)\)',
            raw_test,
            re.DOTALL
        )

        if namespace_match:
            
            param_string = (
                namespace_match
                .group(1)
                .replace('\n', '')
                .replace('\r', '')
            )

            params = re.findall(
                r'(\w+)=([^,]+(?:,\[[^\]]*\])?)',
                param_string
            )

            test_data["parameters"] = {}

            for key, value in params:

                test_data["parameters"][key] = (
                    value.strip().strip("'\"")
                )

        # =====================================================
        # DATASET INFO
        # =====================================================

        counts_match = re.search(
            r"#queries:\s*(\d+);\s*#database:\s*(\d+)",
            raw_test
        )

        if counts_match:

            test_data["dataset_info"] = {
                "queries": int(counts_match.group(1)),
                "database": int(counts_match.group(2))
            }

        # =====================================================
        # RECALL
        # =====================================================

        recall_matches = re.findall(
            r"R@(\d+):\s*([\d.]+)",
            raw_test
        )

        if recall_matches:

            test_data["performance"] = {
                f"R@{k}": float(v)
                for k, v in recall_matches
            }

        # =====================================================
        # EXECUTION TIME
        # =====================================================

        time_match = re.search(
            r"Execution time:\s*([\d.]+)",
            raw_test
        )

        if time_match:

            total_seconds = float(time_match.group(1))
            test_data["execution_time"] = {
                "seconds": total_seconds,
                "formatted": str(
                    timedelta(seconds=int(total_seconds))
                )
            }

        parsed_tests.append(test_data)
        test_id += 1

    return parsed_tests


COMMANDS = [
        COMMAND_TEMPLATE.format(method="megaloc", backbone="ResNet18", descriptors_dimension="4096", database_folder=r"data/svox/images/test/gallery", queries_folder=r"data/svox/images/test/queries", metric="L2"),
        COMMAND_TEMPLATE.format(method="mixvpr", backbone="ResNet50", descriptors_dimension="512", database_folder=r"data/svox/images/test/gallery", queries_folder=r"data/svox/images/test/queries", metric="L2"),
        COMMAND_TEMPLATE.format(method="cosplace", backbone="ResNet18", descriptors_dimension="512", database_folder=r"data/svox/images/test/gallery", queries_folder=r"data/svox/images/test/queries", metric="L2"),
        COMMAND_TEMPLATE.format(method="netvlad", backbone="VGG16", descriptors_dimension="4096", database_folder=r"data/svox/images/test/gallery", queries_folder=r"data/svox/images/test/queries", metric="L2"),
            
        COMMAND_TEMPLATE.format(method="megaloc", backbone="ResNet18", descriptors_dimension="4096", database_folder=r"data/sf_xs/test/database", queries_folder=r"data/sf_xs/test/queries", metric="L2"),
        COMMAND_TEMPLATE.format(method="mixvpr", backbone="ResNet50", descriptors_dimension="512", database_folder=r"data/sf_xs/test/database", queries_folder=r"data/sf_xs/test/queries", metric="L2"),
        COMMAND_TEMPLATE.format(method="cosplace", backbone="ResNet18", descriptors_dimension="512", database_folder=r"data/sf_xs/test/database", queries_folder=r"data/sf_xs/test/queries", metric="L2"),
        COMMAND_TEMPLATE.format(method="netvlad", backbone="VGG16", descriptors_dimension="4096", database_folder=r"data/sf_xs/test/database", queries_folder=r"data/sf_xs/test/queries", metric="L2"),
    
        COMMAND_TEMPLATE.format(method="megaloc", backbone="ResNet18", descriptors_dimension="4096", database_folder=r"data/tokyo_xs/test/database", queries_folder=r"data/tokyo_xs/test/queries", metric="L2"),
        COMMAND_TEMPLATE.format(method="mixvpr", backbone="ResNet50", descriptors_dimension="512", database_folder=r"data/tokyo_xs/test/database", queries_folder=r"data/tokyo_xs/test/queries", metric="L2"),
        COMMAND_TEMPLATE.format(method="cosplace", backbone="ResNet18", descriptors_dimension="512", database_folder=r"data/tokyo_xs/test/database", queries_folder=r"data/tokyo_xs/test/queries", metric="L2"),
        COMMAND_TEMPLATE.format(method="netvlad", backbone="VGG16", descriptors_dimension="4096", database_folder=r"data/tokyo_xs/test/database", queries_folder=r"data/tokyo_xs/test/queries", metric="L2"),
]

if __name__ == "__main__":

    send_message("Avvio batch test")

    run_commands(
        python_exe=PYTHON_EXE_PATH,
        commands=COMMANDS,
        log_file=LOG_FILE,
        delay=DELAY_BETWEEN_TESTS
    )

    send_message("Parsing e pulizia log")

    # CLEAN LOG

    with open(LOG_FILE, "r", encoding="utf-8") as f:
        raw_log = f.read()

    cleaned_log = clean_output(raw_log)

    with open(CLEAN_LOG_FILE, "w", encoding="utf-8") as f:
        f.write(cleaned_log)

    send_message("invio log pulito")

    send_document(CLEAN_LOG_FILE)

    parsed_data = parse_multiple_logs(CLEAN_LOG_FILE)

    with open(RESULT_JSON, "w", encoding="utf-8") as f:
        json.dump(
            parsed_data,
            f,
            indent=4,
            ensure_ascii=False
        )

    send_message("invio JSON risultati")

    send_document(RESULT_JSON)

    send_message("Run completata")
