# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
from pathlib import Path

import pytest
from openai import OpenAI

AUDIO_PATH = Path(__file__).parent.parent / "dataset" / "audio.wav"


@pytest.fixture(scope="module")
def client(host, port):
    client = OpenAI(base_url=f"http://{host}:{port}/v1/", timeout=30, api_key="-")
    yield client
    client.close()


@pytest.mark.qaic_test_config(
    model_name="openai/whisper-tiny.en", dtype="mxfp6", kv_dtype="auto", ctx_len=448,
    decode_bsz=1, num_device_groups=1, device_group_size=1,
)
def test_audio(
    client, server_runner, model_name, host, port, seq_len, ctx_len, decode_bsz,
    dtype, kv_dtype, device_group,
):
    additional_config = {"device_group": device_group}

    with server_runner(
        server_runner.Backend.OPENAI_API_SERVER_MODULE,
        model_name, host, port, seq_len, ctx_len, decode_bsz, dtype, kv_dtype,
        additional_config, max_num_batched_tokens=1500, trust_remote_code=True,
    ):
        with open(AUDIO_PATH, "rb") as f:
            transcription = client.audio.transcriptions.create(
                file=f,
                model=model_name,
                language="en",
                response_format="json",
                temperature=0.0,
            )
            print(transcription)
            print("transcription result:", transcription.text)
