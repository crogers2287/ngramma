"""Compile only pure lens protocol helpers; never link or load a model."""
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def protocol(tmp_path_factory):
    engine = os.environ.get("NGRAMMA_ENGINE_SOURCE")
    compiler = shutil.which(os.environ.get("CXX", "c++"))
    if not engine or not compiler:
        pytest.skip("Set NGRAMMA_ENGINE_SOURCE and provide a C++ compiler for model-free lens tests")
    include = Path(engine) / "vendor" / "nlohmann"
    if not (include / "json.hpp").is_file():
        pytest.skip("Configured engine lacks vendor/nlohmann/json.hpp")
    directory = tmp_path_factory.mktemp("lens-protocol")
    source = directory / "test.cpp"
    source.write_text('#define NGRAMMA_LENS_PROTOCOL_ONLY\n#include "' + str(ROOT / 'src/ngramma_runtime/native/lens.cpp') + '"\n' + r'''
#include <iostream>
int main() {
    using namespace generation_protocol;
    json r; std::cin >> r;
    try {
        int vocabulary=r.value("vocabulary",6);
        auto c=parse(r.at("job"),r.value("context",128),vocabulary,r.value("prompt_length",3));
        json out={{"ok",true},{"top_k",c.top_k},{"compact",c.compact},{"context_tokens",c.context_tokens}};
        if(r.contains("tokens")) out["tokens"]=token_ids(r["tokens"],vocabulary);
        if(r.contains("logits")) {
            auto x=r["logits"].get<std::vector<float>>();
            if(int(x.size())!=vocabulary) throw std::runtime_error("Fixture length mismatch");
            if(r.value("nonfinite","")=="nan") x[0]=std::numeric_limits<float>::quiet_NaN();
            if(r.value("nonfinite","")=="inf") x[0]=std::numeric_limits<float>::infinity();
            out["first_step"]=first_step(x.data(),vocabulary,c);
        }
        std::cout << out.dump();
    } catch(const std::exception & e) {std::cout << json({{"ok",false},{"error",e.what()}}).dump();}
}
''')
    executable = directory / "protocol"
    subprocess.run([compiler, "-std=c++17", "-O0", "-I", str(include), str(source), "-o", str(executable)], check=True, capture_output=True)

    def run(g=None, **extra):
        request = {"job": {"generate": {"max_new_tokens": 4, **(g or {})}}, **extra}
        return json.loads(subprocess.run([str(executable)], input=json.dumps(request), text=True, capture_output=True, check=True).stdout)
    return run


def test_defaults_and_small_vocab(protocol):
    assert protocol()["compact"] is True
    assert protocol(vocabulary=2)["top_k"] == 2


@pytest.mark.parametrize("key,value", [
    ("max_new_tokens", True), ("max_new_tokens", 0), ("max_new_tokens", 1025),
    ("max_new_tokens", 2**64-1), ("max_new_tokens", 2.0),
    ("context_tokens", 129), ("context_tokens", 0), ("context_tokens", False),
    ("top_k", -1), ("top_k", 7), ("top_k", True), ("compact", 1),
    ("score_tokens", [0, 0]), ("score_tokens", [6]), ("score_tokens", [-1]),
    ("score_tokens", [True]), ("score_tokens", [1.0]), ("score_tokens", "1"),
    ("score_tokens", list(range(257))), ("grammar", "A|B"), ("temperature", 0.5),
])
def test_rejects_invalid_or_forcing_options(protocol, key, value):
    assert protocol({key: value})["ok"] is False


@pytest.mark.parametrize("length", [0, 128, 129])
def test_prompt_requires_context_room(protocol, length):
    assert not protocol(prompt_length=length)["ok"]


def test_noncompact_requires_destination(protocol):
    assert not protocol({"compact": False})["ok"]
    assert protocol(job={"generate": {"max_new_tokens": 4, "compact": False}, "output_dir": "synthetic-unused"})["ok"]


def test_tokenize_conflict_and_missing_budget(protocol):
    assert not protocol(job={"generate": {"max_new_tokens": 4}, "tokenize_only": True})["ok"]
    assert not protocol(job={"generate": {}})["ok"]


@pytest.mark.parametrize("tokens", [[True], [1.0], [-1], [6], [2**64-1], "1", [0]*32769])
def test_prompt_token_validation(protocol, tokens):
    assert not protocol(tokens=tokens)["ok"]


def test_scores_do_not_force_answer_and_ties_are_stable(protocol):
    result = protocol({"top_k": 3, "score_tokens": [4, 0]}, logits=[-2, 7, 7, 0, -9, 1])
    assert result["ok"]
    step = result["first_step"]
    assert step["greedy_token_id"] == 1
    assert step["top_logits"] == [{"token_id": 1, "logit": 7}, {"token_id": 2, "logit": 7}, {"token_id": 5, "logit": 1}]
    assert step["requested_logits"] == [{"token_id": 4, "logit": -9}, {"token_id": 0, "logit": -2}]
    assert protocol({"top_k": 0}, logits=[-2, 7, 7, 0, -9, 1])["first_step"]["top_logits"] == []


@pytest.mark.parametrize("kind", ["nan", "inf"])
def test_nonfinite_logits_rejected(protocol, kind):
    assert not protocol(logits=[0]*6, nonfinite=kind)["ok"]
