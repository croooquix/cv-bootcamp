"""경로 해석 규칙 테스트.

인덱싱과 서빙이 서로 다른 작업 디렉토리에서, 때로는 다른 머신에서 실행되므로
경로 규칙이 깨지면 검색 결과의 이미지가 404가 됩니다. paths 모듈은 torch나
cv2를 임포트하지 않아 이 테스트는 모델 의존성 없이 돌아갑니다.
"""
import os
import sys

import pytest

SRC_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"
)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import paths  # noqa: E402

ROOT = paths.PROJECT_ROOT


def j(*parts):
    return os.path.normpath(os.path.join(ROOT, *parts))


def test_project_root_is_repository_root():
    assert os.path.isdir(os.path.join(ROOT, "week4"))
    assert os.path.isfile(paths.DEFAULT_CONFIG_PATH)


@pytest.mark.parametrize("cwd", ["", "week4", os.path.join("week4", "src")])
def test_config_is_independent_of_working_directory(cwd, monkeypatch):
    monkeypatch.chdir(os.path.join(ROOT, cwd) if cwd else ROOT)
    config = paths.load_config()

    assert config["paths"]["dataset_images_dir"] == j(
        "week3", "data", "train", "images"
    )
    assert config["paths"]["index_save_path"] == j(
        "week4", "index", "fashion_index.faiss"
    )


def test_fallback_model_name_is_not_treated_as_a_path():
    # ultralytics가 이름으로 받아 내려받는 값이라 경로로 바꾸면 안 됩니다.
    assert paths.load_config()["paths"]["yolo_fallback_path"] == "yolov8n.pt"


def test_absolute_config_path_is_left_alone():
    assert paths.resolve_path("C:/elsewhere/best.pt") == os.path.normpath(
        "C:/elsewhere/best.pt"
    )


def test_empty_path_passes_through():
    assert paths.resolve_path(None) is None
    assert paths.resolve_path("") == ""


def test_paths_inside_the_repository_become_relative():
    assert paths.to_project_relative(j("week3", "data", "a.jpg")) == (
        "week3/data/a.jpg"
    )


def test_backslashes_are_normalised():
    mixed = ROOT + r"\week3\data\train\images\b.jpg"
    assert paths.to_project_relative(mixed) == "week3/data/train/images/b.jpg"


def test_paths_outside_the_repository_stay_absolute():
    outside = paths.to_project_relative(os.path.join(os.sep, "elsewhere", "c.jpg"))
    assert not outside.startswith("week")


def test_relative_metadata_path_resolves_against_the_root():
    assert paths.resolve_indexed_image_path("week3/data/train/images/a.jpg") == j(
        "week3", "data", "train", "images", "a.jpg"
    )


@pytest.mark.parametrize(
    "recorded",
    [
        "c:/Users/Someone/Projects/cv-bootcamp/week3/data/train/images\\b.jpg",
        "/mnt/c/Users/Someone/Projects/cv-bootcamp/week3/data/train/images/b.jpg",
        "/home/someone/cv-bootcamp/week3/data/train/images/b.jpg",
    ],
)
def test_index_built_on_another_machine_is_remapped(recorded):
    # 윈도우·WSL·리눅스에서 만든 인덱스 모두 현재 루트로 옮겨져야 합니다.
    assert paths.resolve_indexed_image_path(recorded) == j(
        "week3", "data", "train", "images", "b.jpg"
    )


def test_project_root_can_be_overridden(monkeypatch):
    monkeypatch.setenv("FASHION_PROJECT_ROOT", os.path.join(os.sep, "srv", "assets"))
    assert paths.resolve_indexed_image_path("week3/data/a.jpg") == os.path.normpath(
        os.path.join(os.sep, "srv", "assets", "week3", "data", "a.jpg")
    )
