"""설정과 경로 해석.

torch나 cv2 없이 임포트할 수 있도록 utils.py에서 분리했습니다. 경로 규칙만
독립적으로 테스트할 수 있고(week4/tests/test_paths.py), 무거운 모델 의존성을
끌어오지 않습니다.
"""
import os

import yaml

# 이 파일(week4/src/paths.py)에서 두 단계 위가 저장소 루트입니다.
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
)

DEFAULT_CONFIG_PATH = os.path.join(
    PROJECT_ROOT, "week4", "config", "search_config.yaml"
)

# paths 항목 중 실제 파일 경로인 것들. yolo_fallback_path는 ultralytics가 이름으로
# 받아 직접 내려받는 모델명이므로 경로로 해석하지 않습니다.
_RESOLVED_PATH_KEYS = (
    "yolo_model_path",
    "dataset_images_dir",
    "index_save_path",
    "metadata_save_path",
)


def resolve_path(path):
    """상대 경로를 저장소 루트 기준 절대 경로로 바꿉니다.

    이미 절대 경로면 구분자만 정리해 그대로 돌려줍니다. 설정에 상대 경로를
    적어 두면 작업 디렉토리나 머신이 달라져도 같은 위치를 가리킵니다.
    """
    if not path:
        return path

    normalized = str(path).replace("\\", "/")
    if os.path.isabs(normalized):
        return os.path.normpath(normalized)
    return os.path.normpath(os.path.join(PROJECT_ROOT, normalized))


def to_project_relative(path):
    """저장소 안의 경로를 '/' 구분자의 상대 경로로 돌려줍니다.

    저장소 밖이거나 드라이브가 다르면 절대 경로를 그대로 둡니다.
    """
    normalized = os.path.abspath(str(path).replace("\\", "/"))
    try:
        relative = os.path.relpath(normalized, PROJECT_ROOT)
    except ValueError:
        return normalized.replace("\\", "/")

    if relative.startswith(".."):
        return normalized.replace("\\", "/")
    return relative.replace("\\", "/")


def resolve_indexed_image_path(metadata_path, project_root=None):
    """메타데이터에 기록된 이미지 경로를 현재 실행 환경의 경로로 바꿉니다.

    build_index.py는 저장소 루트 기준 상대 경로를 기록하므로 보통은 루트만
    앞에 붙이면 됩니다. 다른 머신에서 만든 과거 인덱스는 그 머신의 절대 경로를
    담고 있어, 파일이 실제로 없을 때만 저장소 이름 뒤쪽을 떼어 현재 루트에
    다시 붙입니다. 윈도우(c:/...)와 WSL(/mnt/c/...) 경로가 같은 방식으로
    처리됩니다.

    이미지를 저장소 밖에 두었다면 FASHION_PROJECT_ROOT로 기준을 바꿉니다.
    """
    if project_root is None:
        project_root = os.environ.get("FASHION_PROJECT_ROOT", PROJECT_ROOT)

    normalized = str(metadata_path).replace("\\", "/")

    # 지금 환경에서 그대로 열리면 더 볼 것이 없습니다.
    if os.path.isfile(normalized):
        return os.path.normpath(normalized)

    # 다른 머신에서 만든 인덱스.
    marker = "/{}/".format(os.path.basename(PROJECT_ROOT)).lower()
    lowered = normalized.lower()
    if marker in lowered:
        tail = normalized[lowered.rindex(marker) + len(marker):]
        return os.path.normpath(os.path.join(project_root, *tail.split("/")))

    # 저장소 루트 기준 상대 경로(현재 build_index.py가 기록하는 형식).
    # 드라이브 문자가 없고 '/'로 시작하지도 않을 때만 해당합니다.
    if not os.path.isabs(normalized) and not normalized.startswith("/"):
        return os.path.normpath(os.path.join(project_root, *normalized.split("/")))

    return os.path.normpath(normalized)


def load_config(config_path=None):
    """YAML 설정을 읽고 paths 항목을 저장소 루트 기준으로 해석합니다."""
    if config_path is None or not os.path.exists(config_path):
        config_path = DEFAULT_CONFIG_PATH

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    paths = config.get("paths", {})
    for key in _RESOLVED_PATH_KEYS:
        if key in paths:
            paths[key] = resolve_path(paths[key])

    return config
