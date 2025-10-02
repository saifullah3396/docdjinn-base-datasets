from pathlib import Path

import numpy as np


def validate_image_configuration(
    target_image_height: int, target_image_width: int, image_size_divisibility: int = 64
) -> None:
    divisable_msg = f"should be divisable by {image_size_divisibility}"
    assert target_image_width % image_size_divisibility == 0, (
        f"Incorect width: {target_image_width}, {divisable_msg}"
    )
    assert target_image_height % image_size_divisibility == 0, (
        f"Incorect max height size: {target_image_height}, {divisable_msg}"
    )


def get_image(
    image_file_path: str,
    feature: object,
    target_image_height: int,
    target_image_width: int,
    target_image_channels: int,
    image_size_divisibility: int = 64,
) -> np.ndarray:
    validate_image_configuration(
        target_image_height=target_image_height,
        target_image_width=target_image_width,
        image_size_divisibility=image_size_divisibility,
    )
    images: list[np.ndarray] = []
    if image_file_path:
        images.extend(
            read_real_images(
                image_file_path,
                feature,
                target_image_height=target_image_height,
                target_image_width=target_image_width,
                target_image_channels=target_image_channels,
            )
        )
    else:
        # do not waste memory for empty images and create 1px height image
        images.append(
            create_dummy_image(
                target_image_height=target_image_height,
                target_image_width=target_image_width,
                target_image_channels=target_image_channels,
            )
        )

    # simply to single image for usage in this case
    return images[0]


def read_real_images(
    image_file_path: str,
    feature: object,
    target_image_height: int,
    target_image_width: int,
    target_image_channels: int,
) -> list[np.ndarray]:
    mask = feature.seg_data["pages"]["masks"]
    num_pages = feature.seg_data["pages"]["ordinals"]
    page_sizes = feature.seg_data["pages"]["bboxes"]
    page_sizes = page_sizes[mask].tolist()
    page_lst = num_pages[mask].tolist()
    return [
        get_page_image(
            image_file_path,
            page_no,
            page_size,
            target_image_height=target_image_height,
            target_image_width=target_image_width,
            target_image_channels=target_image_channels,
        )
        for page_no, page_size in zip(page_lst, page_sizes, strict=True)
    ]


def get_page_image(
    image_file_path: str,
    page_no: int,
    page_size: list,
    target_image_height: int,
    target_image_width: int,
    target_image_channels: int,
) -> np.ndarray:
    page_path: Path = Path(image_file_path) / f"{page_no}.png"
    if page_path.is_file():
        return load_image(
            page_path,
            target_image_height=target_image_height,
            target_image_width=target_image_width,
            target_image_channels=target_image_channels,
        )
    else:
        return create_dummy_image(
            target_image_height=target_image_height,
            target_image_width=target_image_width,
            target_image_channels=target_image_channels,
        )


def create_dummy_image(
    target_image_height: int, target_image_width: int, target_image_channels: int
) -> np.ndarray:
    arr_sz = (
        (target_image_height, target_image_width, 3)
        if target_image_channels == 3
        else (target_image_height, target_image_width)
    )
    return np.full(arr_sz, 255, dtype=np.uint8)


def load_image(
    page_path: Path,
    target_image_height: int,
    target_image_width: int,
    target_image_channels: int,
) -> np.ndarray:
    from PIL import Image

    image = Image.open(page_path)
    if image.mode != "RGB" and target_image_channels == 3:
        image = image.convert("RGB")
    if image.mode != "L" and target_image_channels == 1:
        image = image.convert("L")
    return np.array(image.resize((target_image_width, target_image_height)))
