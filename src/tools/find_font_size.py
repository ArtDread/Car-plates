from PIL import Image, ImageDraw, ImageFont


def find_font_size(text: str, font: str, image: Image.Image, target_width_ratio: float):
    """Find the appropriate font size for display car plate text.

    Define font size for the image under the assumption that
    it's going to be a linear change in size of the font - it's not,
    but for our purpose the error within a few procent is okay.

    Args:
        text: A car plate text
        font: A filename containing a TrueType font
        image: An image likely including car plate
        target_width_ratio: Portion of image width that text width to be

    Returns:
        The estimated font size

    """
    tested_font_size = 100
    tested_font = ImageFont.truetype(font, tested_font_size)
    observed_width = get_text_size(text, image, tested_font)
    estimated_font_size = (
        tested_font_size * image.width * target_width_ratio / observed_width
    )
    return round(estimated_font_size)


def get_text_size(text: str, image: Image.Image, font: ImageFont.ImageFont):
    """Get the width of the text with certain font.

    Args:
        text: A car plate text
        image: An image likely including car plate
        font: A font object with the specific font size

    Returns:
        The estimated text width

    """
    dummy_image = Image.new("RGB", (image.width, image.height))
    draw = ImageDraw.Draw(dummy_image)
    return draw.textlength(text, font)
