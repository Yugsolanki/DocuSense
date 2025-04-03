
def get_text_and_last_paragraph(text):
    """
    Extracts the text and the last paragraph from the given text. 
    The last paragraph is defined as the text after the last double newline character.

    Args:
        text (str): The input text from which to extract the last paragraph.

    Returns:
        tuple: A tuple containing the cleaned text and the last paragraph.
    """
    cleaned_text = text.strip()
    paragraphs = cleaned_text.split('\n\n')
    last_para = paragraphs[-1].strip() if paragraphs else ""
    return [cleaned_text, last_para]
