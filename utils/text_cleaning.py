import re


def clean_markdown_for_tts(markdown_text):
    """Clean markdown syntax for TTS processing"""
    # Remove heading markers (# symbols)
    text = re.sub(r'#+\s+', '', markdown_text)

    # Remove bold/italic markers
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)

    # Remove link syntax but keep link text
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)

    # Remove bullet points but keep text
    text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)

    # Replace ordered list markers (1., 2., etc.) with just their content
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)

    # Remove code blocks and their syntax
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    text = re.sub(r'`(.*?)`', r'\1', text)

    # Remove horizontal rules
    text = re.sub(r'---+', '', text)

    # Remove blockquote markers
    text = re.sub(r'^\s*>\s+', '', text, flags=re.MULTILINE)

    # Normalize multiple newlines to just two
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text
