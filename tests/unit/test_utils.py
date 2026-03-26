from src.core.utils import clean_pdf_text

def test_clean_pdf_text():
    text = "This is a \n\n sentence with  extra spaces and \n a page number \n 123 \n at the end."
    cleaned = clean_pdf_text(text)
    
    # Check extra spaces are removed
    assert "  " not in cleaned
    # Check page numbers are removed (based on the regex in the code)
    assert "123" not in cleaned
    # Check newlines are handled
    assert "\n" not in cleaned
    assert "sentence with extra spaces" in cleaned

def test_clean_pdf_text_hyphens():
    text = "This is a multi-\nline word."
    cleaned = clean_pdf_text(text)
    assert "multiline" in cleaned
