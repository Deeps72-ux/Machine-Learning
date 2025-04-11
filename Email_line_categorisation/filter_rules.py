# Auto-generated keyword filter
def is_urgent(text):
    text = text.lower()
    if 'today' in text:
        return True
    if 'tomorrow' in text:
        return True
    if 'by' in text:
        return True
    if 'need' in text:
        return True
    return False
